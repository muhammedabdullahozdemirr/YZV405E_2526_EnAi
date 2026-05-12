"""
Multilingual semantic unpacker via OpenAI LLM with structured JSON outputs.

Uses ``response_format`` JSON schema when supported, with fallbacks for models
or API paths that reject strict schema mode.
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from openai import APIError, OpenAI

from src.config import (
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_MIN_REQUEST_INTERVAL_SEC,
    LLM_TEMPERATURE,
    LLM_TOP3_RERANK_MAX_TOKENS,
    OPENAI_API_KEY,
)
from src.llm_rerank_prompts import (
    ADMIRE_JUDGE_SYSTEM_TOP3,
    build_user_message_top3,
)

logger = logging.getLogger(__name__)

_SEMANTIC_UNPACK_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "usage_type": {"type": "string", "enum": ["figurative", "literal"]},
        "english_translation": {"type": "string"},
        "literal_meaning": {"type": "string"},
        "context_visual": {"type": "string"},
        "compound_visual": {"type": "string"},
        "de_idiomatized_sentence": {"type": "string"},
    },
    "required": [
        "usage_type",
        "english_translation",
        "literal_meaning",
        "context_visual",
        "compound_visual",
        "de_idiomatized_sentence",
    ],
    "additionalProperties": False,
}

_RESPONSE_FORMAT_SCHEMA: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "semantic_unpack",
        "strict": True,
        "schema": _SEMANTIC_UNPACK_SCHEMA,
    },
}

_SYSTEM_PROMPT = (
    "You are a multilingual idiom and compound-expression expert.\n\n"
    "COMPOUND and SENTENCE may be in any language; write all JSON string values "
    "in English.\n\n"
    "The user sends COMPOUND (the target phrase) and SENTENCE (full context).\n"
    "Decide usage_type:\n"
    "- literal: the compound names a real physical object or concrete action "
    "in this sentence.\n"
    "- figurative: idiom, metaphor, or abstract use.\n\n"
    "Fill every JSON field:\n"
    "- english_translation: fluent English of the whole sentence.\n"
    "- literal_meaning: plain English message (no metaphor left unexplained).\n"
    "- context_visual: ONE vivid scene (20-35 words) matching the sentence's "
    "core idea. Use one main subject, one primary action, and 2-3 distinctive "
    "objects. If figurative, include mood (tense, secretive, joyful, etc.). "
    "Name concrete people, actions, objects.\n"
    "- compound_visual: ignore context; only the literal picture of the "
    "compound's words (under ~20 words). E.g. black box -> sealed dark box.\n"
    "- de_idiomatized_sentence: ONLY if figurative. Rewrite english_translation "
    "as exactly one sentence so the compound words disappear entirely; replace "
    "with ordinary words for the intended meaning. Do not reuse any word from "
    "the compound phrase. If literal, use empty string \"\".\n\n"
    "Few-shot:\n"
    "Figurative compound 'black box' about AI: de_idiomatized_sentence might be: "
    "\"The system's decisions are opaque and not explainable to outsiders.\"\n"
    "Literal compound 'black box' (flight recorder): de_idiomatized_sentence: \"\"\n\n"
    "Output must be valid JSON only (no markdown fences)."
)

_SYSTEM_PROMPT_SCHEMA = _SYSTEM_PROMPT + "\n\nPopulate every schema field."

class SemanticUnpacker:
    """OpenAI chat completion → structured unpack dict."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = LLM_MODEL,
    ) -> None:
        self._client = OpenAI(api_key=api_key or OPENAI_API_KEY or None)
        self._model = model
        self._last_request_ts = 0.0
        logger.info("SemanticUnpacker  |  model=%s", self._model)

    def unpack(self, sentence: str, compound: str = "") -> Dict[str, str]:
        user_msg = f"COMPOUND: {compound}\nSENTENCE: {sentence}" if compound else sentence
        logger.debug("LLM user message: %.100s…", user_msg)

        raw = self._complete_with_fallback(user_msg)
        parsed = self._parse_json(raw)
        logger.info(
            "Unpacked  |  usage=%s  |  context=%.60s…",
            parsed["usage_type"],
            parsed["context_visual"],
        )
        return parsed

    def rerank_top3(
        self,
        compound: str,
        sentence: str,
        unpacked: Dict[str, str],
        image_names_top3: List[str],
        captions_top3: List[str],
        usage_type_label: str,
    ) -> List[str]:
        """LLM re-order of top-3 CE candidates (figurative flat spread or literal low-conf)."""
        if len(image_names_top3) != 3 or len(captions_top3) != 3:
            logger.warning("rerank_top3: need exactly 3 candidates; skipping")
            return list(image_names_top3)

        user_msg = build_user_message_top3(
            compound,
            sentence,
            image_names_top3,
            captions_top3,
            unpacked,
            usage_type_label,
        )

        self._respect_rate_limit()
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                temperature=LLM_TEMPERATURE,
                max_completion_tokens=LLM_TOP3_RERANK_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": ADMIRE_JUDGE_SYSTEM_TOP3},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("Top-3 LLM rerank failed (%s); keeping ranker order", exc)
            return list(image_names_top3)

        order = self._parse_ranked_filenames_json(
            raw, set(image_names_top3), 3,
        )
        if order is None:
            logger.warning("Top-3 LLM rerank parse failed; keeping ranker order")
            return list(image_names_top3)
        logger.info("Top-3 LLM rerank applied  |  new order=%s", order)
        return order

    @staticmethod
    def _parse_ranked_filenames_json(
        text: str,
        expected: set[str],
        n: int,
    ) -> Optional[List[str]]:
        cleaned = re.sub(r"```(?:json)?\s*", "", text)
        cleaned = cleaned.replace("```", "").strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*}", cleaned, re.DOTALL)
            if not match:
                return None
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return None
        order = data.get("ranked_filenames")
        if not isinstance(order, list) or len(order) != n:
            return None
        names = [str(x).strip() for x in order]
        if set(names) != expected:
            return None
        return names

    def _complete_with_fallback(self, user_msg: str) -> str:
        """Try JSON schema mode, then json_object, then unconstrained text."""
        self._respect_rate_limit()
        attempts = [
            self._request_schema,
            self._request_json_object,
            self._request_plain,
        ]
        last_err: Optional[Exception] = None
        for fn in attempts:
            try:
                return fn(user_msg)
            except APIError as exc:
                last_err = exc
                logger.warning("OpenAI API error (%s), trying fallback LLM path", exc)
            except Exception as exc:
                last_err = exc
                logger.warning("LLM request failed (%s), trying fallback", exc)
        if last_err:
            raise last_err
        return ""

    def _respect_rate_limit(self) -> None:
        """Enforce minimum spacing between LLM requests (disabled when interval <= 0)."""
        if LLM_MIN_REQUEST_INTERVAL_SEC <= 0:
            self._last_request_ts = time.monotonic()
            return
        now = time.monotonic()
        elapsed = now - self._last_request_ts
        if elapsed < LLM_MIN_REQUEST_INTERVAL_SEC:
            sleep_s = LLM_MIN_REQUEST_INTERVAL_SEC - elapsed
            logger.info("Rate-limit guard: sleeping %.2fs before LLM call", sleep_s)
            time.sleep(sleep_s)
        self._last_request_ts = time.monotonic()

    def _request_schema(self, user_msg: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=LLM_TEMPERATURE,
            max_completion_tokens=LLM_MAX_TOKENS,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT_SCHEMA},
                {"role": "user", "content": user_msg},
            ],
            response_format=_RESPONSE_FORMAT_SCHEMA,
        )
        return resp.choices[0].message.content or ""

    def _request_json_object(self, user_msg: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=LLM_TEMPERATURE,
            max_completion_tokens=LLM_MAX_TOKENS,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""

    def _request_plain(self, user_msg: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=LLM_TEMPERATURE,
            max_completion_tokens=LLM_MAX_TOKENS,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        return resp.choices[0].message.content or ""

    @staticmethod
    def _parse_json(text: str) -> Dict[str, str]:
        cleaned = re.sub(r"```(?:json)?\s*", "", text)
        cleaned = cleaned.replace("```", "").strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*}", cleaned, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.warning("JSON parse failed after regex extraction")
                    data = {}
            else:
                logger.warning("No JSON object in LLM response")
                data = {}

        return {
            "usage_type": data.get("usage_type", "figurative"),
            "english_translation": data.get("english_translation", ""),
            "literal_meaning": data.get("literal_meaning", ""),
            "context_visual": data.get("context_visual", ""),
            "compound_visual": data.get("compound_visual", ""),
            "de_idiomatized_sentence": data.get("de_idiomatized_sentence", ""),
        }
