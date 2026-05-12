"""
Caption ranking: dual cross-encoder ensemble (STS-B + MS-MARCO).

STS-B is trained for semantic similarity; MS-MARCO for query--passage
relevance. Both match ``query text ↔ English image caption`` far better than
NLI models, which target logical entailment between short sentences and do not
preserve the ranking signal needed here.

Figurative rows use de_idiomatized_sentence + context_visual heavy weights;
literal rows emphasize compound_visual + MS-MARCO.
"""

import logging
import math
import re
from typing import Dict, List, Tuple

from sentence_transformers import CrossEncoder

from src.config import CROSS_ENCODER_MODEL, DEVICE, MSMARCO_MODEL

logger = logging.getLogger(__name__)

_LOW_CONFIDENCE_THRESHOLD = 0.08

def _softmax(scores: List[float]) -> List[float]:
    max_s = max(scores)
    exps = [math.exp(s - max_s) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]

def _truncate_caption(caption: str, max_sentences: int = 4) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", caption.strip())
    return " ".join(sentences[:max_sentences])

class CaptionRanker:
    """Scores captions with STS-B + MS-MARCO, usage-dependent fusion."""

    _WEIGHTS_FIGURATIVE = {
        "de_idiomatized_sentence": 0.25,
        "english_translation": 0.05,
        "literal_meaning": 0.15,
        "context_visual": 0.55,
        "compound_visual": 0.00,
    }
    _WEIGHTS_LITERAL = {
        "english_translation": 0.10,
        "literal_meaning": 0.25,
        "context_visual": 0.20,
        "compound_visual": 0.45,
    }
    _UNIFIED_QUERY_WEIGHT = 0.10
    _MODEL_WEIGHTS_FIGURATIVE = {"stsb": 0.80, "msmarco": 0.20}
    _MODEL_WEIGHTS_LITERAL = {"stsb": 0.30, "msmarco": 0.70}

    def __init__(
        self,
        stsb_model: str = CROSS_ENCODER_MODEL,
        msmarco_model: str = MSMARCO_MODEL,
        device: str = DEVICE,
    ) -> None:
        logger.info("Loading STS-B cross-encoder: %s  (device=%s)", stsb_model, device)
        self._stsb = CrossEncoder(stsb_model, device=device)
        logger.info("Loading MS-MARCO cross-encoder: %s  (device=%s)", msmarco_model, device)
        self._msmarco = CrossEncoder(msmarco_model, device=device)
        logger.info("Cross-encoders loaded")

    def _score_ensemble(
        self, query: str, captions: List[str], usage: str = "figurative",
    ) -> List[float]:
        short_caps = [_truncate_caption(c) for c in captions]
        pairs = [[query, cap] for cap in short_caps]
        stsb_raw = self._stsb.predict(pairs).tolist()
        marco_raw = self._msmarco.predict(pairs).tolist()
        stsb_norm = _softmax(stsb_raw)
        marco_norm = _softmax(marco_raw)
        mw = (
            self._MODEL_WEIGHTS_LITERAL if usage == "literal"
            else self._MODEL_WEIGHTS_FIGURATIVE
        )
        w_s, w_m = mw["stsb"], mw["msmarco"]
        return [w_s * s + w_m * m for s, m in zip(stsb_norm, marco_norm)]

    def rank_captions(
        self,
        unpacked_dict: Dict[str, str],
        captions: List[str],
        image_names: List[str],
    ) -> Tuple[int, List[float], str, bool, float]:
        usage = unpacked_dict.get("usage_type", "figurative")
        weights = (
            self._WEIGHTS_LITERAL if usage == "literal"
            else self._WEIGHTS_FIGURATIVE
        )

        scores_trans = self._score_ensemble(
            unpacked_dict["english_translation"], captions, usage,
        )
        scores_literal = self._score_ensemble(
            unpacked_dict["literal_meaning"], captions, usage,
        )
        scores_context = self._score_ensemble(
            unpacked_dict["context_visual"], captions, usage,
        )

        scores_compound: List[float] | None = None
        if weights["compound_visual"] > 0:
            scores_compound = self._score_ensemble(
                unpacked_dict["compound_visual"], captions, usage,
            )

        scores_deidiom: List[float] | None = None
        de_idiom_text = unpacked_dict.get("de_idiomatized_sentence", "")
        if usage == "figurative" and de_idiom_text:
            scores_deidiom = self._score_ensemble(de_idiom_text, captions, usage)
            logger.info("de_idiom query: %.80s…", de_idiom_text)

        if usage == "figurative":
            unified_query = (
                f"This is a metaphorical scene: {unpacked_dict['context_visual']}. "
                f"Intended meaning: {de_idiom_text or unpacked_dict['literal_meaning']}."
            )
        else:
            unified_query = (
                f"A literal physical scene showing {unpacked_dict['compound_visual']}. "
                f"Plain meaning: {unpacked_dict['literal_meaning']}."
            )
        scores_unified = self._score_ensemble(unified_query, captions, usage)

        combined: List[float] = []
        for j in range(len(captions)):
            score = (
                weights["english_translation"] * scores_trans[j]
                + weights["literal_meaning"] * scores_literal[j]
                + weights["context_visual"] * scores_context[j]
            )
            if scores_compound is not None:
                score += weights["compound_visual"] * scores_compound[j]
            if scores_deidiom is not None:
                score += weights["de_idiomatized_sentence"] * scores_deidiom[j]
            score += self._UNIFIED_QUERY_WEIGHT * scores_unified[j]
            combined.append(score)

        probabilities = _softmax(combined)
        mean_prob = sum(probabilities) / len(probabilities)
        max_prob = max(probabilities)
        prob_gap_pre_fallback = max_prob - mean_prob
        low_confidence_used = prob_gap_pre_fallback < _LOW_CONFIDENCE_THRESHOLD

        if low_confidence_used:
            logger.warning(
                "LOW CONFIDENCE (gap=%.4f) -- re-scoring",
                prob_gap_pre_fallback,
            )
            if usage == "literal" and scores_compound is not None:
                fallback_signal = scores_compound
            else:
                fallback_signal = scores_context
            fallback = _softmax(fallback_signal)
            probabilities = [
                0.25 * p + 0.75 * f for p, f in zip(probabilities, fallback)
            ]

        ranked_indices = sorted(
            range(len(probabilities)),
            key=lambda i: probabilities[i],
            reverse=True,
        )
        best_index = ranked_indices[0] + 1
        expected_order = ", ".join(image_names[i] for i in ranked_indices)

        logger.info(
            "Ranked  |  usage=%s  |  best=#%d (p=%.3f)  |  order=%s",
            usage,
            best_index,
            probabilities[ranked_indices[0]],
            expected_order,
        )

        return (
            best_index,
            probabilities,
            expected_order,
            low_confidence_used,
            prob_gap_pre_fallback,
        )
