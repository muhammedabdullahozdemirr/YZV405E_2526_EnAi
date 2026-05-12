"""
Shared prompts for LLM-assisted **top-3** caption reranking (pipeline second stage).

Same AdMIRe judge framing as llm-only: literal vs figurative, literal trap,
iconic idiom scenes, long-caption gist; JSON ``ranked_filenames`` (length 3).
"""

from __future__ import annotations

from typing import Dict, List

_ADMIRE_JUDGE_BASE = (
    "You are an expert judge for the AdMIRe benchmark: given a multi-word "
    "COMPOUND (often Chinese or another language), a SENTENCE that uses it, and "
    "English captions (each tied to an image filename), you rank images by "
    "visual fit to how the compound is used in that sentence.\n\n"
    "## How to interpret the compound in context\n"
    "1) Decide if the compound is used LITERALLY (names a real object, concrete "
    "scene, or physical action in the sentence) or FIGURATIVELY (idiom, metaphor, "
    "abstract institutional/process sense).\n"
    "2) LITERAL: the winning image should show the same concrete kind of scene "
    "the sentence describes (objects, materials, layout). Match the *situation*, "
    "not random words from the compound if they are not what the sentence is "
    "about.\n"
    "3) FIGURATIVE: prefer images that express the *concept* (secrecy, dispute, "
    "being ahead, tiny amount vs whole, danger on a thread, etc.). Strongly "
    "penalize 'literal trap' images that only depict the compound's dictionary "
    "nouns while the sentence is clearly metaphorical (e.g. institutional "
    "'black box' vs a physical black box).\n"
    "4) Idioms sometimes have an 'iconic' literal picture (horse for 'take the "
    "lead', tightrope for 'hair-trigger moment'). If the sentence evokes that "
    "idiom reading, an image that matches that iconic scene can rank high even "
    "under figurative use—use the sentence, not only free association.\n\n"
    "## How to read captions\n"
    "Captions can be long. For each image, extract the main subject, setting, "
    "and action in a few mental phrases; ignore decorative detail. Compare all "
    "given candidates before committing.\n\n"
)

_OUTPUT_TOP3 = (
    "## Output\n"
    "Return one JSON object. Required key:\n"
    '  "ranked_filenames": [ ... exactly 3 strings ... ]\n'
    "These must be exactly the three image filenames given in the user message "
    "(same spelling), each exactly once, best match first, worst last.\n"
    "Optional key (helps you think; keep short): \"reasoning\" — at most 3 "
    "sentences in English.\n"
    "No markdown fences, no text outside the JSON object."
)

ADMIRE_JUDGE_SYSTEM_TOP3: str = _ADMIRE_JUDGE_BASE + _OUTPUT_TOP3

def _candidate_blocks(image_names: List[str], captions: List[str]) -> List[str]:
    blocks: List[str] = []
    for i, (name, cap) in enumerate(zip(image_names, captions), start=1):
        blocks.append(f"Image {i}\nfile={name}\nCaption: {cap}")
    return blocks

def build_user_message_top3(
    compound: str,
    sentence: str,
    image_names: List[str],
    captions: List[str],
    unpacked: Dict[str, str],
    usage_type_label: str,
) -> str:
    """Re-rank three CE-leading candidates (same prompt for figurative spread or literal low-conf)."""
    blocks = _candidate_blocks(image_names, captions)
    hints = (
        "--- Semantic hints (from pipeline unpack; guidance only, do not invent facts) ---\n"
        f"usage_type: {usage_type_label}\n"
        f"english_translation: {unpacked.get('english_translation', '')}\n"
        f"literal_meaning: {unpacked.get('literal_meaning', '')}\n"
        f"context_visual: {unpacked.get('context_visual', '')}\n"
        f"compound_visual: {unpacked.get('compound_visual', '')}\n"
        f"de_idiomatized_sentence: {unpacked.get('de_idiomatized_sentence', '')}\n"
    )
    return (
        "Re-rank ONLY the three candidate images below (best → worst). "
        "A cross-encoder ensemble placed these as the top three but their scores "
        "were very close, or (for literal usage_type) the ranker signalled low "
        "confidence—break the tie using COMPOUND, SENTENCE, hints, and captions. "
        "Reply with JSON only; "
        '"ranked_filenames" must contain exactly those three filenames, best first.\n\n'
        f"COMPOUND: {compound}\n"
        f"SENTENCE: {sentence}\n\n"
        f"{hints}\n"
        "--- Top three candidates (same three files only) ---\n"
        + "\n\n".join(blocks)
    )
