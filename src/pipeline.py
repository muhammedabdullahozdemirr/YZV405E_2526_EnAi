"""
End-to-end orchestration pipeline for AdMIRe 2.0.

Connects the TSV data loader, OpenAI LLM unpacker (JSON schema when available),
and dual cross-encoder ranker (STS-B + MS-MARCO) into a single
callable workflow that produces a predictions TSV.  Supports single-file,
batch (directory), and mini-test modes.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from tqdm import tqdm

from src.data_loader import TSVLoader
from src.llm_unpacker import SemanticUnpacker
from src.ranker import CaptionRanker
from src.config import (
    DATA_DIR,
    DEFAULT_INPUT_TSV,
    DEFAULT_OUTPUT_TSV,
    LLM_TOP3_PROB_SPREAD_MAX,
    LLM_TOP3_RERANK_ENABLED,
    OPENAI_API_KEY,
    OUTPUT_DIR,
)

logger = logging.getLogger(__name__)

class AdmirePipeline:
    """Orchestrates the full AdMIRe 2.0 text-only prediction flow."""

    def __init__(
        self,
        input_path: str | Path = DEFAULT_INPUT_TSV,
        output_path: str | Path = DEFAULT_OUTPUT_TSV,
    ) -> None:
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

        logger.info("Initializing SemanticUnpacker (OpenAI LLM)…")
        self._unpacker = SemanticUnpacker()

        logger.info("Initializing CaptionRanker (Cross-Encoder)…")
        self._ranker = CaptionRanker()

    def _process_rows(
        self,
        loader: TSVLoader,
        label: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Iterate over loader rows, unpack and rank each one."""
        total = min(limit, len(loader)) if limit else len(loader)
        results: List[Dict[str, Any]] = []

        for i, row in enumerate(tqdm(loader, total=total, desc=f"[{label}]")):
            if limit and i >= limit:
                break

            sentence: str = row["sentence"]
            captions: List[str] = row["captions"]
            image_names: List[str] = row["image_names"]

            compound: str = row["compound"]
            logger.info("Row %d  |  compound=%s", i + 1, compound)

            unpacked: Dict[str, str] = self._unpacker.unpack(sentence, compound=compound)
            (
                best_idx,
                scores,
                expected_order,
                low_confidence_used,
                prob_gap_pre_fallback,
            ) = self._ranker.rank_captions(
                unpacked, captions, image_names,
            )

            top3_prob_spread: Optional[float] = None
            llm_rerank_invoked = False
            llm_rerank_kind = "none"
            ranked_idx = sorted(
                range(len(scores)), key=lambda j: scores[j], reverse=True,
            )
            if len(scores) >= 3:
                top3_prob_spread = float(
                    scores[ranked_idx[0]] - scores[ranked_idx[2]],
                )

            usage_l = str(unpacked.get("usage_type", "")).strip().lower()
            trigger_fig = (
                usage_l == "figurative"
                and top3_prob_spread is not None
                and top3_prob_spread <= LLM_TOP3_PROB_SPREAD_MAX
            )
            trigger_lit = usage_l == "literal" and low_confidence_used

            if (
                LLM_TOP3_RERANK_ENABLED
                and OPENAI_API_KEY.strip()
                and (trigger_fig or trigger_lit)
            ):
                top3_names = [image_names[j] for j in ranked_idx[:3]]
                top3_caps = [captions[j] for j in ranked_idx[:3]]
                new3 = self._unpacker.rerank_top3(
                    compound,
                    sentence,
                    unpacked,
                    top3_names,
                    top3_caps,
                    usage_type_label=usage_l,
                )
                llm_rerank_invoked = True
                llm_rerank_kind = (
                    "top3_figurative_spread" if trigger_fig else "top3_literal_lowconf"
                )
                rest_names = [image_names[j] for j in ranked_idx[3:5]]
                full_names = new3 + rest_names
                expected_order = ", ".join(full_names)
                best_idx = image_names.index(full_names[0]) + 1

            results.append(
                {
                    "compound": row["compound"],
                    "sentence": sentence,
                    "usage_type": unpacked["usage_type"],
                    "english_translation": unpacked["english_translation"],
                    "literal_meaning": unpacked["literal_meaning"],
                    "context_visual": unpacked["context_visual"],
                    "compound_visual": unpacked["compound_visual"],
                    "de_idiomatized_sentence": unpacked.get("de_idiomatized_sentence", ""),
                    "predicted_index": best_idx,
                    "expected_order": expected_order,
                    "scores": scores,
                    "low_confidence_used": low_confidence_used,
                    "prob_gap_pre_fallback": prob_gap_pre_fallback,
                    "top3_prob_spread": top3_prob_spread,
                    "llm_rerank_invoked": llm_rerank_invoked,
                    "llm_rerank_kind": llm_rerank_kind,
                }
            )

        return results

    def _save(self, results: List[Dict[str, Any]], out_path: Path) -> pd.DataFrame:
        """Write results to a TSV and return as DataFrame."""
        df = pd.DataFrame(results)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, sep="\t", index=False)
        logger.info("Saved %d predictions -> %s", len(df), out_path)
        print(f"  -> Saved {len(df)} predictions to {out_path}")
        return df

    def run(
        self,
        limit: Optional[int] = None,
        input_path: Optional[str | Path] = None,
        output_path: Optional[str | Path] = None,
    ) -> pd.DataFrame:
        """Process one TSV."""
        in_path = Path(input_path) if input_path is not None else self.input_path
        out_path = Path(output_path) if output_path is not None else self.output_path
        logger.info("Loading TSV: %s", in_path)
        loader = TSVLoader(in_path)
        lang = in_path.stem.replace("submission_", "")
        results = self._process_rows(loader, lang, limit=limit)
        return self._save(results, out_path)

    def run_mini_benchmarks(
        self,
        data_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> Dict[str, pd.DataFrame]:
        """Process Chinese (20 rows) + Turkish (30 rows) mini suite."""
        root = Path(data_dir) if data_dir is not None else Path(DATA_DIR)
        out_dir = Path(output_dir) if output_dir is not None else Path(OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)

        specs: List[tuple[str, int, str]] = [
            ("Chinese", 20, "mini_predictions_Chinese.tsv"),
            ("Turkish", 30, "mini_predictions_Turkish.tsv"),
        ]
        dfs: Dict[str, pd.DataFrame] = {}
        for lang, row_limit, out_name in specs:
            tsv = root / f"submission_{lang}.tsv"
            if not tsv.is_file():
                raise FileNotFoundError(f"Mini benchmark input missing: {tsv}")
            logger.info("Mini benchmark  |  %s  |  first %d rows", lang, row_limit)
            df = self.run(
                limit=row_limit,
                input_path=tsv,
                output_path=out_dir / out_name,
            )
            dfs[lang.lower()] = df
        return dfs

    def run_all(self, data_dir: str | Path) -> pd.DataFrame:
        """Process every submission_*.tsv in data_dir."""
        data_dir = Path(data_dir)
        tsv_files = sorted(data_dir.glob("submission_*.tsv"))

        if not tsv_files:
            raise FileNotFoundError(f"No submission_*.tsv files in {data_dir}")

        logger.info("Found %d language file(s) in %s", len(tsv_files), data_dir)
        print(f"Found {len(tsv_files)} language file(s) in {data_dir}\n")

        out_dir = Path(OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)

        all_dfs: List[pd.DataFrame] = []

        for tsv_file in tsv_files:
            lang = tsv_file.stem.replace("submission_", "")
            out_path = out_dir / f"predictions_{lang}.tsv"
            logger.info("Processing language: %s", lang)
            loader = TSVLoader(tsv_file)
            results = self._process_rows(loader, lang)
            df = self._save(results, out_path)
            all_dfs.append(df)

        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = out_dir / "all_predictions.tsv"
        combined.to_csv(combined_path, sep="\t", index=False)
        logger.info("Combined %d rows -> %s", len(combined), combined_path)
        print(f"\nCombined predictions ({len(combined)} rows) saved to {combined_path}")
        return combined
