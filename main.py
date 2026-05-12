"""
Entry point for the AdMIRe 2.0 Text-Only Bottleneck pipeline.

Usage
-----
    cd admire_text_bottleneck

    # Default: process ALL languages sequentially and export submissions
    python main.py

    # Mini-test on Turkish (30 rows):
    python main.py --lang Turkish

    # Chinese 20 + Turkish 30 in one run (ranker loaded once):
    python main.py --mini-benchmarks
    # Second-stage LLM: same top-3 AdMIRe judge rerank when (a) figurative + flat
    # top-3 CE probs, or (b) literal + ranker low-confidence. Env:
    # LLM_TOP3_RERANK_ENABLED=0  LLM_TOP3_PROB_SPREAD_MAX=0.06

    # Process a fixed list of languages (full rows, predictions + submissions under outputs/):
    python main.py --langs Turkish,Georgian,Greek,Russian,Serbian,Slovak --full --export-submission

    # Process a single file fully:
    python main.py --input ../data/submission_Turkish.tsv --full

    # Process ALL language files:
    python main.py --data-dir ../data --full
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from src.pipeline import AdmirePipeline
from src.evaluate import evaluate
from src.config import DATA_DIR

_BENCHMARKS_DIR = Path(__file__).parent / "benchmarks"

_LANG_BENCHMARKS = {
    "chinese": ("ground_truth_chinese_20.json", 20),
    "turkish": ("ground_truth_turkish_30.json", 30),
}

_LANG_TO_SUBMISSION_CODE: Dict[str, str] = {
    "chinese": "zh",
    "georgian": "ka",
    "greek": "el",
    "igbo": "ig",
    "kazakh": "kk",
    "norwegian": "no",
    "portuguese-brazil": "pt-br",
    "portuguese-portugal": "pt-pt",
    "russian": "ru",
    "serbian": "sr",
    "slovak": "sk",
    "slovenian": "sl",
    "spanish-ecuador": "es-ec",
    "turkish": "tr",
    "uzbek": "uz",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("admire")

def _to_expected_order_list_literal(order_csv: str) -> str:
    names = [x.strip() for x in order_csv.split(",") if x.strip()]
    return "[" + ", ".join(f"'{n}'" for n in names) + "]"

def _export_submission(
    df,
    submission_code: str,
    output_dir: Path,
) -> Path:
    code = submission_code.strip().lower()
    required = ("compound", "expected_order", "sentence")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Cannot export submission: predictions are missing columns "
            f"{missing!r}; AdMIRe Task A requires {list(required)}."
        )
    sub_df = df[list(required)].copy()
    sub_df["expected_order"] = sub_df["expected_order"].map(_to_expected_order_list_literal)
    out_path = output_dir / f"submission_{code}.tsv"
    output_dir.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(out_path, sep="\t", index=False)
    logger.info("Submission file exported -> %s", out_path)
    print(f"  -> Submission exported to {out_path}")
    return out_path

def _norm_lang_stem(s: str) -> str:
    return s.strip().lower().replace(" ", "")

def _select_submission_files(
    data_dir: Path,
    langs_csv: str,
) -> List[Tuple[Path, str]]:
    """Return (path, stem) pairs in the order given in *langs_csv* (e.g. Turkish,Georgian)."""
    want = [_norm_lang_stem(x) for x in langs_csv.split(",") if x.strip()]
    if not want:
        raise ValueError("--langs is empty")

    by_norm: Dict[str, Tuple[Path, str]] = {}
    for p in sorted(data_dir.glob("submission_*.tsv")):
        stem = p.stem.replace("submission_", "")
        by_norm[_norm_lang_stem(stem)] = (p, stem)

    selected: List[Tuple[Path, str]] = []
    missing: List[str] = []
    for w in want:
        if w in by_norm:
            selected.append(by_norm[w])
        else:
            missing.append(w)
    if missing:
        avail = ", ".join(sorted(s for _, s in by_norm.values()))
        raise FileNotFoundError(
            f"No submission_* file for: {missing!r} under {data_dir}. "
            f"Available stems: {avail}",
        )
    return selected

def main() -> None:
    """Parse CLI arguments and launch the pipeline."""
    parser = argparse.ArgumentParser(
        description="AdMIRe 2.0 - Text-Only Multilingual Bottleneck",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to a single input TSV file.",
    )
    group.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory with submission_*.tsv files (processes all).",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help="Language for mini-test mode (e.g. Chinese, Turkish). "
             "If omitted, runs all languages.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output TSV path (only used with --input).",
    )
    parser.add_argument(
        "--mini-benchmarks",
        action="store_true",
        help="Run Chinese (20) + Turkish (30) mini tests with nDCG benchmarks; "
        "writes outputs/mini_predictions_Chinese.tsv and mini_predictions_Turkish.tsv.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Process all rows (no mini-test limit).",
    )
    parser.add_argument(
        "--export-submission",
        action="store_true",
        help="Export CodaBench-style submission_XX.tsv after run.",
    )
    parser.add_argument(
        "--langs",
        type=str,
        default=None,
        help="Comma-separated submission_* stems (e.g. Turkish,Georgian,Greek). "
        "Writes outputs/predictions_<Stem>.tsv; use --full for all rows. "
        "Uses DATA_DIR or --data-dir.",
    )
    parser.add_argument(
        "--submission-output-dir",
        type=str,
        default="outputs",
        help="Directory for submission_*.tsv when using --langs --export-submission "
        "(default: outputs, relative to admire_text_bottleneck/).",
    )
    parser.add_argument(
        "--submission-code",
        type=str,
        default=None,
        help="Override submission language tag (e.g. tr, zh, pt-br); lowercased in filename.",
    )
    args = parser.parse_args()

    if args.langs and args.input:
        parser.error("--langs cannot be used with --input")

    if args.mini_benchmarks:
        logger.info("=== MINI BENCHMARK SUITE  |  Chinese 20 + Turkish 30 ===")
        pipeline = AdmirePipeline()
        dfs = pipeline.run_mini_benchmarks()
        for lang_key, df in dfs.items():
            info = _LANG_BENCHMARKS.get(lang_key)
            if info:
                evaluate(df, _BENCHMARKS_DIR / info[0], limit=info[1])
            else:
                logger.info("No benchmark JSON for %s", lang_key)
        print("\nMini suite done. Predictions under outputs/mini_predictions_*.tsv")
        return

    if args.langs:
        data_dir = Path(args.data_dir) if args.data_dir else Path(DATA_DIR)
        if not data_dir.is_dir():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        pairs = _select_submission_files(data_dir, args.langs)
        limit = None if args.full else 20
        out_sub_root = Path(__file__).parent / args.submission_output_dir
        out_sub_root.mkdir(parents=True, exist_ok=True)
        logger.info(
            "=== MULTI-LANG MODE  |  %d file(s) from %s  |  limit=%s ===",
            len(pairs),
            data_dir,
            "all rows" if limit is None else f"first {limit}",
        )
        for tsv_path, stem in pairs:
            stem_lower = stem.lower()
            pred_out = Path("outputs") / f"predictions_{stem}.tsv"
            logger.info("Language file: %s", tsv_path.name)
            pipeline = AdmirePipeline(input_path=tsv_path, output_path=pred_out)
            df = pipeline.run(limit=limit)
            print(f"\n--- {stem} ({len(df)} rows) ---")
            print(df[["compound", "predicted_index", "expected_order"]].to_string(index=False))
            info = _LANG_BENCHMARKS.get(stem_lower)
            if info:
                evaluate(df, _BENCHMARKS_DIR / info[0], limit=limit)
            if args.export_submission:
                code = args.submission_code or _LANG_TO_SUBMISSION_CODE.get(stem_lower)
                if not code:
                    logger.warning(
                        "No submission code for '%s'; use --submission-code or extend "
                        "_LANG_TO_SUBMISSION_CODE",
                        stem_lower,
                    )
                else:
                    _export_submission(df, code, out_sub_root)
        print("\nDone. Predictions under outputs/predictions_<Lang>.tsv")
        if args.export_submission:
            print(f"Submissions under {out_sub_root}/submission_<code>.tsv")
        return

    if args.data_dir:
        logger.info("=== BATCH MODE  |  directory=%s ===", args.data_dir)
        pipeline = AdmirePipeline()
        df = pipeline.run_all(args.data_dir)
        print(f"\nTotal rows processed: {len(df)}")
        print(df[["compound", "predicted_index", "expected_order"]].to_string(index=False))
        if args.export_submission:
            print(
                "Batch mode writes multiple languages; "
                "use single-file or --lang mode for one submission file.",
            )

    elif args.input:
        out = args.output or "outputs/predictions.tsv"
        limit = None if args.full else 20
        logger.info("=== SINGLE FILE MODE  |  input=%s ===", args.input)
        pipeline = AdmirePipeline(input_path=args.input, output_path=out)
        df = pipeline.run(limit=limit)
        print(f"\nTotal rows processed: {len(df)}")
        print(df[["compound", "predicted_index", "expected_order"]].to_string(index=False))
        stem = Path(args.input).stem.replace("submission_", "").lower()
        info = _LANG_BENCHMARKS.get(stem)
        if info:
            evaluate(df, _BENCHMARKS_DIR / info[0], limit=limit)
        if args.export_submission:
            code = args.submission_code or _LANG_TO_SUBMISSION_CODE.get(stem)
            if not code:
                logger.warning(
                    "No submission code mapping for '%s'; provide --submission-code",
                    stem,
                )
            else:
                _export_submission(df, code, Path(__file__).parent)

    else:
        if args.lang is None:
            logger.info("=== DEFAULT MODE  |  ALL LANGUAGES sequential ===")
            data_dir = Path(DATA_DIR)
            tsv_files = sorted(data_dir.glob("submission_*.tsv"))
            if not tsv_files:
                raise FileNotFoundError(f"No submission_*.tsv files in {data_dir}")

            for tsv_file in tsv_files:
                stem = tsv_file.stem.replace("submission_", "")
                stem_lower = stem.lower()
                out = Path("outputs") / f"predictions_{stem}.tsv"
                logger.info("Language file: %s", tsv_file.name)
                pipeline = AdmirePipeline(input_path=tsv_file, output_path=out)
                df = pipeline.run(limit=None)

                info = _LANG_BENCHMARKS.get(stem_lower)
                if info:
                    evaluate(df, _BENCHMARKS_DIR / info[0], limit=None)

                code = _LANG_TO_SUBMISSION_CODE.get(stem_lower)
                if code:
                    _export_submission(df, code, Path(__file__).parent)
                else:
                    logger.warning(
                        "No submission code mapping for '%s' (file=%s)",
                        stem_lower,
                        tsv_file.name,
                    )
            return

        lang = args.lang
        lang_lower = lang.lower()
        info = _LANG_BENCHMARKS.get(lang_lower)
        limit = None if args.full else (info[1] if info else 20)
        input_path = Path(DATA_DIR) / f"submission_{lang}.tsv"
        out = "mini_test_output.tsv"
        logger.info("=== MINI-TEST MODE  |  lang=%s  |  first %d rows ===", lang, limit)
        pipeline = AdmirePipeline(input_path=input_path, output_path=out)
        df = pipeline.run(limit=limit)
        print(f"\nTotal rows processed: {len(df)}")
        print(df[["compound", "predicted_index", "expected_order"]].to_string(index=False))
        if info:
            evaluate(df, _BENCHMARKS_DIR / info[0], limit=limit)
        else:
            logger.info("No benchmark available for %s", lang)
        if args.export_submission:
            code = args.submission_code or _LANG_TO_SUBMISSION_CODE.get(lang_lower)
            if not code:
                logger.warning(
                    "No submission code mapping for '%s'; provide --submission-code",
                    lang_lower,
                )
            else:
                _export_submission(df, code, Path(__file__).parent)

if __name__ == "__main__":
    main()
