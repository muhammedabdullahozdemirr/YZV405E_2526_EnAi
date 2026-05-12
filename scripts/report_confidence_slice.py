"""Analyze confidence / Top-1 on saved mini predictions.

Two modes:

1. ``--posthoc`` (default): stdlib only — reads ``expected_order`` + ``scores`` from
   TSV vs ground truth. Reports Top-1 vs **final** probability *top1−top2 margin*
   (not the same as ranker pre-fallback ``low_confidence_used``, but answers
   “MISS’ler ince marjda mı?”). Also prints **where gold Top-1 sits** in the
   predicted order (counts for ranks 1–5 and recall@k).

2. ``--replay-ranker``: loads cross-encoders + submission TSV, replays
   ``rank_captions`` with unpack columns from predictions — exact
   ``low_confidence_used`` / ``prob_gap_pre_fallback`` (needs HF cache / venv).

Run from ``admire_text_bottleneck``::

    python3 scripts/report_confidence_slice.py --posthoc
    ./venv/bin/python scripts/report_confidence_slice.py --replay-ranker
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

def _read_tsv_dicts(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))

def _unpacked_from_prediction_row(row: Dict[str, str]) -> Dict[str, str]:
    return {
        "usage_type": str(row["usage_type"]),
        "english_translation": str(row["english_translation"]),
        "literal_meaning": str(row["literal_meaning"]),
        "context_visual": str(row["context_visual"]),
        "compound_visual": str(row["compound_visual"]),
        "de_idiomatized_sentence": str(row.get("de_idiomatized_sentence") or ""),
    }

def _rank_of_gt_top1_in_prediction(
    pred_order_csv: str,
    gt_expected_order: List[str],
) -> int:
    """1-based rank of ground-truth #1 image in predicted order; 0 if absent."""
    gold_top1 = gt_expected_order[0]
    names = [s.strip() for s in pred_order_csv.split(",") if s.strip()]
    try:
        return names.index(gold_top1) + 1
    except ValueError:
        return 0

def _print_top1_accuracy_by_usage_type(
    fig_ok: int,
    fig_tot: int,
    lit_ok: int,
    lit_tot: int,
    other_ok: int,
    other_tot: int,
    source_note: str,
) -> None:
    """Top-1 = predicted first image == ground-truth first image."""
    print()
    print(f"  --- Top-1 accuracy by usage_type ({source_note}) ---")
    print("      (usage_type from predictions TSV = LLM unpack label)")
    if fig_tot:
        print(
            f"      figurative:  {fig_ok}/{fig_tot}  ({100.0 * fig_ok / fig_tot:.1f}%)",
        )
    else:
        print("      figurative:  (no rows)")
    if lit_tot:
        print(
            f"      literal:     {lit_ok}/{lit_tot}  ({100.0 * lit_ok / lit_tot:.1f}%)",
        )
    else:
        print("      literal:     (no rows)")
    if other_tot:
        print(
            f"      other:       {other_ok}/{other_tot}  ({100.0 * other_ok / other_tot:.1f}%)",
        )

def _print_gt_top1_rank_histogram(ranks: List[int], n: int, label: str) -> None:
    counts = [0, 0, 0, 0, 0]
    missing = 0
    for r in ranks:
        if 1 <= r <= 5:
            counts[r - 1] += 1
        else:
            missing += 1
    print()
    print(f"  --- ground-truth Top-1 image rank in *predicted* order ({label}) ---")
    print("      (rank 1 = prediction Top-1 matches gold Top-1)")
    for k in range(1, 6):
        c = counts[k - 1]
        pct = 100.0 * c / n if n else 0.0
        print(f"      rank {k}:  {c:3d} / {n}  ({pct:5.1f}%)")
    if missing:
        print(f"      not in top-5 list: {missing}")
    acc_at_k = [sum(counts[: j + 1]) for j in range(5)]
    print()
    for j, ak in enumerate(acc_at_k, start=1):
        pct = 100.0 * ak / n if n else 0.0
        print(f"      gold Top-1 within predicted top-{j}: {ak}/{n} ({pct:.1f}%)")

def _contingency(
    rows: List[Tuple[bool, bool]],
) -> Tuple[int, int, int, int]:
    """Returns (fb_ok, fb_miss, nofb_ok, nofb_miss) with fb=low_confidence_used."""
    fb_ok = fb_miss = nofb_ok = nofb_miss = 0
    for low_conf, top1_ok in rows:
        if low_conf:
            if top1_ok:
                fb_ok += 1
            else:
                fb_miss += 1
        else:
            if top1_ok:
                nofb_ok += 1
            else:
                nofb_miss += 1
    return fb_ok, fb_miss, nofb_ok, nofb_miss

def _run_posthoc_suite(
    label: str,
    predictions_path: Path,
    ground_truth_path: Path,
    margin_lo: float = 0.015,
) -> None:
    """Cross Top-1 OK/MISS with final-prob top1−top2 margin (saved ``scores``)."""
    pred_rows = _read_tsv_dicts(predictions_path)
    with open(ground_truth_path, encoding="utf-8") as f:
        gt: List[Dict[str, Any]] = json.load(f)

    n = min(len(pred_rows), len(gt))
    thin_ok = thin_miss = wide_ok = wide_miss = 0
    gt_top1_ranks: List[int] = []
    fig_ok = fig_tot = lit_ok = lit_tot = other_ok = other_tot = 0
    for i in range(n):
        pr = pred_rows[i]
        order = str(pr["expected_order"]).strip()
        gt_top1_ranks.append(
            _rank_of_gt_top1_in_prediction(order, gt[i]["expected_order"]),
        )
        pred_top1 = order.split(",")[0].strip()
        gt_top1 = gt[i]["expected_order"][0]
        ok = pred_top1 == gt_top1
        usage_raw = str(pr.get("usage_type", "")).strip().lower()
        if usage_raw == "literal":
            lit_tot += 1
            if ok:
                lit_ok += 1
        elif usage_raw == "figurative":
            fig_tot += 1
            if ok:
                fig_ok += 1
        else:
            other_tot += 1
            if ok:
                other_ok += 1
        scores = ast.literal_eval(str(pr["scores"]))
        s_desc = sorted(float(x) for x in scores)
        margin = s_desc[-1] - s_desc[-2]
        thin = margin < margin_lo
        if thin:
            if ok:
                thin_ok += 1
            else:
                thin_miss += 1
        else:
            if ok:
                wide_ok += 1
            else:
                wide_miss += 1

    print(f"\n{'=' * 70}")
    print(f"  POSTHOC SCORE SLICE  ({label})  n={n}")
    print(f"  predictions: {predictions_path}")
    print(f"  margin = max(scores) - second_max (after ranker, incl. any fallback)")
    print(f"  'thin' if margin < {margin_lo}")
    print(f"{'=' * 70}")
    print()
    print(f"                    Top-1 OK    Top-1 MISS")
    print(f"  thin margin         {thin_ok:4d}        {thin_miss:4d}")
    print(f"  wide margin         {wide_ok:4d}        {wide_miss:4d}")
    print()
    miss_total = thin_miss + wide_miss
    if miss_total:
        print(
            f"  Of all MISS: thin margin  {100 * thin_miss / miss_total:.1f}%  |  "
            f"wide margin {100 * wide_miss / miss_total:.1f}%",
        )
        print(
            "  (wide-margin MISS = confidently wrong on final distribution.)",
        )
    if pred_rows and "low_confidence_used" in pred_rows[0]:
        pairs_fb: List[Tuple[bool, bool]] = []
        for i in range(n):
            pr = pred_rows[i]
            raw = str(pr.get("low_confidence_used", "")).strip().lower()
            low = raw in ("true", "1", "yes", "t")
            pred_top1 = str(pr["expected_order"]).strip().split(",")[0].strip()
            pairs_fb.append((low, pred_top1 == gt[i]["expected_order"][0]))
        fb_ok, fb_miss, nofb_ok, nofb_miss = _contingency(pairs_fb)
        print("  --- from TSV columns (exact ranker fallback) ---")
        print(f"                      Top-1 OK    Top-1 MISS")
        print(f"  fallback used          {fb_ok:4d}        {fb_miss:4d}")
        print(f"  no fallback            {nofb_ok:4d}        {nofb_miss:4d}")
        nnofb = nofb_ok + nofb_miss
        if nnofb:
            print(f"  P(Top-1 OK | no fallback) = {100 * nofb_ok / nnofb:.1f}%")
        nfb = fb_ok + fb_miss
        if nfb:
            print(f"  P(Top-1 OK | fallback)    = {100 * fb_ok / nfb:.1f}%")
    else:
        print(
            "  (No ``low_confidence_used`` column — re-run pipeline after ranker update,",
        )
        print(
            "   or use --replay-ranker for exact fallback flag.)",
        )
    _print_gt_top1_rank_histogram(gt_top1_ranks, n, "saved expected_order in TSV")
    _print_top1_accuracy_by_usage_type(
        fig_ok,
        fig_tot,
        lit_ok,
        lit_tot,
        other_ok,
        other_tot,
        "saved expected_order in TSV",
    )

def _run_replay_suite(
    label: str,
    predictions_path: Path,
    submission_path: Path,
    ground_truth_path: Path,
    ranker: Any,
) -> None:
    from src.data_loader import TSVLoader

    pred_rows = _read_tsv_dicts(predictions_path)
    with open(ground_truth_path, encoding="utf-8") as f:
        gt: List[Dict[str, Any]] = json.load(f)

    loader = TSVLoader(submission_path)
    submission_rows = list(loader)
    n = min(len(pred_rows), len(submission_rows), len(gt))
    if n == 0:
        print(f"[{label}] No rows to evaluate.")
        return

    pairs: List[Tuple[bool, bool]] = []
    order_mismatches = 0
    gt_top1_ranks: List[int] = []
    fig_ok = fig_tot = lit_ok = lit_tot = other_ok = other_tot = 0

    for i in range(n):
        pr = pred_rows[i]
        sub = submission_rows[i]
        unpacked = _unpacked_from_prediction_row(pr)
        (
            _best_idx,
            _scores,
            replay_order,
            low_confidence_used,
            _prob_gap_pre,
        ) = ranker.rank_captions(
            unpacked,
            sub["captions"],
            sub["image_names"],
        )
        saved_order = str(pr["expected_order"]).strip()
        if saved_order != replay_order:
            order_mismatches += 1

        gt_top1_ranks.append(
            _rank_of_gt_top1_in_prediction(replay_order, gt[i]["expected_order"]),
        )
        pred_top1 = replay_order.split(",")[0].strip()
        gt_top1 = gt[i]["expected_order"][0]
        top1_ok = pred_top1 == gt_top1
        usage_raw = str(pr.get("usage_type", "")).strip().lower()
        if usage_raw == "literal":
            lit_tot += 1
            if top1_ok:
                lit_ok += 1
        elif usage_raw == "figurative":
            fig_tot += 1
            if top1_ok:
                fig_ok += 1
        else:
            other_tot += 1
            if top1_ok:
                other_ok += 1
        pairs.append((low_confidence_used, top1_ok))

    fb_ok, fb_miss, nofb_ok, nofb_miss = _contingency(pairs)
    total_ok = fb_ok + nofb_ok
    total_miss = fb_miss + nofb_miss
    nfb = fb_ok + fb_miss
    nnofb = nofb_ok + nofb_miss

    print(f"\n{'=' * 70}")
    print(f"  REPLAY RANKER — FALLBACK SLICE  ({label})  n={n}")
    print(f"  predictions: {predictions_path}")
    print(f"  replay order == saved TSV order: {n - order_mismatches}/{n} match")
    if order_mismatches:
        print(
            "  (mismatches usually mean different ranker/code vs when TSV was written)",
        )
    print(f"{'=' * 70}")
    print()
    print("  Top-1 vs ground truth × low_confidence_used")
    print("  (pre-fallback prob_gap = max(p)-mean(p) < 0.08 → 75% context/compound mix)")
    print()
    print(f"                      Top-1 OK    Top-1 MISS    row %")
    print(f"  fallback used          {fb_ok:4d}        {fb_miss:4d}     {100 * (fb_ok + fb_miss) / n:.1f}%")
    print(f"  no fallback            {nofb_ok:4d}        {nofb_miss:4d}     {100 * (nnofb) / n:.1f}%")
    print()
    if nnofb:
        print(
            f"  P(Top-1 OK | no fallback) = {nofb_ok}/{nnofb} = "
            f"{100 * nofb_ok / nnofb:.1f}%",
        )
    if nfb:
        print(
            f"  P(Top-1 OK | fallback)    = {fb_ok}/{nfb} = "
            f"{100 * fb_ok / nfb:.1f}%",
        )
    print()
    print(
        f"  Overall Top-1 accuracy (replay): {total_ok}/{n} = {100 * total_ok / n:.1f}%",
    )
    if total_miss:
        print(
            f"  Share of MISS with no fallback: {nofb_miss}/{total_miss} = "
            f"{100 * nofb_miss / total_miss:.1f}%",
        )
    _print_gt_top1_rank_histogram(gt_top1_ranks, n, "replay ranker order")
    _print_top1_accuracy_by_usage_type(
        fig_ok,
        fig_tot,
        lit_ok,
        lit_tot,
        other_ok,
        other_tot,
        "replay ranker order",
    )

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--lang",
        choices=("Chinese", "Turkish", "both"),
        default="both",
        help="Which mini suite to report (default: both).",
    )
    g = ap.add_mutually_exclusive_group()
    g.add_argument(
        "--posthoc",
        action="store_true",
        help="Only TSV + GT (default if neither flag given).",
    )
    g.add_argument(
        "--replay-ranker",
        action="store_true",
        help="Load cross-encoders and replay ranker (needs venv + HF models).",
    )
    args = ap.parse_args()
    posthoc = args.posthoc or not args.replay_ranker

    data_dir = _ROOT.parent / "data"
    out_dir = _ROOT / "outputs"
    bench_dir = _ROOT / "benchmarks"

    ranker = None
    if not posthoc:
        from src.ranker import CaptionRanker

        print("Loading cross-encoders (one load for all suites)…")
        ranker = CaptionRanker()

    specs: List[Tuple[str, Path, Path, Path]] = []
    if args.lang in ("Chinese", "both"):
        specs.append(
            (
                "Chinese (20)",
                out_dir / "mini_predictions_Chinese.tsv",
                data_dir / "submission_Chinese.tsv",
                bench_dir / "ground_truth_chinese_20.json",
            ),
        )
    if args.lang in ("Turkish", "both"):
        specs.append(
            (
                "Turkish (30)",
                out_dir / "mini_predictions_Turkish.tsv",
                data_dir / "submission_Turkish.tsv",
                bench_dir / "ground_truth_turkish_30.json",
            ),
        )

    for label, pred_p, sub_p, gt_p in specs:
        if not pred_p.is_file():
            print(f"[skip] missing predictions: {pred_p}")
            continue
        if not gt_p.is_file():
            print(f"[skip] missing ground truth: {gt_p}")
            continue
        if posthoc:
            _run_posthoc_suite(label, pred_p, gt_p)
        else:
            if not sub_p.is_file():
                print(f"[skip] missing submission: {sub_p}")
                continue
            assert ranker is not None
            _run_replay_suite(label, pred_p, sub_p, gt_p, ranker)

if __name__ == "__main__":
    main()
