"""
Benchmark evaluation utilities.

Compares pipeline predictions against a ground-truth JSON file and reports:
- nDCG@5 with AdMIRe relevance mapping (3, 1, 0, 0, 0)
- Top-1 accuracy
- per-row and average Spearman rank correlation
"""

import json
import logging
import math
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_ADMIRE_RELEVANCE = [3.0, 1.0, 0.0, 0.0, 0.0]

def _spearman_rho(predicted: List[str], expected: List[str]) -> float:
    """Compute Spearman rank correlation between two orderings."""
    n = len(expected)
    if n <= 1:
        return 1.0

    expected_rank = {name: i for i, name in enumerate(expected)}
    d_sq_sum = 0.0
    for pred_rank, name in enumerate(predicted):
        exp_rank = expected_rank.get(name, pred_rank)
        d_sq_sum += (pred_rank - exp_rank) ** 2

    return 1.0 - (6.0 * d_sq_sum) / (n * (n * n - 1))

def _dcg_at_k(relevances: List[float], k: int = 5) -> float:
    """Compute DCG@k with log2 discount starting at rank 2."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += rel / math.log2(i + 2)
    return dcg

def _ndcg_admire(predicted: List[str], expected: List[str], k: int = 5) -> float:
    """Compute nDCG@k using AdMIRe relevance mapping [3,1,0,0,0]."""
    expected_rel = {
        name: _ADMIRE_RELEVANCE[i] if i < len(_ADMIRE_RELEVANCE) else 0.0
        for i, name in enumerate(expected)
    }
    pred_rels = [expected_rel.get(name, 0.0) for name in predicted]
    dcg = _dcg_at_k(pred_rels, k=k)
    idcg = _dcg_at_k(sorted(expected_rel.values(), reverse=True), k=k)
    return dcg / idcg if idcg > 0 else 0.0

def evaluate(
    predictions_df: pd.DataFrame,
    ground_truth_path: str | Path,
    limit: Optional[int] = None,
) -> None:
    """Print accuracy metrics comparing predictions to ground truth."""
    gt_path = Path(ground_truth_path)
    if not gt_path.exists():
        logger.info("No ground truth file at %s -- skipping evaluation", gt_path)
        return

    with open(gt_path) as f:
        ground_truth = json.load(f)

    n_gt = len(ground_truth)
    n_pred = len(predictions_df)
    n_eval = min(n_gt, n_pred)

    if n_eval == 0:
        return

    top1_correct = 0
    spearman_sum = 0.0
    ndcg_sum = 0.0
    details: List[str] = []

    for i in range(n_eval):
        gt_row = ground_truth[i]
        gt_order: List[str] = gt_row["expected_order"]
        gt_top1: str = gt_order[0]

        pred_order_str: str = predictions_df.iloc[i]["expected_order"]
        pred_order: List[str] = [s.strip() for s in pred_order_str.split(",")]
        pred_top1: str = pred_order[0]

        correct = pred_top1 == gt_top1
        if correct:
            top1_correct += 1

        rho = _spearman_rho(pred_order, gt_order)
        spearman_sum += rho
        ndcg = _ndcg_admire(pred_order, gt_order, k=5)
        ndcg_sum += ndcg

        mark = "OK" if correct else "MISS"
        details.append(
            f"  Row {i+1:2d} [{mark:4s}]  "
            f"compound={gt_row['compound']}  "
            f"pred={pred_top1}  "
            f"expected={gt_top1}  "
            f"rho={rho:.3f}  "
            f"ndcg@5={ndcg:.3f}"
        )

    accuracy = top1_correct / n_eval
    avg_rho = spearman_sum / n_eval
    avg_ndcg = ndcg_sum / n_eval

    print("\n" + "=" * 70)
    print(f"  BENCHMARK EVALUATION  ({n_eval} rows)")
    print("=" * 70)
    for line in details:
        print(line)
    print("-" * 70)
    print(f"  nDCG@5 (3/1/0/0/0): {avg_ndcg:.3f}")
    print(f"  Top-1 Accuracy : {top1_correct}/{n_eval} = {accuracy:.1%}")
    print(f"  Avg Spearman ρ : {avg_rho:.3f}")
    print("=" * 70)
