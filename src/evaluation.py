from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def parse_matches(value: object) -> set[str]:
    text = "" if value is None else str(value)
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return set()
    return {part.strip() for part in text.split(",") if part.strip()}


def truth_map(gt: pd.DataFrame) -> dict[str, set[str]]:
    return {str(row.source1_entity_id): parse_matches(row.matched_entity_ids) for row in gt.itertuples(index=False)}


def macro_scores(truth: dict[str, set[str]], pred: dict[str, set[str]]) -> dict[str, float]:
    precisions, recalls, f05s = [], [], []
    beta2 = 0.25
    for s1_id, true_set in truth.items():
        pred_set = pred.get(s1_id, set())
        if not true_set and not pred_set:
            precisions.append(1.0)
            recalls.append(1.0)
            f05s.append(1.0)
            continue
        if not true_set or not pred_set:
            precisions.append(0.0)
            recalls.append(0.0)
            f05s.append(0.0)
            continue
        tp = len(true_set & pred_set)
        precision = tp / len(pred_set) if pred_set else 0.0
        recall = tp / len(true_set) if true_set else 0.0
        denom = beta2 * precision + recall
        f05 = (1 + beta2) * precision * recall / denom if denom else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f05s.append(f05)
    return {
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f05": float(np.mean(f05s)),
    }


def predictions_at_threshold(candidates: pd.DataFrame, threshold: float) -> dict[str, set[str]]:
    pred = defaultdict(set)
    for row in candidates.loc[candidates["probability"] >= threshold, ["s1_id", "candidate_id"]].itertuples(index=False):
        pred[str(row.s1_id)].add(str(row.candidate_id))
    return dict(pred)


def tune_thresholds(candidates: pd.DataFrame, truth: dict[str, set[str]], thresholds: list[float]) -> pd.DataFrame:
    rows = []
    total_entities = len(truth)
    for threshold in thresholds:
        pred = predictions_at_threshold(candidates, threshold)
        scores = macro_scores(truth, pred)
        predicted_singletons = sum(1 for s1_id in truth if not pred.get(s1_id))
        avg_matches = sum(len(v) for v in pred.values()) / total_entities
        rows.append(
            {
                "threshold": threshold,
                **scores,
                "predicted_singleton_pct": predicted_singletons / total_entities * 100,
                "avg_predicted_matches_per_s1": avg_matches,
            }
        )
    return pd.DataFrame(rows)

