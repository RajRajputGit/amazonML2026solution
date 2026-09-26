from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from .blocking import candidate_stats, generate_candidates
from .config import RANDOM_SEED, make_paths
from .data_loader import load_training, verify_training_files
from .evaluation import parse_matches, truth_map, tune_thresholds
from .features import build_features


def save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["original_name"] = out["business_name"].astype(str)
    name = out["business_name"].astype(str).str.casefold().str.replace("&", " and ", regex=False)
    name = name.str.replace(r"[^\w\s]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    out["unicode_name"] = name
    out["clean_name"] = name
    out["compact_name"] = name.str.replace(r"\s+", "", regex=True)
    out["suffix_normalized_name"] = name.str.replace(
        r"\b(inc|incorporated|corp|corporation|ltd|limited|llc|llp|plc|gmbh|ag|bv|sarl|srl|sa|sas|pvt|private|co|company)\b",
        " ",
        regex=True,
    ).str.replace(r"\s+", " ", regex=True).str.strip()
    out["ascii_folded_name"] = ""
    out["original_address"] = out["business_address"].astype(str)
    addr = out["business_address"].astype(str).str.casefold().str.replace("&", " and ", regex=False)
    addr = addr.str.replace(r"[^\w\s]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    out["clean_address"] = addr
    out["compact_address"] = out["clean_address"].str.replace(r"\s+", "", regex=True)
    return out


def eda(s1, s2, s3, gt, paths):
    def profile(df, name):
        name_len = df["business_name"].astype(str).str.len()
        addr_len = df["business_address"].astype(str).str.len()
        return {
            "rows": int(len(df)),
            "missing_values": {k: int(v) for k, v in df.replace("", pd.NA).isna().sum().items()},
            "unique_entity_ids": int(df["entity_id"].nunique()),
            "duplicate_entity_ids": int(len(df) - df["entity_id"].nunique()),
            "duplicate_business_names": int(len(df) - df["business_name"].nunique()),
            "duplicate_addresses": int(len(df) - df["business_address"].nunique()),
            "country_distribution": {str(k): int(v) for k, v in df["country"].value_counts(dropna=False).items()},
            "business_name_length": name_len.describe(percentiles=[0.5, 0.95]).to_dict(),
            "address_length": addr_len.describe(percentiles=[0.5, 0.95]).to_dict(),
        }
    truth = truth_map(gt)
    counts = np.array([len(v) for v in truth.values()])
    categories = {"singleton": 0, "s2_only": 0, "s3_only": 0, "s2_s3": 0}
    true_pairs = set()
    for s1_id, matches in truth.items():
        true_pairs.update((s1_id, m) for m in matches)
        has_s2 = any(m.startswith("S2-") for m in matches)
        has_s3 = any(m.startswith("S3-") for m in matches)
        if not matches:
            categories["singleton"] += 1
        elif has_s2 and has_s3:
            categories["s2_s3"] += 1
        elif has_s2:
            categories["s2_only"] += 1
        elif has_s3:
            categories["s3_only"] += 1
    gt_stats = {
        "gt_rows": int(len(gt)),
        "total_true_links": int(len(true_pairs)),
        "singleton_count": int(categories["singleton"]),
        "singleton_pct": float(categories["singleton"] / len(gt) * 100),
        "average_matches_per_s1": float(counts.mean()),
        "median_matches_per_s1": float(np.median(counts)),
        "maximum_matches_per_s1": int(counts.max()),
        "match_count_distribution": {str(k): int(v) for k, v in pd.Series(counts).value_counts().sort_index().items()},
        **categories,
    }
    report = {"S1": profile(s1, "S1"), "S2": profile(s2, "S2"), "S3": profile(s3, "S3"), "ground_truth": gt_stats}
    save_json(paths.eda_dir / "eda_summary.json", report)
    examples = []
    for s1_id, matches in truth.items():
        examples.append({"s1_id": s1_id, "match_ids": sorted(matches)})
        if len(examples) >= 30:
            break
    needed_s1 = {e["s1_id"] for e in examples}
    needed_other = {m for e in examples for m in e["match_ids"]}
    lookup_frames = [
        s1[s1["entity_id"].isin(needed_s1 | needed_other)],
        s2[s2["entity_id"].isin(needed_other)],
        s3[s3["entity_id"].isin(needed_other)],
    ]
    lookup = pd.concat(lookup_frames).set_index("entity_id")[["business_name", "business_address", "country"]].to_dict("index")
    for item in examples:
        item["s1"] = lookup.get(item["s1_id"])
        item["matches"] = {m: lookup.get(m) for m in item.pop("match_ids")}
    save_json(paths.eda_dir / "ground_truth_examples.json", examples)
    return truth, true_pairs, report


def exact_baseline(s1, s2, s3, true_pairs):
    from .blocking import exact_block

    cands = {}
    exact_block(s1, {"S2": s2, "S3": s3}, "clean_name", "clean_name", "exact_name_block", cands)
    df = pd.DataFrame(cands.values())
    return df, candidate_stats(df, len(s1), len(s2) + len(s3), true_pairs)


def downsample_training(features: pd.DataFrame, max_neg_per_pos: int = 20) -> pd.DataFrame:
    pos = features[features["label"] == 1]
    neg = features[features["label"] == 0].copy()
    neg["hardness"] = neg[["name_tfidf_similarity", "address_tfidf_similarity", "name_ratio", "address_ratio", "blocking_method_count"]].sum(axis=1)
    keep = min(len(neg), max(len(pos) * max_neg_per_pos, 50000))
    neg = neg.sort_values("hardness", ascending=False).head(keep).drop(columns=["hardness"])
    return pd.concat([pos, neg], ignore_index=True)


def run_phase1(args):
    t_start = time.time()
    paths = make_paths(args.data_dir, args.artifacts_dir)
    print("[1/10] verifying training files", flush=True)
    verify = verify_training_files(paths)
    print("[2/10] loading training data", flush=True)
    s1, s2, s3, gt = load_training(paths)
    print("[3/10] running EDA", flush=True)
    truth, true_pairs, eda_report = eda(s1, s2, s3, gt, paths)
    print("[4/10] normalizing S1", flush=True)
    s1 = normalize_frame(s1)
    print("[4/10] normalizing S2", flush=True)
    s2 = normalize_frame(s2)
    print("[4/10] normalizing S3", flush=True)
    s3 = normalize_frame(s3)
    print("[5/10] creating validation split", flush=True)
    train_ids, val_ids = train_test_split(sorted(truth), test_size=args.validation_size, random_state=RANDOM_SEED)
    split = {"seed": RANDOM_SEED, "validation_size": args.validation_size, "train_s1_ids": train_ids, "validation_s1_ids": val_ids}
    save_json(paths.metrics_dir / "validation_split.json", split)
    print("[6/10] exact-name baseline blocking", flush=True)
    baseline_cands, baseline_stats = exact_baseline(s1, s2, s3, true_pairs)
    save_json(paths.metrics_dir / "baseline_exact_name_blocking.json", baseline_stats)
    print("[7/10] multi-pass candidate generation", flush=True)
    candidates, runtime_stats = generate_candidates(s1, s2, s3, {"name_top_k": args.name_top_k, "address_top_k": args.address_top_k})
    candidates.to_pickle(paths.metrics_dir / "candidate_pairs_phase1.pkl")
    final_stats = {**candidate_stats(candidates, len(s1), len(s2) + len(s3), true_pairs), **runtime_stats}
    save_json(paths.metrics_dir / "final_blocking_stats.json", final_stats)
    print("[8/10] building pair features", flush=True)
    train_truth = {k: truth[k] for k in train_ids}
    val_truth = {k: truth[k] for k in val_ids}
    train_pairs = {(s, m) for s, ms in train_truth.items() for m in ms}
    train_features, feature_cols = build_features(candidates[candidates["s1_id"].isin(train_ids)], s1, s2, s3, train_pairs)
    val_features, _ = build_features(candidates[candidates["s1_id"].isin(val_ids)], s1, s2, s3, None)
    train_sample = downsample_training(train_features, args.max_neg_per_pos)
    scale_pos_weight = max(1.0, (train_sample["label"] == 0).sum() / max(1, (train_sample["label"] == 1).sum()))
    print("[9/10] training XGBoost", flush=True)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_SEED,
        scale_pos_weight=scale_pos_weight,
        n_jobs=max(1, psutil.cpu_count(logical=True) - 1),
    )
    fit_t0 = time.time()
    model.fit(train_sample[feature_cols], train_sample["label"])
    train_runtime = time.time() - fit_t0
    print("[10/10] threshold tuning and error analysis", flush=True)
    val_features["probability"] = model.predict_proba(val_features[feature_cols])[:, 1] if len(val_features) else []
    coarse = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    coarse_df = tune_thresholds(val_features, val_truth, coarse)
    best = float(coarse_df.sort_values("macro_f05", ascending=False).iloc[0]["threshold"])
    fine = sorted(set([round(x, 3) for x in np.arange(max(0.01, best - 0.05), min(0.99, best + 0.051), 0.01)]))
    fine_df = tune_thresholds(val_features, val_truth, fine)
    thresholds = pd.concat([coarse_df, fine_df]).drop_duplicates("threshold").sort_values("threshold")
    thresholds.to_csv(paths.metrics_dir / "threshold_tuning.csv", index=False)
    best_row = thresholds.sort_values("macro_f05", ascending=False).iloc[0].to_dict()
    importances = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: x[1], reverse=True)
    train_metrics = {
        "training_pairs": int(len(train_sample)),
        "positive_pairs": int((train_sample["label"] == 1).sum()),
        "negative_pairs": int((train_sample["label"] == 0).sum()),
        "feature_count": len(feature_cols),
        "xgboost_params": model.get_params(),
        "training_runtime_sec": round(train_runtime, 3),
        "most_important_features": [{"feature": k, "importance": float(v)} for k, v in importances[:20]],
        "best_threshold": best_row,
    }
    save_json(paths.metrics_dir / "training_validation_metrics.json", train_metrics)
    joblib.dump({"model": model, "feature_columns": feature_cols, "threshold": best_row["threshold"]}, paths.models_dir / "xgboost_phase1.joblib")
    experiments = pd.DataFrame([{**final_stats, **train_metrics, **{f"best_{k}": v for k, v in best_row.items()}, "runtime": round(time.time() - t_start, 3)}])
    exp_path = paths.metrics_dir / "experiments.csv"
    experiments.to_csv(exp_path, mode="a", header=not exp_path.exists(), index=False)
    fp_fn = error_analysis(val_features, val_truth, best_row["threshold"], true_pairs)
    save_json(paths.metrics_dir / "validation_error_analysis.json", fp_fn)
    summary = {
        "data_verification": verify,
        "eda": eda_report,
        "baseline_blocking": baseline_stats,
        "final_blocking": final_stats,
        "training": train_metrics,
        "error_analysis": fp_fn,
        "artifacts": {
            "experiments": str(exp_path),
            "model": str(paths.models_dir / "xgboost_phase1.joblib"),
            "candidate_pairs": str(paths.metrics_dir / "candidate_pairs_phase1.pkl"),
            "metrics": str(paths.metrics_dir),
            "eda": str(paths.eda_dir),
        },
    }
    save_json(paths.metrics_dir / "phase1_execution_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def error_analysis(val_features, val_truth, threshold, true_pairs):
    pred_pairs = set(zip(val_features.loc[val_features["probability"] >= threshold, "s1_id"], val_features.loc[val_features["probability"] >= threshold, "candidate_id"]))
    val_true_pairs = {(s, m) for s, ms in val_truth.items() for m in ms}
    candidate_pairs = set(zip(val_features["s1_id"], val_features["candidate_id"]))
    blocking_failures = sorted([{"s1_id": s, "candidate_id": m} for s, m in (val_true_pairs - candidate_pairs)][:200], key=lambda x: (x["s1_id"], x["candidate_id"]))
    fp = val_features[[((a, b) in pred_pairs and (a, b) not in val_true_pairs) for a, b in zip(val_features["s1_id"], val_features["candidate_id"])]].nlargest(200, "probability")
    fn = val_features[[((a, b) in val_true_pairs and (a, b) not in pred_pairs) for a, b in zip(val_features["s1_id"], val_features["candidate_id"])]].nlargest(200, "probability")
    return {
        "false_positive_count": int(len(pred_pairs - val_true_pairs)),
        "false_negative_count": int(len(val_true_pairs - pred_pairs)),
        "blocking_failure_count": int(len(val_true_pairs - candidate_pairs)),
        "blocking_failure_examples": blocking_failures[:50],
        "false_positive_examples": fp[["s1_id", "candidate_id", "probability", "name_tfidf_similarity", "address_tfidf_similarity", "name_ratio", "address_ratio"]].to_dict("records"),
        "false_negative_examples": fn[["s1_id", "candidate_id", "probability", "name_tfidf_similarity", "address_tfidf_similarity", "name_ratio", "address_ratio"]].to_dict("records"),
        "notes": [
            "BLOCKING FAILURE means true validation pair absent from generated candidates.",
            "MODEL/THRESHOLD FAILURE means pair was candidate but probability below selected threshold.",
            "FALSE POSITIVES with high name score and weak address score suggest branch/franchise confusion.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["train"], default="train")
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--name-top-k", type=int, default=12)
    parser.add_argument("--address-top-k", type=int, default=8)
    parser.add_argument("--max-neg-per-pos", type=int, default=20)
    args = parser.parse_args()
    run_phase1(args)


if __name__ == "__main__":
    main()
