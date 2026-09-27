from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

from .fast_train_model import FEATURE_COLUMNS, _f05, _macro_entity_f05, _metrics, connect, coverage_report, label_and_feature, prepare_views


COSINE_COLUMNS = ["name_tfidf_cosine", "address_tfidf_cosine"]
FEATURE_COLUMNS_COSINE = FEATURE_COLUMNS + COSINE_COLUMNS
THRESHOLDS = [0.80, 0.85, 0.90, 0.92, 0.95]


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _entity_text(con: duckdb.DuckDBPyConnection, table: str, field: str) -> pd.DataFrame:
    return con.execute(
        f"""
        SELECT entity_id, coalesce({field}, '') AS text
        FROM {table}
        """
    ).fetchdf()


def _build_text_vector_map(texts: pd.Series, max_features: int) -> tuple[TfidfVectorizer, sparse.csr_matrix, dict[str, int]]:
    unique_texts = pd.Series(texts.fillna("").astype(str).unique())
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=max_features,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(unique_texts)
    text_to_row = {text: idx for idx, text in enumerate(unique_texts)}
    return vectorizer, matrix.tocsr(), text_to_row


def _cosine_for_pairs(
    pairs: pd.DataFrame,
    s1_text_by_id: pd.Series,
    cand_text_by_id: pd.Series,
    text_to_row: dict[str, int],
    matrix: sparse.csr_matrix,
    batch_size: int,
) -> np.ndarray:
    scores = np.zeros(len(pairs), dtype=np.float32)
    s1_text = pairs["s1_id"].map(s1_text_by_id).fillna("").astype(str)
    cand_text = pairs["candidate_id"].map(cand_text_by_id).fillna("").astype(str)
    left_rows = s1_text.map(text_to_row).fillna(-1).astype(np.int32).to_numpy()
    right_rows = cand_text.map(text_to_row).fillna(-1).astype(np.int32).to_numpy()

    for start in range(0, len(pairs), batch_size):
        end = min(start + batch_size, len(pairs))
        left = left_rows[start:end]
        right = right_rows[start:end]
        valid = (left >= 0) & (right >= 0)
        if not valid.any():
            continue
        left_mat = matrix[left[valid]]
        right_mat = matrix[right[valid]]
        target_idx = np.arange(start, end, dtype=np.int64)[valid]
        scores[target_idx] = np.asarray(left_mat.multiply(right_mat).sum(axis=1)).ravel().astype(np.float32)
    return scores


def add_cosine_features(args: argparse.Namespace, base_features: pd.DataFrame, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    s1_names = _entity_text(con, "s1", "clean_name").set_index("entity_id")["text"]
    s1_addresses = _entity_text(con, "s1", "clean_address").set_index("entity_id")["text"]
    cand_names = _entity_text(con, "others", "clean_name").set_index("entity_id")["text"]
    cand_addresses = _entity_text(con, "others", "clean_address").set_index("entity_id")["text"]

    name_vectorizer, name_matrix, name_rows = _build_text_vector_map(
        pd.concat([s1_names, cand_names], ignore_index=True),
        args.max_features,
    )
    base_features["name_tfidf_cosine"] = _cosine_for_pairs(
        base_features,
        s1_names,
        cand_names,
        name_rows,
        name_matrix,
        args.cosine_batch_size,
    )
    del name_vectorizer, name_matrix, name_rows

    address_vectorizer, address_matrix, address_rows = _build_text_vector_map(
        pd.concat([s1_addresses, cand_addresses], ignore_index=True),
        args.max_features,
    )
    base_features["address_tfidf_cosine"] = _cosine_for_pairs(
        base_features,
        s1_addresses,
        cand_addresses,
        address_rows,
        address_matrix,
        args.cosine_batch_size,
    )
    del address_vectorizer, address_matrix, address_rows
    return base_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--max-features", type=int, default=100000)
    parser.add_argument("--cosine-batch-size", type=int, default=500000)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    artifacts_dir = Path(args.artifacts_dir)
    metrics_dir = artifacts_dir / "metrics"
    model_dir = artifacts_dir / "model"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    con = connect(cache_dir)
    try:
        print("load", flush=True)
        prepare_views(con, cache_dir, Path(args.data_dir))
        coverage = coverage_report(con)
        print(f"candidate recall {coverage['candidate_recall']:.6f}", flush=True)

        print("features", flush=True)
        feature_path = label_and_feature(con, cache_dir)
        data = pd.read_parquet(feature_path)

        cosine_start = time.time()
        data = add_cosine_features(args, data, con)
        cosine_seconds = round(time.time() - cosine_start, 3)
        print(f"cosine feature build seconds {cosine_seconds}", flush=True)
    finally:
        con.close()

    split_key = pd.util.hash_pandas_object(data["s1_id"], index=False) % 100
    train_mask = split_key < 80
    val_mask = ~train_mask

    x_train = data.loc[train_mask, FEATURE_COLUMNS_COSINE].astype("float32")
    y_train = data.loc[train_mask, "label"].astype("int8")
    x_val = data.loc[val_mask, FEATURE_COLUMNS_COSINE].astype("float32")
    y_val = data.loc[val_mask, "label"].astype("int8")
    scale_pos_weight = float(max(1.0, (y_train == 0).sum() / max(1, (y_train == 1).sum())))

    print("train", flush=True)
    train_start = time.time()
    model = XGBClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        tree_method="hist",
        n_jobs=-1,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=2026,
    )
    model.fit(x_train, y_train)
    train_seconds = round(time.time() - train_start, 3)
    print(f"train seconds {train_seconds}", flush=True)

    val_prob = model.predict_proba(x_val)[:, 1]
    val_eval = data.loc[val_mask, ["s1_id", "candidate_id", "label"]].copy()
    val_eval["probability"] = val_prob

    threshold_rows = []
    y_val_np = y_val.to_numpy()
    for threshold in THRESHOLDS:
        row = _metrics(y_val_np, val_prob, threshold)
        row["entity_macro_f0_5"] = _macro_entity_f05(val_eval, threshold)
        threshold_rows.append(row)
        print(
            f"{threshold}: precision={row['precision']:.6f} recall={row['recall']:.6f} F0.5={row['f0_5']:.6f}",
            flush=True,
        )

    best = max(threshold_rows, key=lambda row: row["f0_5"])
    print(f"BEST THRESHOLD {best['threshold']}", flush=True)

    metrics = {
        "coverage": coverage,
        "rows": {
            "total_candidates": int(len(data)),
            "train_candidates": int(train_mask.sum()),
            "validation_candidates": int(val_mask.sum()),
            "train_positive": int(y_train.sum()),
            "validation_positive": int(y_val.sum()),
            "scale_pos_weight": scale_pos_weight,
        },
        "timing": {
            "cosine_feature_build_seconds": cosine_seconds,
            "train_seconds": train_seconds,
        },
        "feature_columns": FEATURE_COLUMNS_COSINE,
        "thresholds": threshold_rows,
        "best_threshold": best,
    }
    model.save_model(model_dir / "xgb_entity_match_cosine.json")
    _write_json(metrics_dir / "model_metrics_cosine.json", metrics)
    _write_json(metrics_dir / "best_threshold_cosine.json", best)


if __name__ == "__main__":
    main()
