from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from xgboost import XGBClassifier


THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975]
FEATURE_COLUMNS = [
    "exact_clean_name",
    "exact_sorted_name",
    "name_token_jaccard",
    "exact_compact_address",
    "address_token_jaccard",
    "address_numeric_agreement",
    "country_equal",
    "name_missing_s1",
    "name_missing_candidate",
    "address_missing_s1",
    "address_missing_candidate",
    "source_is_s2",
    "source_is_s3",
]


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _f05(precision: float, recall: float) -> float:
    beta2 = 0.25
    denom = beta2 * precision + recall
    return float((1.0 + beta2) * precision * recall / denom) if denom else 0.0


def _metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    pred = y_prob >= threshold
    tp = int(((y_true == 1) & pred).sum())
    fp = int(((y_true == 0) & pred).sum())
    fn = int(((y_true == 1) & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f0_5": _f05(precision, recall),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
    }


def _macro_entity_f05(df: pd.DataFrame, threshold: float) -> float:
    pred = df["probability"].to_numpy() >= threshold
    work = pd.DataFrame({"s1_id": df["s1_id"].to_numpy(), "label": df["label"].to_numpy(), "pred": pred})
    rows = []
    for _, group in work.groupby("s1_id", sort=False):
        y = group["label"].to_numpy()
        p = group["pred"].to_numpy()
        tp = int(((y == 1) & p).sum())
        fp = int(((y == 0) & p).sum())
        fn = int(((y == 1) & ~p).sum())
        precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
        recall = tp / (tp + fn) if tp + fn else 1.0
        rows.append(_f05(precision, recall))
    return float(np.mean(rows)) if rows else 0.0


def connect(cache_dir: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("PRAGMA memory_limit='4GB'")
    con.execute("PRAGMA threads=4")
    con.execute(f"PRAGMA temp_directory='{_sql_path(cache_dir / 'duckdb_tmp')}'")
    return con


def prepare_views(con: duckdb.DuckDBPyConnection, cache_dir: Path, data_dir: Path) -> None:
    paths = {
        "candidates": cache_dir / "train_candidates.parquet",
        "s1": cache_dir / "train_s1_normalized.parquet",
        "s2": cache_dir / "train_s2_normalized.parquet",
        "s3": cache_dir / "train_s3_normalized.parquet",
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(str(path))
        con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{_sql_path(path)}')")

    gt_path = data_dir / "train" / "train_ground_truth.tsv"
    if not gt_path.exists():
        raise FileNotFoundError(str(gt_path))
    gt_cols = con.execute(
        f"DESCRIBE SELECT * FROM read_csv('{_sql_path(gt_path)}', delim='\\t', header=true, all_varchar=true)"
    ).fetchdf()["column_name"].tolist()
    if "source1_entity_id" not in gt_cols or "matched_entity_ids" not in gt_cols:
        raise ValueError(f"Unexpected ground-truth columns: {gt_cols}")

    con.execute(
        f"""
        CREATE OR REPLACE TABLE gt_pairs AS
        SELECT
            CAST(source1_entity_id AS VARCHAR) AS s1_id,
            trim(candidate_id) AS candidate_id
        FROM read_csv('{_sql_path(gt_path)}', delim='\\t', header=true, all_varchar=true),
             UNNEST(string_split(coalesce(matched_entity_ids, ''), ',')) AS t(candidate_id)
        WHERE trim(candidate_id) <> ''
        """
    )
    con.execute("CREATE OR REPLACE VIEW others AS SELECT 'S2' AS inferred_source, * FROM s2 UNION ALL SELECT 'S3' AS inferred_source, * FROM s3")


def label_and_feature(con: duckdb.DuckDBPyConnection, cache_dir: Path) -> Path:
    feature_path = cache_dir / "train_features.parquet"
    con.execute(
        f"""
        COPY (
            WITH joined AS (
                SELECT
                    c.s1_id,
                    c.candidate_id,
                    coalesce(c.candidate_source, o.inferred_source) AS candidate_source,
                    CASE WHEN g.candidate_id IS NULL THEN 0 ELSE 1 END AS label,
                    s.country AS s1_country,
                    o.country AS cand_country,
                    coalesce(s.clean_name, '') AS s1_clean_name,
                    coalesce(o.clean_name, '') AS cand_clean_name,
                    coalesce(s.sorted_name, '') AS s1_sorted_name,
                    coalesce(o.sorted_name, '') AS cand_sorted_name,
                    coalesce(s.significant_tokens, '') AS s1_name_tokens_text,
                    coalesce(o.significant_tokens, '') AS cand_name_tokens_text,
                    coalesce(s.clean_address, '') AS s1_clean_address,
                    coalesce(o.clean_address, '') AS cand_clean_address,
                    s.address_numbers AS s1_address_numbers,
                    o.address_numbers AS cand_address_numbers
                FROM candidates c
                JOIN s1 s ON c.s1_id = s.entity_id
                JOIN others o ON c.candidate_id = o.entity_id
                LEFT JOIN gt_pairs g ON c.s1_id = g.s1_id AND c.candidate_id = g.candidate_id
            ),
            tokenized AS (
                SELECT
                    *,
                    list_filter(string_split(s1_name_tokens_text, ' '), x -> length(x) > 1) AS s1_name_tokens,
                    list_filter(string_split(cand_name_tokens_text, ' '), x -> length(x) > 1) AS cand_name_tokens,
                    list_filter(string_split(s1_clean_address, ' '), x -> length(x) > 1) AS s1_address_tokens,
                    list_filter(string_split(cand_clean_address, ' '), x -> length(x) > 1) AS cand_address_tokens,
                    regexp_replace(s1_clean_address, '\\s+', '', 'g') AS s1_compact_address,
                    regexp_replace(cand_clean_address, '\\s+', '', 'g') AS cand_compact_address
                FROM joined
            )
            SELECT
                s1_id,
                candidate_id,
                label,
                CAST(s1_clean_name <> '' AND s1_clean_name = cand_clean_name AS UTINYINT) AS exact_clean_name,
                CAST(s1_sorted_name <> '' AND s1_sorted_name = cand_sorted_name AS UTINYINT) AS exact_sorted_name,
                CAST(
                    coalesce(
                        list_unique(list_intersect(s1_name_tokens, cand_name_tokens))::DOUBLE
                        / nullif(list_unique(list_distinct(list_concat(s1_name_tokens, cand_name_tokens))), 0),
                        0.0
                    ) AS FLOAT
                ) AS name_token_jaccard,
                CAST(s1_compact_address <> '' AND s1_compact_address = cand_compact_address AS UTINYINT) AS exact_compact_address,
                CAST(
                    coalesce(
                        list_unique(list_intersect(s1_address_tokens, cand_address_tokens))::DOUBLE
                        / nullif(list_unique(list_distinct(list_concat(s1_address_tokens, cand_address_tokens))), 0),
                        0.0
                    ) AS FLOAT
                ) AS address_token_jaccard,
                CAST(list_has_any(coalesce(s1_address_numbers, []::VARCHAR[]), coalesce(cand_address_numbers, []::VARCHAR[])) AS UTINYINT) AS address_numeric_agreement,
                CAST(coalesce(s1_country, '') <> '' AND coalesce(cand_country, '') <> '' AND s1_country = cand_country AS UTINYINT) AS country_equal,
                CAST(s1_clean_name = '' AS UTINYINT) AS name_missing_s1,
                CAST(cand_clean_name = '' AS UTINYINT) AS name_missing_candidate,
                CAST(s1_clean_address = '' AS UTINYINT) AS address_missing_s1,
                CAST(cand_clean_address = '' AS UTINYINT) AS address_missing_candidate,
                CAST(candidate_source = 'S2' AS UTINYINT) AS source_is_s2,
                CAST(candidate_source = 'S3' AS UTINYINT) AS source_is_s3
            FROM tokenized
        ) TO '{_sql_path(feature_path)}' (FORMAT PARQUET, COMPRESSION ZSTD, OVERWRITE_OR_IGNORE TRUE)
        """
    )
    return feature_path


def coverage_report(con: duckdb.DuckDBPyConnection) -> dict:
    row = con.execute(
        """
        WITH candidate_gt AS (
            SELECT DISTINCT g.s1_id, g.candidate_id
            FROM gt_pairs g
            JOIN candidates c ON g.s1_id = c.s1_id AND g.candidate_id = c.candidate_id
        ),
        s1_gt AS (
            SELECT s1_id, count(*) AS gt_links
            FROM gt_pairs
            GROUP BY s1_id
        ),
        s1_hit AS (
            SELECT s1_id, count(*) AS hit_links
            FROM candidate_gt
            GROUP BY s1_id
        )
        SELECT
            (SELECT count(*) FROM gt_pairs) AS total_true_gt_links,
            (SELECT count(*) FROM candidate_gt) AS true_gt_links_present_in_candidates,
            (SELECT count(*) FROM candidates c JOIN gt_pairs g ON c.s1_id = g.s1_id AND c.candidate_id = g.candidate_id) AS positive_candidate_count,
            (SELECT count(*) FROM candidates c LEFT JOIN gt_pairs g ON c.s1_id = g.s1_id AND c.candidate_id = g.candidate_id WHERE g.candidate_id IS NULL) AS negative_candidate_count,
            (SELECT count(*) FROM s1_gt) AS s1_entities_with_gt_matches,
            (SELECT count(*) FROM s1_gt l LEFT JOIN s1_hit h USING (s1_id) WHERE coalesce(h.hit_links, 0) = 0) AS s1_entities_whose_gt_matches_completely_missed
        """
    ).fetchdf().iloc[0].to_dict()
    report = {k: int(v) for k, v in row.items()}
    report["candidate_recall"] = (
        report["true_gt_links_present_in_candidates"] / report["total_true_gt_links"]
        if report["total_true_gt_links"]
        else 0.0
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--artifacts-dir", default="artifacts")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    artifacts_dir = Path(args.artifacts_dir)
    metrics_dir = artifacts_dir / "metrics"
    model_dir = artifacts_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print("load", flush=True)
    con = connect(cache_dir)
    try:
        prepare_views(con, cache_dir, Path(args.data_dir))
        print("label", flush=True)
        coverage = coverage_report(con)
        print(f"candidate recall {coverage['candidate_recall']:.6f}", flush=True)
        print("features", flush=True)
        feature_path = label_and_feature(con, cache_dir)
    finally:
        con.close()

    data = pd.read_parquet(feature_path)
    split_key = pd.util.hash_pandas_object(data["s1_id"], index=False) % 100
    train_mask = split_key < 80
    val_mask = ~train_mask

    x_train = data.loc[train_mask, FEATURE_COLUMNS].astype("float32")
    y_train = data.loc[train_mask, "label"].astype("int8")
    x_val = data.loc[val_mask, FEATURE_COLUMNS].astype("float32")
    y_val = data.loc[val_mask, "label"].astype("int8")
    scale_pos_weight = float(max(1.0, (y_train == 0).sum() / max(1, (y_train == 1).sum())))

    print("train", flush=True)
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

    val_prob = model.predict_proba(x_val)[:, 1]
    val_eval = data.loc[val_mask, ["s1_id", "candidate_id", "label"]].copy()
    val_eval["probability"] = val_prob

    print("threshold results", flush=True)
    threshold_rows = []
    y_val_np = y_val.to_numpy()
    for threshold in THRESHOLDS:
        row = _metrics(y_val_np, val_prob, threshold)
        row["entity_macro_f0_5"] = _macro_entity_f05(val_eval, threshold)
        threshold_rows.append(row)
        print(
            f"{threshold}: precision={row['precision']:.6f} recall={row['recall']:.6f} f0.5={row['f0_5']:.6f}",
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
        "feature_columns": FEATURE_COLUMNS,
        "thresholds": threshold_rows,
        "best_threshold": best,
    }
    model.save_model(model_dir / "xgb_entity_match.json")
    _write_json(metrics_dir / "model_metrics.json", metrics)
    _write_json(metrics_dir / "best_threshold.json", best)


if __name__ == "__main__":
    main()
