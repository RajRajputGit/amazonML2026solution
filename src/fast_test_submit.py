from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from xgboost import XGBClassifier


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

NORMALIZE_SQL = r"""
WITH raw AS (
    SELECT
        CAST(entity_id AS VARCHAR) AS entity_id,
        CAST(business_name AS VARCHAR) AS business_name,
        CAST(business_address AS VARCHAR) AS business_address,
        CAST(country AS VARCHAR) AS country
    FROM read_csv('{src_path}', delim = '\t', header = true, all_varchar = true)
),
base AS (
    SELECT
        entity_id,
        country,
        trim(regexp_replace(regexp_replace(lower(coalesce(business_name, '')), '&', ' and ', 'g'), '[^a-z0-9 ]+', ' ', 'g')) AS clean_name_raw,
        trim(regexp_replace(regexp_replace(lower(coalesce(business_address, '')), '&', ' and ', 'g'), '[^a-z0-9 ]+', ' ', 'g')) AS clean_address_raw
    FROM raw
),
cleaned AS (
    SELECT
        entity_id,
        country,
        trim(regexp_replace(clean_name_raw, '\s+', ' ', 'g')) AS clean_name,
        trim(regexp_replace(clean_address_raw, '\s+', ' ', 'g')) AS clean_address
    FROM base
)
SELECT
    entity_id,
    country,
    clean_name,
    array_to_string(list_sort(list_filter(string_split(clean_name, ' '), x -> length(x) > 1)), ' ') AS sorted_name,
    array_to_string(list_filter(string_split(clean_name, ' '), x -> length(x) > 1), ' ') AS significant_tokens,
    regexp_extract_all(clean_address, '[0-9]+') AS address_numbers,
    clean_address
FROM cleaned
"""


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def connect(cache_dir: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("PRAGMA memory_limit='4GB'")
    con.execute("PRAGMA threads=4")
    con.execute(f"PRAGMA temp_directory='{_sql_path(cache_dir / 'duckdb_tmp')}'")
    return con


def normalize_test(con: duckdb.DuckDBPyConnection, data_dir: Path, cache_dir: Path) -> None:
    test_dir = data_dir / "test"
    specs = {
        "s1": "test_source1.tsv",
        "s2": "test_source2.tsv",
        "s3": "test_source3.tsv",
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    for label, filename in specs.items():
        src = test_dir / filename
        out = cache_dir / f"test_{label}_normalized.parquet"
        if not src.exists():
            raise FileNotFoundError(str(src))
        sql = NORMALIZE_SQL.format(src_path=_sql_path(src))
        con.execute(f"COPY ({sql}) TO '{_sql_path(out)}' (FORMAT PARQUET, COMPRESSION ZSTD, OVERWRITE_OR_IGNORE TRUE)")


def prepare_tables(con: duckdb.DuckDBPyConnection, cache_dir: Path) -> None:
    for label in ("s1", "s2", "s3"):
        path = cache_dir / f"test_{label}_normalized.parquet"
        if not path.exists():
            raise FileNotFoundError(str(path))
        con.execute(f"CREATE OR REPLACE VIEW {label} AS SELECT * FROM read_parquet('{_sql_path(path)}')")
    con.execute("CREATE OR REPLACE TABLE s1_work AS SELECT row_number() OVER () AS rn, * FROM s1")
    con.execute("CREATE OR REPLACE TABLE others AS SELECT 'S2' AS candidate_source, * FROM s2 UNION ALL SELECT 'S3' AS candidate_source, * FROM s3")
    con.execute(
        """
        CREATE OR REPLACE TABLE other_clean_names AS
        SELECT o.*
        FROM others o
        JOIN (
            SELECT country, clean_name
            FROM others
            WHERE clean_name <> '' AND length(clean_name) >= 4
            GROUP BY country, clean_name
            HAVING count(*) <= 20
        ) b USING (country, clean_name)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE other_sorted_names AS
        SELECT o.*
        FROM others o
        JOIN (
            SELECT country, sorted_name
            FROM others
            WHERE sorted_name <> '' AND length(sorted_name) >= 4
            GROUP BY country, sorted_name
            HAVING count(*) <= 20
        ) b USING (country, sorted_name)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE other_compact_addresses AS
        WITH keys AS (
            SELECT entity_id, candidate_source, country, regexp_replace(clean_address, '\\s+', '', 'g') AS compact_address
            FROM others
        ),
        capped AS (
            SELECT country, compact_address
            FROM keys
            WHERE compact_address <> '' AND length(compact_address) >= 8
            GROUP BY country, compact_address
            HAVING count(*) <= 20
        )
        SELECT k.*
        FROM keys k
        JOIN capped c USING (country, compact_address)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE other_name_rare_tokens AS
        WITH tokens AS (
            SELECT entity_id, candidate_source, country, token
            FROM others, UNNEST(string_split(clean_name, ' ')) AS t(token)
            WHERE length(token) >= 4
        ),
        capped AS (
            SELECT country, token
            FROM tokens
            GROUP BY country, token
            HAVING count(*) <= 20
        )
        SELECT t.*
        FROM tokens t
        JOIN capped c USING (country, token)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE other_name_prefixes AS
        WITH keys AS (
            SELECT entity_id, candidate_source, country, left(regexp_replace(clean_name, '[^a-z0-9]', '', 'g'), 6) AS key
            FROM others
        ),
        capped AS (
            SELECT country, key
            FROM keys
            WHERE length(key) >= 6
            GROUP BY country, key
            HAVING count(*) <= 20
        )
        SELECT k.*
        FROM keys k
        JOIN capped c USING (country, key)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE other_name_suffixes AS
        WITH keys AS (
            SELECT entity_id, candidate_source, country, right(regexp_replace(clean_name, '[^a-z0-9]', '', 'g'), 6) AS key
            FROM others
        ),
        capped AS (
            SELECT country, key
            FROM keys
            WHERE length(key) >= 6
            GROUP BY country, key
            HAVING count(*) <= 20
        )
        SELECT k.*
        FROM keys k
        JOIN capped c USING (country, key)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE other_address_rare_tokens AS
        WITH tokens AS (
            SELECT entity_id, candidate_source, country, token
            FROM others, UNNEST(string_split(clean_address, ' ')) AS t(token)
            WHERE length(token) >= 4
        ),
        capped AS (
            SELECT country, token
            FROM tokens
            GROUP BY country, token
            HAVING count(*) <= 15
        )
        SELECT t.*
        FROM tokens t
        JOIN capped c USING (country, token)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE other_address_num_rare_tokens AS
        WITH nums AS (
            SELECT DISTINCT o.entity_id, o.candidate_source, o.country, n.num
            FROM others o, UNNEST(o.address_numbers) AS n(num)
            WHERE length(n.num) >= 2
        ),
        keys AS (
            SELECT n.entity_id, n.candidate_source, n.country, n.num, r.token
            FROM nums n
            JOIN other_address_rare_tokens r ON n.entity_id = r.entity_id
        ),
        capped AS (
            SELECT country, num, token
            FROM keys
            GROUP BY country, num, token
            HAVING count(*) <= 20
        )
        SELECT k.*
        FROM keys k
        JOIN capped c USING (country, num, token)
        """
    )


def write_candidate_chunk(con: duckdb.DuckDBPyConnection, start_rn: int, end_rn: int, chunk_path: Path) -> int:
    start_rn = int(start_rn)
    end_rn = int(end_rn)
    con.execute(f"CREATE OR REPLACE TEMP VIEW s1_chunk AS SELECT * FROM s1_work WHERE rn BETWEEN {start_rn} AND {end_rn}")
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE s1_name_tokens AS
        SELECT entity_id, country, token
        FROM s1_chunk, UNNEST(string_split(clean_name, ' ')) AS t(token)
        WHERE length(token) >= 4
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE s1_address_tokens AS
        SELECT entity_id, country, token
        FROM s1_chunk, UNNEST(string_split(clean_address, ' ')) AS t(token)
        WHERE length(token) >= 4
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE s1_address_num_tokens AS
        SELECT DISTINCT s.entity_id, s.country, n.num, t.token
        FROM s1_chunk s,
             UNNEST(s.address_numbers) AS n(num),
             UNNEST(string_split(s.clean_address, ' ')) AS t(token)
        WHERE length(n.num) >= 2 AND length(t.token) >= 4
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE chunk_candidates AS
        SELECT s.entity_id AS s1_id, o.entity_id AS candidate_id, o.candidate_source, 'exact_clean_name' AS blocking_method
        FROM s1_chunk s
        JOIN other_clean_names o
          ON (s.country = '' OR o.country = '' OR s.country = o.country)
         AND s.clean_name <> '' AND length(s.clean_name) >= 4 AND s.clean_name = o.clean_name
        UNION ALL
        SELECT s.entity_id, o.entity_id, o.candidate_source, 'sorted_name'
        FROM s1_chunk s
        JOIN other_sorted_names o
          ON (s.country = '' OR o.country = '' OR s.country = o.country)
         AND s.sorted_name <> '' AND length(s.sorted_name) >= 4 AND s.sorted_name = o.sorted_name
        UNION ALL
        SELECT s.entity_id, o.entity_id, o.candidate_source, 'compact_address'
        FROM s1_chunk s
        JOIN other_compact_addresses o
          ON (s.country = '' OR o.country = '' OR s.country = o.country)
         AND regexp_replace(s.clean_address, '\\s+', '', 'g') <> ''
         AND length(regexp_replace(s.clean_address, '\\s+', '', 'g')) >= 8
         AND regexp_replace(s.clean_address, '\\s+', '', 'g') = o.compact_address
        UNION ALL
        SELECT s.entity_id, o.entity_id, o.candidate_source, 'rare_name_token'
        FROM s1_name_tokens s
        JOIN other_name_rare_tokens o
          ON (s.country = '' OR o.country = '' OR s.country = o.country) AND s.token = o.token
        UNION ALL
        SELECT s.entity_id, o.entity_id, o.candidate_source, 'name_prefix'
        FROM s1_chunk s
        JOIN other_name_prefixes o
          ON (s.country = '' OR o.country = '' OR s.country = o.country)
         AND left(regexp_replace(s.clean_name, '[^a-z0-9]', '', 'g'), 6) = o.key
        WHERE length(left(regexp_replace(s.clean_name, '[^a-z0-9]', '', 'g'), 6)) >= 6
        UNION ALL
        SELECT s.entity_id, o.entity_id, o.candidate_source, 'name_suffix'
        FROM s1_chunk s
        JOIN other_name_suffixes o
          ON (s.country = '' OR o.country = '' OR s.country = o.country)
         AND right(regexp_replace(s.clean_name, '[^a-z0-9]', '', 'g'), 6) = o.key
        WHERE length(right(regexp_replace(s.clean_name, '[^a-z0-9]', '', 'g'), 6)) >= 6
        UNION ALL
        SELECT s.entity_id, o.entity_id, o.candidate_source, 'rare_address_token'
        FROM s1_address_tokens s
        JOIN other_address_rare_tokens o
          ON (s.country = '' OR o.country = '' OR s.country = o.country) AND s.token = o.token
        UNION ALL
        SELECT DISTINCT s.entity_id, o.entity_id, o.candidate_source, 'address_number_rare_token'
        FROM s1_address_num_tokens s
        JOIN other_address_num_rare_tokens o
          ON (s.country = '' OR o.country = '' OR s.country = o.country)
         AND s.num = o.num AND s.token = o.token
        """
    )
    count = con.execute("SELECT count(DISTINCT s1_id || '|' || candidate_id) FROM chunk_candidates").fetchone()[0]
    con.execute(
        f"""
        COPY (
            SELECT s1_id, candidate_id, any_value(candidate_source) AS candidate_source, min(blocking_method) AS blocking_method
            FROM chunk_candidates
            GROUP BY s1_id, candidate_id
        ) TO '{_sql_path(chunk_path)}' (FORMAT PARQUET, COMPRESSION ZSTD, OVERWRITE_OR_IGNORE TRUE)
        """
    )
    for name in ("chunk_candidates", "s1_address_num_tokens", "s1_address_tokens", "s1_name_tokens"):
        con.execute(f"DROP TABLE {name}")
    con.execute("DROP VIEW s1_chunk")
    return int(count)


def build_candidates(con: duckdb.DuckDBPyConnection, cache_dir: Path, chunk_size: int) -> Path:
    chunk_dir = cache_dir / "test_candidate_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    total = con.execute("SELECT count(*) FROM s1_work").fetchone()[0]
    paths = []
    for idx, start in enumerate(range(1, total + 1, chunk_size), start=1):
        end = min(start + chunk_size - 1, total)
        path = chunk_dir / f"chunk_{idx:06d}.parquet"
        count = write_candidate_chunk(con, start, end, path)
        paths.append(path)
        print(f"rows processed {end}; candidates {count}", flush=True)
    out_path = cache_dir / "test_candidates.parquet"
    parquet_list = ", ".join(f"'{_sql_path(path)}'" for path in paths)
    con.execute(
        f"""
        COPY (
            SELECT s1_id, candidate_id, any_value(candidate_source) AS candidate_source, min(blocking_method) AS blocking_method
            FROM read_parquet([{parquet_list}])
            GROUP BY s1_id, candidate_id
        ) TO '{_sql_path(out_path)}' (FORMAT PARQUET, COMPRESSION ZSTD, OVERWRITE_OR_IGNORE TRUE)
        """
    )
    return out_path


def build_features(con: duckdb.DuckDBPyConnection, cache_dir: Path) -> Path:
    feature_path = cache_dir / "test_features.parquet"
    con.execute(
        f"""
        CREATE OR REPLACE VIEW candidates AS SELECT * FROM read_parquet('{_sql_path(cache_dir / 'test_candidates.parquet')}')
        """
    )
    con.execute(
        f"""
        COPY (
            WITH joined AS (
                SELECT
                    c.s1_id,
                    c.candidate_id,
                    coalesce(c.candidate_source, o.inferred_source) AS candidate_source,
                    coalesce(s.country, '') AS s1_country,
                    coalesce(o.country, '') AS cand_country,
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
                JOIN (SELECT 'S2' AS inferred_source, * FROM s2 UNION ALL SELECT 'S3' AS inferred_source, * FROM s3) o
                  ON c.candidate_id = o.entity_id
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
                CAST(s1_clean_name <> '' AND s1_clean_name = cand_clean_name AS UTINYINT) AS exact_clean_name,
                CAST(s1_sorted_name <> '' AND s1_sorted_name = cand_sorted_name AS UTINYINT) AS exact_sorted_name,
                CAST(coalesce(list_unique(list_intersect(s1_name_tokens, cand_name_tokens))::DOUBLE / nullif(list_unique(list_distinct(list_concat(s1_name_tokens, cand_name_tokens))), 0), 0.0) AS FLOAT) AS name_token_jaccard,
                CAST(s1_compact_address <> '' AND s1_compact_address = cand_compact_address AS UTINYINT) AS exact_compact_address,
                CAST(coalesce(list_unique(list_intersect(s1_address_tokens, cand_address_tokens))::DOUBLE / nullif(list_unique(list_distinct(list_concat(s1_address_tokens, cand_address_tokens))), 0), 0.0) AS FLOAT) AS address_token_jaccard,
                CAST(list_has_any(coalesce(s1_address_numbers, []::VARCHAR[]), coalesce(cand_address_numbers, []::VARCHAR[])) AS UTINYINT) AS address_numeric_agreement,
                CAST(s1_country <> '' AND cand_country <> '' AND s1_country = cand_country AS UTINYINT) AS country_equal,
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


def load_threshold(path: Path) -> float:
    if not path.exists():
        return 0.90
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data.get("threshold", 0.90))


def predict(con: duckdb.DuckDBPyConnection, feature_path: Path, model_path: Path, threshold: float, output_dir: Path) -> None:
    model = XGBClassifier()
    model.load_model(model_path)
    predictions_path = output_dir / "test_predictions.parquet"
    pred_dir = output_dir / "prediction_chunks"
    pred_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"CREATE OR REPLACE VIEW features AS SELECT * FROM read_parquet('{_sql_path(feature_path)}')")
    total = con.execute("SELECT count(*) FROM features").fetchone()[0]
    offset = 0
    batch_size = 500000
    pred_paths = []
    while offset < total:
        df = con.execute(f"SELECT * FROM features LIMIT {batch_size} OFFSET {offset}").fetchdf()
        probs = model.predict_proba(df[FEATURE_COLUMNS].astype("float32"))[:, 1]
        pred = pd.DataFrame({"s1_id": df["s1_id"], "candidate_id": df["candidate_id"], "probability": probs})
        con.register("pred_batch", pred)
        target = pred_dir / f"pred_batch_{offset // batch_size:06d}.parquet"
        con.execute(f"COPY pred_batch TO '{_sql_path(target)}' (FORMAT PARQUET, COMPRESSION ZSTD, OVERWRITE_OR_IGNORE TRUE)")
        con.unregister("pred_batch")
        pred_paths.append(target)
        offset += len(df)
    parquet_list = ", ".join(f"'{_sql_path(path)}'" for path in pred_paths)
    con.execute(
        f"""
        COPY (
            SELECT s1_id, candidate_id, probability
            FROM read_parquet([{parquet_list}])
            WHERE probability >= {float(threshold)}
        ) TO '{_sql_path(predictions_path)}' (FORMAT PARQUET, COMPRESSION ZSTD, OVERWRITE_OR_IGNORE TRUE)
        """
    )


def write_tsv_outputs(con: duckdb.DuckDBPyConnection, output_dir: Path) -> tuple[int, int, int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    matching_path = output_dir / "matching_results.tsv"
    candidate_path = output_dir / "candidate_pairs.tsv"
    con.execute(f"CREATE OR REPLACE VIEW predictions AS SELECT * FROM read_parquet('{_sql_path(output_dir / 'test_predictions.parquet')}')")
    test_s1_count = con.execute("SELECT count(*) FROM s1").fetchone()[0]
    candidate_pair_count = con.execute("SELECT count(*) FROM candidates").fetchone()[0]
    predicted_matched_s1_count = con.execute("SELECT count(DISTINCT s1_id) FROM predictions").fetchone()[0]
    singleton_count = int(test_s1_count - predicted_matched_s1_count)
    con.execute(
        f"""
        COPY (
            SELECT
                s.entity_id AS source1_entity_id,
                coalesce(string_agg(DISTINCT p.candidate_id, ',' ORDER BY p.candidate_id), '') AS matched_entity_ids
            FROM s1 s
            LEFT JOIN predictions p ON s.entity_id = p.s1_id
            GROUP BY s.entity_id
            ORDER BY s.entity_id
        ) TO '{_sql_path(matching_path)}' (HEADER, DELIMITER '\t')
        """
    )
    con.execute(
        f"""
        COPY (
            SELECT
                s.entity_id AS source1_entity_id,
                coalesce(string_agg(DISTINCT c.candidate_id, ',' ORDER BY c.candidate_id), '') AS candidate_entity_ids
            FROM s1 s
            LEFT JOIN candidates c ON s.entity_id = c.s1_id
            GROUP BY s.entity_id
            ORDER BY s.entity_id
        ) TO '{_sql_path(candidate_path)}' (HEADER, DELIMITER '\t')
        """
    )
    return int(test_s1_count), int(candidate_pair_count), int(predicted_matched_s1_count), singleton_count


def validate_and_zip(data_dir: Path, output_dir: Path) -> tuple[str, Path]:
    cmd = [
        sys.executable,
        "utils/validate_submission.py",
        "--matching",
        str(output_dir / "matching_results.tsv"),
        "--candidate",
        str(output_dir / "candidate_pairs.tsv"),
        "--test-dir",
        str(data_dir / "test"),
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    validator_result = "PASS" if result.returncode == 0 else "FAIL"
    if result.stdout:
        print(result.stdout, flush=True)
    if result.stderr:
        print(result.stderr, flush=True)
    zip_path = output_dir / "submission.zip"
    if result.returncode == 0:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(output_dir / "matching_results.tsv", "output/matching_results.tsv")
            zf.write(output_dir / "candidate_pairs.tsv", "output/candidate_pairs.tsv")
    return validator_result, zip_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--chunk-size", type=int, default=10000)
    args = parser.parse_args()

    started = time.time()
    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    artifacts_dir = Path(args.artifacts_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    con = connect(cache_dir)
    try:
        print("normalize test", flush=True)
        normalize_test(con, data_dir, cache_dir)
        print("prepare tables", flush=True)
        prepare_tables(con, cache_dir)
        print("build candidates", flush=True)
        build_candidates(con, cache_dir, args.chunk_size)
        print("features", flush=True)
        feature_path = build_features(con, cache_dir)
        threshold = load_threshold(artifacts_dir / "metrics" / "best_threshold.json")
        print(f"predict threshold {threshold}", flush=True)
        predict(con, feature_path, artifacts_dir / "model" / "xgb_entity_match.json", threshold, output_dir)
        test_s1_count, candidate_pair_count, predicted_matched_s1_count, singleton_count = write_tsv_outputs(con, output_dir)
    finally:
        con.close()

    validator_result, zip_path = validate_and_zip(data_dir, output_dir)
    print(f"test S1 count {test_s1_count}", flush=True)
    print(f"candidate pair count {candidate_pair_count}", flush=True)
    print(f"predicted matched S1 count {predicted_matched_s1_count}", flush=True)
    print(f"singleton count {singleton_count}", flush=True)
    print(f"validator result {validator_result}", flush=True)
    print(f"final ZIP path {zip_path if validator_result == 'PASS' else ''}", flush=True)
    print(f"runtime seconds {round(time.time() - started, 1)}", flush=True)


if __name__ == "__main__":
    main()
