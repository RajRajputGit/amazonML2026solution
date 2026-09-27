from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


SOURCE_FILES = {
    "s1": "train_source1.tsv",
    "s2": "train_source2.tsv",
    "s3": "train_source3.tsv",
}


NORMALIZE_SQL = r"""
WITH raw AS (
    SELECT
        CAST(entity_id AS VARCHAR) AS entity_id,
        CAST(business_name AS VARCHAR) AS business_name,
        CAST(business_address AS VARCHAR) AS business_address,
        CAST(country AS VARCHAR) AS country
    FROM read_csv($path, delim = '\t', header = true, all_varchar = true)
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
),
tokens AS (
    SELECT
        entity_id,
        country,
        clean_name,
        array_to_string(list_sort(list_filter(string_split(clean_name, ' '), x -> length(x) > 1)), ' ') AS sorted_name,
        array_to_string(list_filter(string_split(clean_name, ' '), x -> length(x) > 1), ' ') AS significant_tokens,
        regexp_extract_all(clean_address, '[0-9]+') AS address_numbers,
        clean_address
    FROM cleaned
)
SELECT * FROM tokens
"""


def connect(cache_dir: Path) -> duckdb.DuckDBPyConnection:
    cache_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(cache_dir / "fast_prepare.duckdb"))
    con.execute("PRAGMA memory_limit='3GB'")
    con.execute("PRAGMA threads=4")
    con.execute(f"PRAGMA temp_directory='{str(cache_dir / 'duckdb_tmp').replace(chr(92), '/')}'")
    return con


def normalize_sources(con: duckdb.DuckDBPyConnection, data_dir: Path, cache_dir: Path) -> None:
    for label, filename in SOURCE_FILES.items():
        src_path = data_dir / "train" / filename
        out_path = cache_dir / f"train_{label}_normalized.parquet"
        print(f"normalize {label}", flush=True)
        con.execute(f"COPY ({NORMALIZE_SQL}) TO $out_path (FORMAT PARQUET, COMPRESSION ZSTD)", {"path": str(src_path), "out_path": str(out_path)})


def prepare_tables(con: duckdb.DuckDBPyConnection, cache_dir: Path) -> None:
    for label in SOURCE_FILES:
        path = cache_dir / f"train_{label}_normalized.parquet"
        parquet_path = str(path).replace("'", "''")
        con.execute(f"CREATE OR REPLACE VIEW {label} AS SELECT * FROM read_parquet('{parquet_path}')")

    con.execute("CREATE OR REPLACE TABLE s1_work AS SELECT row_number() OVER () AS rn, * FROM s1")
    con.execute("CREATE OR REPLACE TABLE others AS SELECT 'S2' AS candidate_source, * FROM s2 UNION ALL SELECT 'S3' AS candidate_source, * FROM s3")
    con.execute("CREATE OR REPLACE TABLE other_name_tokens AS SELECT entity_id, candidate_source, country, token FROM others, UNNEST(string_split(significant_tokens, ' ')) AS t(token) WHERE length(token) > 1")
    con.execute("CREATE OR REPLACE TABLE rare_name_tokens AS SELECT token FROM other_name_tokens GROUP BY token HAVING count(*) BETWEEN 2 AND 500")
    con.execute("CREATE OR REPLACE TABLE other_rare_tokens AS SELECT o.* FROM other_name_tokens o JOIN rare_name_tokens r USING (token)")
    con.execute("CREATE OR REPLACE TABLE other_address_numbers AS SELECT entity_id, candidate_source, country, num, split_part(significant_tokens, ' ', 1) AS first_token FROM others, UNNEST(address_numbers) AS a(num) WHERE length(num) >= 2")

    con.execute(
        """
        CREATE OR REPLACE TABLE train_candidates (
            s1_id VARCHAR,
            candidate_id VARCHAR,
            candidate_source VARCHAR,
            blocking_method VARCHAR
        )
        """
    )


def insert_chunk(con: duckdb.DuckDBPyConnection, start_rn: int, end_rn: int) -> None:
    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW s1_chunk AS
        SELECT * FROM s1_work WHERE rn BETWEEN $start_rn AND $end_rn
        """,
        {"start_rn": start_rn, "end_rn": end_rn},
    )
    con.execute(
        """
        INSERT INTO train_candidates
        SELECT s.entity_id, o.entity_id, o.candidate_source, 'exact_clean_name'
        FROM s1_chunk s
        JOIN others o
          ON s.country = o.country AND s.clean_name <> '' AND s.clean_name = o.clean_name
        """
    )
    con.execute(
        """
        INSERT INTO train_candidates
        SELECT s.entity_id, o.entity_id, o.candidate_source, 'sorted_name'
        FROM s1_chunk s
        JOIN others o
          ON s.country = o.country AND s.sorted_name <> '' AND s.sorted_name = o.sorted_name
        """
    )
    con.execute(
        """
        INSERT INTO train_candidates
        SELECT s.entity_id, o.entity_id, o.candidate_source, 'first_significant_tokens'
        FROM s1_chunk s
        JOIN others o
          ON s.country = o.country
         AND split_part(s.significant_tokens, ' ', 1) <> ''
         AND split_part(s.significant_tokens, ' ', 1) = split_part(o.significant_tokens, ' ', 1)
         AND split_part(s.significant_tokens, ' ', 2) = split_part(o.significant_tokens, ' ', 2)
        """
    )
    con.execute(
        """
        INSERT INTO train_candidates
        SELECT s.entity_id, t.entity_id, t.candidate_source, 'rare_name_token'
        FROM s1_chunk s,
             UNNEST(string_split(s.significant_tokens, ' ')) AS st(token)
        JOIN other_rare_tokens t
          ON s.country = t.country AND st.token = t.token
        WHERE length(st.token) > 1
        """
    )
    con.execute(
        """
        INSERT INTO train_candidates
        SELECT s.entity_id, a.entity_id, a.candidate_source, 'address_number_name_token'
        FROM s1_chunk s,
             UNNEST(s.address_numbers) AS sn(num)
        JOIN other_address_numbers a
          ON s.country = a.country
         AND sn.num = a.num
         AND split_part(s.significant_tokens, ' ', 1) <> ''
         AND split_part(s.significant_tokens, ' ', 1) = a.first_token
        WHERE length(sn.num) >= 2
        """
    )


def build_candidates(con: duckdb.DuckDBPyConnection, cache_dir: Path, chunk_size: int) -> None:
    total = con.execute("SELECT count(*) FROM s1_work").fetchone()[0]
    for start in range(1, total + 1, chunk_size):
        end = min(start + chunk_size - 1, total)
        print(f"block s1 rows {start}-{end}", flush=True)
        insert_chunk(con, start, end)
        con.execute("CHECKPOINT")

    out_path = cache_dir / "train_candidates.parquet"
    con.execute(
        """
        COPY (
            SELECT DISTINCT s1_id, candidate_id, candidate_source, blocking_method
            FROM train_candidates
        ) TO $out_path (FORMAT PARQUET, COMPRESSION ZSTD)
        """,
        {"out_path": str(out_path)},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--chunk-size", type=int, default=50000)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    con = connect(cache_dir)
    try:
        normalize_sources(con, data_dir, cache_dir)
        prepare_tables(con, cache_dir)
        build_candidates(con, cache_dir, args.chunk_size)
    finally:
        con.close()


if __name__ == "__main__":
    main()
