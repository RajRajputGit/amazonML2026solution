from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def connect(cache_dir: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("PRAGMA memory_limit='3GB'")
    con.execute("PRAGMA threads=4")
    con.execute(f"PRAGMA temp_directory='{_sql_path(cache_dir / 'duckdb_tmp')}'")
    return con


def prepare_cached_tables(con: duckdb.DuckDBPyConnection, cache_dir: Path) -> None:
    for label in ("s1", "s2", "s3"):
        path = cache_dir / f"train_{label}_normalized.parquet"
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
            SELECT
                entity_id,
                candidate_source,
                country,
                regexp_replace(clean_address, '\\s+', '', 'g') AS compact_address
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


def write_chunk(con: duckdb.DuckDBPyConnection, start_rn: int, end_rn: int, chunk_path: Path) -> int:
    start_rn = int(start_rn)
    end_rn = int(end_rn)
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW s1_chunk AS
        SELECT * FROM s1_work WHERE rn BETWEEN {start_rn} AND {end_rn}
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE chunk_candidates AS
        SELECT
            s.entity_id AS s1_id,
            o.entity_id AS candidate_id,
            o.candidate_source AS candidate_source,
            'exact_clean_name' AS blocking_method
        FROM s1_chunk s
        JOIN other_clean_names o
          ON (s.country = '' OR o.country = '' OR s.country = o.country)
         AND s.clean_name <> ''
         AND length(s.clean_name) >= 4
         AND s.clean_name = o.clean_name
        """
    )
    con.execute(
        """
        INSERT INTO chunk_candidates
        SELECT
            s.entity_id AS s1_id,
            o.entity_id AS candidate_id,
            o.candidate_source AS candidate_source,
            'sorted_name' AS blocking_method
        FROM s1_chunk s
        JOIN other_sorted_names o
          ON (s.country = '' OR o.country = '' OR s.country = o.country)
         AND s.sorted_name <> ''
         AND length(s.sorted_name) >= 4
         AND s.sorted_name = o.sorted_name
        """
    )
    con.execute(
        """
        INSERT INTO chunk_candidates
        SELECT
            s.entity_id AS s1_id,
            o.entity_id AS candidate_id,
            o.candidate_source AS candidate_source,
            'compact_address' AS blocking_method
        FROM s1_chunk s
        JOIN other_compact_addresses o
          ON (s.country = '' OR o.country = '' OR s.country = o.country)
         AND regexp_replace(s.clean_address, '\\s+', '', 'g') <> ''
         AND length(regexp_replace(s.clean_address, '\\s+', '', 'g')) >= 8
         AND regexp_replace(s.clean_address, '\\s+', '', 'g') = o.compact_address
        """
    )
    candidates_added = con.execute(
        """
        SELECT count(*)
        FROM (
            SELECT s1_id, candidate_id
            FROM chunk_candidates
            GROUP BY s1_id, candidate_id
        )
        """
    ).fetchone()[0]
    con.execute(
        f"""
        COPY (
            SELECT
                s1_id,
                candidate_id,
                any_value(candidate_source) AS candidate_source,
                min(blocking_method) AS blocking_method
            FROM chunk_candidates
            GROUP BY s1_id, candidate_id
        ) TO '{_sql_path(chunk_path)}' (FORMAT PARQUET, COMPRESSION ZSTD, OVERWRITE_OR_IGNORE TRUE)
        """
    )
    con.execute("DROP TABLE chunk_candidates")
    con.execute("DROP VIEW s1_chunk")
    return int(candidates_added)


def write_outputs(con: duckdb.DuckDBPyConnection, cache_dir: Path, metrics_dir: Path, chunk_paths: list[Path]) -> None:
    out_path = cache_dir / "train_candidates.parquet"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    parquet_list = ", ".join(f"'{_sql_path(path)}'" for path in chunk_paths)
    con.execute(
        f"""
        COPY (
            SELECT DISTINCT s1_id, candidate_id, candidate_source, blocking_method
            FROM read_parquet([{parquet_list}])
        ) TO '{_sql_path(out_path)}' (FORMAT PARQUET, COMPRESSION ZSTD, OVERWRITE_OR_IGNORE TRUE)
        """
    )
    metrics = con.execute(
        f"""
        SELECT
            count(*) AS candidate_rows,
            count(DISTINCT s1_id || '|' || candidate_id) AS distinct_candidate_pairs,
            count(DISTINCT s1_id) AS covered_s1,
            count(DISTINCT candidate_id) AS covered_candidates
        FROM read_parquet('{_sql_path(out_path)}')
        """
    ).fetchdf().iloc[0].to_dict()
    by_method = con.execute(
        f"""
        SELECT blocking_method, count(*) AS rows, count(DISTINCT s1_id || '|' || candidate_id) AS distinct_pairs
        FROM read_parquet('{_sql_path(out_path)}')
        GROUP BY blocking_method
        ORDER BY blocking_method
        """
    ).fetchdf().to_dict("records")
    metrics = {k: int(v) for k, v in metrics.items()}
    metrics["by_method"] = [{k: (int(v) if k != "blocking_method" else v) for k, v in row.items()} for row in by_method]
    (metrics_dir / "fast_blocking.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--chunk-size", type=int, default=10000)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    print("connect", flush=True)
    con = connect(cache_dir)
    try:
        print("prepare cached tables", flush=True)
        prepare_cached_tables(con, cache_dir)
        print("prepare complete", flush=True)
        total = con.execute("SELECT count(*) FROM s1_work").fetchone()[0]
        chunk_dir = cache_dir / "candidate_chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_paths = []
        started = time.time()
        print("start blocking", flush=True)
        chunk_index = 1
        for start in range(1, total + 1, args.chunk_size):
            end = min(start + args.chunk_size - 1, total)
            chunk_path = chunk_dir / f"chunk_{chunk_index:06d}.parquet"
            candidates_added = write_chunk(con, start, end, chunk_path)
            chunk_paths.append(chunk_path)
            print(
                f"rows processed {end}; UNIQUE candidates added {candidates_added}; avg candidates per S1 {round(candidates_added / (end - start + 1), 3)}; elapsed seconds {round(time.time() - started, 1)}",
                flush=True,
            )
            chunk_index += 1
        write_outputs(con, cache_dir, Path(args.artifacts_dir) / "metrics", chunk_paths)
    finally:
        con.close()


if __name__ == "__main__":
    main()
