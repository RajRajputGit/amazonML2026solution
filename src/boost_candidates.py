from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def connect(cache_dir: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("PRAGMA memory_limit='4GB'")
    con.execute("PRAGMA threads=4")
    con.execute(f"PRAGMA temp_directory='{_sql_path(cache_dir / 'duckdb_tmp')}'")
    return con


def prepare_tables(con: duckdb.DuckDBPyConnection, cache_dir: Path) -> None:
    paths = {
        "existing_candidates": cache_dir / "train_candidates.parquet",
        "s1": cache_dir / "train_s1_normalized.parquet",
        "s2": cache_dir / "train_s2_normalized.parquet",
        "s3": cache_dir / "train_s3_normalized.parquet",
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(str(path))
        con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{_sql_path(path)}')")

    con.execute("CREATE OR REPLACE TABLE s1_work AS SELECT row_number() OVER () AS rn, * FROM s1")
    con.execute("CREATE OR REPLACE TABLE others AS SELECT 'S2' AS candidate_source, * FROM s2 UNION ALL SELECT 'S3' AS candidate_source, * FROM s3")
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
            SELECT
                entity_id,
                candidate_source,
                country,
                left(regexp_replace(clean_name, '[^a-z0-9]', '', 'g'), 6) AS key
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
            SELECT
                entity_id,
                candidate_source,
                country,
                right(regexp_replace(clean_name, '[^a-z0-9]', '', 'g'), 6) AS key
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
            SELECT DISTINCT
                o.entity_id,
                o.candidate_source,
                o.country,
                n.num
            FROM others o, UNNEST(o.address_numbers) AS n(num)
            WHERE length(n.num) >= 2
        ),
        keys AS (
            SELECT
                n.entity_id,
                n.candidate_source,
                n.country,
                n.num,
                r.token
            FROM nums n
            JOIN other_address_rare_tokens r
              ON n.entity_id = r.entity_id
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
        SELECT s.entity_id AS s1_id, o.entity_id AS candidate_id, o.candidate_source, 'rare_name_token' AS blocking_method
        FROM s1_name_tokens s
        JOIN other_name_rare_tokens o
          ON (s.country = '' OR o.country = '' OR s.country = o.country)
         AND s.token = o.token
        """
    )
    con.execute(
        """
        INSERT INTO chunk_candidates
        SELECT s.entity_id, o.entity_id, o.candidate_source, 'name_prefix'
        FROM s1_chunk s
        JOIN other_name_prefixes o
          ON (s.country = '' OR o.country = '' OR s.country = o.country)
         AND left(regexp_replace(s.clean_name, '[^a-z0-9]', '', 'g'), 6) = o.key
        WHERE length(left(regexp_replace(s.clean_name, '[^a-z0-9]', '', 'g'), 6)) >= 6
        """
    )
    con.execute(
        """
        INSERT INTO chunk_candidates
        SELECT s.entity_id, o.entity_id, o.candidate_source, 'name_suffix'
        FROM s1_chunk s
        JOIN other_name_suffixes o
          ON (s.country = '' OR o.country = '' OR s.country = o.country)
         AND right(regexp_replace(s.clean_name, '[^a-z0-9]', '', 'g'), 6) = o.key
        WHERE length(right(regexp_replace(s.clean_name, '[^a-z0-9]', '', 'g'), 6)) >= 6
        """
    )
    con.execute(
        """
        INSERT INTO chunk_candidates
        SELECT s.entity_id, o.entity_id, o.candidate_source, 'rare_address_token'
        FROM s1_address_tokens s
        JOIN other_address_rare_tokens o
          ON (s.country = '' OR o.country = '' OR s.country = o.country)
         AND s.token = o.token
        """
    )
    con.execute(
        """
        INSERT INTO chunk_candidates
        SELECT DISTINCT s.entity_id, o.entity_id, o.candidate_source, 'address_number_rare_token'
        FROM s1_address_num_tokens s
        JOIN other_address_num_rare_tokens o
          ON (s.country = '' OR o.country = '' OR s.country = o.country)
         AND s.num = o.num
         AND s.token = o.token
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE new_chunk_candidates AS
        SELECT
            c.s1_id,
            c.candidate_id,
            any_value(c.candidate_source) AS candidate_source,
            min(c.blocking_method) AS blocking_method
        FROM chunk_candidates c
        LEFT JOIN existing_candidates e
          ON c.s1_id = e.s1_id AND c.candidate_id = e.candidate_id
        WHERE e.candidate_id IS NULL
        GROUP BY c.s1_id, c.candidate_id
        """
    )
    added = con.execute("SELECT count(*) FROM new_chunk_candidates").fetchone()[0]
    con.execute(
        f"""
        COPY new_chunk_candidates
        TO '{_sql_path(chunk_path)}' (FORMAT PARQUET, COMPRESSION ZSTD, OVERWRITE_OR_IGNORE TRUE)
        """
    )
    con.execute("DROP TABLE new_chunk_candidates")
    con.execute("DROP TABLE chunk_candidates")
    con.execute("DROP TABLE s1_address_num_tokens")
    con.execute("DROP TABLE s1_address_tokens")
    con.execute("DROP TABLE s1_name_tokens")
    con.execute("DROP VIEW s1_chunk")
    return int(added)


def final_write(con: duckdb.DuckDBPyConnection, cache_dir: Path, chunk_paths: list[Path]) -> dict:
    existing_path = cache_dir / "train_candidates.parquet"
    boosted_path = cache_dir / "train_candidates_boosted.parquet"
    parquet_list = ", ".join(f"'{_sql_path(path)}'" for path in chunk_paths)
    old_pairs = con.execute("SELECT count(DISTINCT s1_id || '|' || candidate_id) FROM existing_candidates").fetchone()[0]

    if chunk_paths:
        con.execute(
            f"""
            COPY (
                SELECT
                    s1_id,
                    candidate_id,
                    any_value(candidate_source) AS candidate_source,
                    min(blocking_method) AS blocking_method
                FROM (
                    SELECT s1_id, candidate_id, candidate_source, blocking_method FROM existing_candidates
                    UNION ALL
                    SELECT s1_id, candidate_id, candidate_source, blocking_method FROM read_parquet([{parquet_list}])
                )
                GROUP BY s1_id, candidate_id
            ) TO '{_sql_path(boosted_path)}' (FORMAT PARQUET, COMPRESSION ZSTD, OVERWRITE_OR_IGNORE TRUE)
            """
        )
    else:
        con.execute(
            f"""
            COPY (
                SELECT s1_id, candidate_id, candidate_source, blocking_method
                FROM existing_candidates
            ) TO '{_sql_path(boosted_path)}' (FORMAT PARQUET, COMPRESSION ZSTD, OVERWRITE_OR_IGNORE TRUE)
            """
        )

    final_pairs = con.execute(
        f"SELECT count(DISTINCT s1_id || '|' || candidate_id), count(DISTINCT s1_id) FROM read_parquet('{_sql_path(boosted_path)}')"
    ).fetchone()
    return {
        "old_candidate_pairs": int(old_pairs),
        "new_candidate_pairs_added": int(final_pairs[0] - old_pairs),
        "final_candidate_pairs": int(final_pairs[0]),
        "average_candidates_per_s1": float(final_pairs[0] / final_pairs[1]) if final_pairs[1] else 0.0,
        "boosted_path": str(boosted_path),
        "existing_path": str(existing_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--chunk-size", type=int, default=10000)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    chunk_dir = cache_dir / "boost_candidate_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    con = connect(cache_dir)
    try:
        prepare_tables(con, cache_dir)
        total = con.execute("SELECT count(*) FROM s1_work").fetchone()[0]
        chunk_paths = []
        chunk_index = 1
        for start in range(1, total + 1, args.chunk_size):
            end = min(start + args.chunk_size - 1, total)
            chunk_path = chunk_dir / f"chunk_{chunk_index:06d}.parquet"
            added = write_chunk(con, start, end, chunk_path)
            if added:
                chunk_paths.append(chunk_path)
            print(f"rows processed {end}; new pairs added {added}", flush=True)
            chunk_index += 1
        metrics = final_write(con, cache_dir, chunk_paths)
    finally:
        con.close()

    Path(metrics["boosted_path"]).replace(Path(metrics["existing_path"]))
    print(f"old candidate pairs {metrics['old_candidate_pairs']}", flush=True)
    print(f"new candidate pairs added {metrics['new_candidate_pairs_added']}", flush=True)
    print(f"final candidate pairs {metrics['final_candidate_pairs']}", flush=True)
    print(f"average candidates per S1 {metrics['average_candidates_per_s1']:.3f}", flush=True)


if __name__ == "__main__":
    main()
