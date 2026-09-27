from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def connect(cache_dir: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("PRAGMA memory_limit='4GB'")
    con.execute("PRAGMA threads=4")
    con.execute(f"PRAGMA temp_directory='{_sql_path(cache_dir / 'duckdb_tmp')}'")
    return con


def prepare_views(con: duckdb.DuckDBPyConnection, cache_dir: Path) -> None:
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
    con.execute("CREATE OR REPLACE VIEW others AS SELECT 'S2' AS candidate_source, * FROM s2 UNION ALL SELECT 'S3' AS candidate_source, * FROM s3")


def load_country_frame(con: duckdb.DuckDBPyConnection, table: str, country: str, field: str) -> pd.DataFrame:
    return con.execute(
        f"""
        SELECT entity_id, country, {field} AS text
        FROM {table}
        WHERE country = ? AND coalesce({field}, '') <> ''
        """,
        [country],
    ).fetchdf()


def load_country_others(con: duckdb.DuckDBPyConnection, country: str, field: str) -> pd.DataFrame:
    return con.execute(
        f"""
        SELECT entity_id, candidate_source, country, {field} AS text
        FROM others
        WHERE country = ? AND coalesce({field}, '') <> ''
        """,
        [country],
    ).fetchdf()


def retrieve_for_country(
    left: pd.DataFrame,
    right: pd.DataFrame,
    method: str,
    top_k: int,
    min_similarity: float,
    max_features: int,
    batch_size: int,
) -> pd.DataFrame:
    if left.empty or right.empty:
        return pd.DataFrame(columns=["s1_id", "candidate_id", "candidate_source", "blocking_method", "name_tfidf_cosine", "address_tfidf_cosine"])

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=max_features,
        dtype=np.float32,
    )
    try:
        right_matrix = vectorizer.fit_transform(right["text"].fillna("").astype(str))
    except ValueError:
        return pd.DataFrame(columns=["s1_id", "candidate_id", "candidate_source", "blocking_method", "name_tfidf_cosine", "address_tfidf_cosine"])
    if right_matrix.shape[1] == 0:
        return pd.DataFrame(columns=["s1_id", "candidate_id", "candidate_source", "blocking_method", "name_tfidf_cosine", "address_tfidf_cosine"])

    neighbors = NearestNeighbors(
        n_neighbors=min(top_k, right_matrix.shape[0]),
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    neighbors.fit(right_matrix)

    frames = []
    score_col = "name_tfidf_cosine" if method == "name_cosine" else "address_tfidf_cosine"
    null_col = "address_tfidf_cosine" if method == "name_cosine" else "name_tfidf_cosine"
    for start in range(0, len(left), batch_size):
        batch = left.iloc[start : start + batch_size]
        left_matrix = vectorizer.transform(batch["text"].fillna("").astype(str))
        distances, indices = neighbors.kneighbors(left_matrix, return_distance=True)
        rows = []
        for row_i, s1_id in enumerate(batch["entity_id"].astype(str).to_numpy()):
            for distance, right_i in zip(distances[row_i], indices[row_i]):
                similarity = 1.0 - float(distance)
                if similarity < min_similarity:
                    continue
                item = right.iloc[int(right_i)]
                rows.append(
                    {
                        "s1_id": s1_id,
                        "candidate_id": str(item["entity_id"]),
                        "candidate_source": str(item["candidate_source"]),
                        "blocking_method": method,
                        score_col: similarity,
                        null_col: None,
                    }
                )
        if rows:
            frames.append(pd.DataFrame(rows))

    if not frames:
        return pd.DataFrame(columns=["s1_id", "candidate_id", "candidate_source", "blocking_method", "name_tfidf_cosine", "address_tfidf_cosine"])
    return pd.concat(frames, ignore_index=True)


def write_country_chunks(
    con: duckdb.DuckDBPyConnection,
    cache_dir: Path,
    chunk_dir: Path,
    countries: list[str],
    args: argparse.Namespace,
) -> list[Path]:
    chunk_paths = []
    chunk_index = 1
    for country in countries:
        if not country:
            continue
        for field, method, top_k, min_similarity, max_features in [
            ("clean_name", "name_cosine", args.name_top_k, args.name_min_similarity, args.name_max_features),
            ("clean_address", "address_cosine", args.address_top_k, args.address_min_similarity, args.address_max_features),
        ]:
            left = load_country_frame(con, "s1", country, field)
            right = load_country_others(con, country, field)
            if len(left) == 0 or len(right) == 0:
                continue
            retrieved = retrieve_for_country(
                left=left,
                right=right,
                method=method,
                top_k=top_k,
                min_similarity=min_similarity,
                max_features=max_features,
                batch_size=args.batch_size,
            )
            if retrieved.empty:
                continue
            con.register("retrieved", retrieved)
            chunk_path = chunk_dir / f"cosine_{chunk_index:06d}.parquet"
            con.execute(
                f"""
                COPY (
                    SELECT
                        r.s1_id,
                        r.candidate_id,
                        any_value(r.candidate_source) AS candidate_source,
                        min(r.blocking_method) AS blocking_method,
                        max(r.name_tfidf_cosine) AS name_tfidf_cosine,
                        max(r.address_tfidf_cosine) AS address_tfidf_cosine
                    FROM retrieved r
                    LEFT JOIN existing_candidates e
                      ON r.s1_id = e.s1_id AND r.candidate_id = e.candidate_id
                    WHERE e.candidate_id IS NULL
                    GROUP BY r.s1_id, r.candidate_id
                ) TO '{_sql_path(chunk_path)}' (FORMAT PARQUET, COMPRESSION ZSTD, OVERWRITE_OR_IGNORE TRUE)
                """
            )
            added = con.execute(f"SELECT count(*) FROM read_parquet('{_sql_path(chunk_path)}')").fetchone()[0]
            con.unregister("retrieved")
            if added:
                chunk_paths.append(chunk_path)
                chunk_index += 1
                print(f"country {country}; {method}; added {added}", flush=True)
            else:
                chunk_path.unlink(missing_ok=True)
    return chunk_paths


def final_union(con: duckdb.DuckDBPyConnection, cache_dir: Path, chunk_paths: list[Path]) -> dict:
    existing_path = cache_dir / "train_candidates.parquet"
    boosted_path = cache_dir / "train_candidates_cosine.parquet"
    old_pairs = con.execute("SELECT count(DISTINCT s1_id || '|' || candidate_id) FROM existing_candidates").fetchone()[0]

    if chunk_paths:
        parquet_list = ", ".join(f"'{_sql_path(path)}'" for path in chunk_paths)
        con.execute(
            f"""
            COPY (
                SELECT
                    s1_id,
                    candidate_id,
                    any_value(candidate_source) AS candidate_source,
                    min(blocking_method) AS blocking_method,
                    max(name_tfidf_cosine) AS name_tfidf_cosine,
                    max(address_tfidf_cosine) AS address_tfidf_cosine
                FROM (
                    SELECT
                        s1_id,
                        candidate_id,
                        candidate_source,
                        blocking_method,
                        NULL::DOUBLE AS name_tfidf_cosine,
                        NULL::DOUBLE AS address_tfidf_cosine
                    FROM existing_candidates
                    UNION ALL
                    SELECT
                        s1_id,
                        candidate_id,
                        candidate_source,
                        blocking_method,
                        name_tfidf_cosine,
                        address_tfidf_cosine
                    FROM read_parquet([{parquet_list}])
                )
                GROUP BY s1_id, candidate_id
            ) TO '{_sql_path(boosted_path)}' (FORMAT PARQUET, COMPRESSION ZSTD, OVERWRITE_OR_IGNORE TRUE)
            """
        )
    else:
        con.execute(
            f"""
            COPY (
                SELECT
                    s1_id,
                    candidate_id,
                    candidate_source,
                    blocking_method,
                    NULL::DOUBLE AS name_tfidf_cosine,
                    NULL::DOUBLE AS address_tfidf_cosine
                FROM existing_candidates
            ) TO '{_sql_path(boosted_path)}' (FORMAT PARQUET, COMPRESSION ZSTD, OVERWRITE_OR_IGNORE TRUE)
            """
        )

    final_pairs, covered_s1 = con.execute(
        f"SELECT count(*), count(DISTINCT s1_id) FROM read_parquet('{_sql_path(boosted_path)}')"
    ).fetchone()
    return {
        "old_pair_count": int(old_pairs),
        "cosine_pairs_added": int(final_pairs - old_pairs),
        "final_pair_count": int(final_pairs),
        "avg_candidates_per_s1": float(final_pairs / covered_s1) if covered_s1 else 0.0,
        "boosted_path": str(boosted_path),
        "existing_path": str(existing_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--name-top-k", type=int, default=5)
    parser.add_argument("--address-top-k", type=int, default=3)
    parser.add_argument("--name-min-similarity", type=float, default=0.45)
    parser.add_argument("--address-min-similarity", type=float, default=0.50)
    parser.add_argument("--name-max-features", type=int, default=100000)
    parser.add_argument("--address-max-features", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()

    started = time.time()
    cache_dir = Path(args.cache_dir)
    chunk_dir = cache_dir / "cosine_candidate_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    con = connect(cache_dir)
    try:
        prepare_views(con, cache_dir)
        countries = [
            row[0]
            for row in con.execute(
                """
                SELECT country FROM s1 WHERE coalesce(country, '') <> ''
                INTERSECT
                SELECT country FROM others WHERE coalesce(country, '') <> ''
                ORDER BY country
                """
            ).fetchall()
        ]
        chunk_paths = write_country_chunks(con, cache_dir, chunk_dir, countries, args)
        metrics = final_union(con, cache_dir, chunk_paths)
    finally:
        con.close()

    Path(metrics["boosted_path"]).replace(Path(metrics["existing_path"]))
    print(f"old pair count {metrics['old_pair_count']}", flush=True)
    print(f"cosine pairs added {metrics['cosine_pairs_added']}", flush=True)
    print(f"final pair count {metrics['final_pair_count']}", flush=True)
    print(f"avg candidates/S1 {metrics['avg_candidates_per_s1']:.3f}", flush=True)
    print(f"runtime {round(time.time() - started, 1)}", flush=True)


if __name__ == "__main__":
    main()
