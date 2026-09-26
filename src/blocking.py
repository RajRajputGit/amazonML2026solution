from __future__ import annotations

from collections import Counter, defaultdict
import re
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def _append(cands: dict[tuple[str, str], dict], s1_id: str, cid: str, source: str, method: str, score=None, rank=None):
    key = (s1_id, cid)
    row = cands.setdefault(
        key,
        {
            "s1_id": s1_id,
            "candidate_id": cid,
            "candidate_source": source,
            "exact_name_block": 0,
            "suffix_name_block": 0,
            "rare_token_block": 0,
            "address_block": 0,
            "name_tfidf_block": 0,
            "address_tfidf_block": 0,
            "name_tfidf_score": 0.0,
            "name_tfidf_rank": 9999,
            "address_tfidf_score": 0.0,
            "address_tfidf_rank": 9999,
        },
    )
    row[method] = 1
    if score is not None:
        score_col = "name_tfidf_score" if method == "name_tfidf_block" else "address_tfidf_score"
        rank_col = "name_tfidf_rank" if method == "name_tfidf_block" else "address_tfidf_rank"
        row[score_col] = max(row[score_col], float(score))
        row[rank_col] = min(row[rank_col], int(rank))


def exact_block(s1, others, left_col, right_col, method, cands):
    left = s1[["entity_id", "country", left_col]].rename(columns={"entity_id": "s1_id", left_col: "_key"})
    left = left[left["_key"].astype(str) != ""]
    for source, df in others.items():
        right = df[["entity_id", "country", right_col]].rename(columns={"entity_id": "candidate_id", right_col: "_key"})
        right = right[right["_key"].astype(str) != ""]
        merged = left.merge(right, on=["country", "_key"], how="inner")
        for s1_id, cid in merged[["s1_id", "candidate_id"]].itertuples(index=False):
            _append(cands, str(s1_id), str(cid), source, method)


def rare_token_block(s1, others, cands, max_df=500, max_tokens_per_s1=6, max_bucket=250):
    token_counts = Counter()
    for df in others.values():
        for text in df["clean_name"]:
            token_counts.update(set(t for t in str(text).split() if len(t) > 1))
    indexes = {}
    for source, df in others.items():
        idx = defaultdict(list)
        for eid, country, text in df[["entity_id", "country", "clean_name"]].itertuples(index=False):
            toks = tuple(t for t in str(text).split() if len(t) > 1)
            for token in set(toks):
                if 1 < token_counts[token] <= max_df:
                    idx[(str(country), token)].append(str(eid))
        indexes[source] = idx
    for eid, country, text in s1[["entity_id", "country", "clean_name"]].itertuples(index=False):
        toks = tuple(t for t in str(text).split() if len(t) > 1)
        tokens = sorted(set(toks), key=lambda t: token_counts[t])[:max_tokens_per_s1]
        for token in tokens:
            for source, idx in indexes.items():
                bucket = idx.get((str(country), token), [])
                if len(bucket) <= max_bucket:
                    for cid in bucket:
                        _append(cands, str(eid), cid, source, "rare_token_block")


def address_block(s1, others, cands, max_bucket=200):
    indexes = {}
    for source, df in others.items():
        idx = defaultdict(list)
        for eid, country, addr in df[["entity_id", "country", "business_address"]].itertuples(index=False):
            nums = tuple(re.findall(r"\d+", str(addr)))[:3]
            for num in set(nums[:3]):
                if len(num) >= 2:
                    idx[(str(country), num)].append(str(eid))
        indexes[source] = idx
    for eid, country, addr in s1[["entity_id", "country", "business_address"]].itertuples(index=False):
        nums = tuple(re.findall(r"\d+", str(addr)))[:3]
        for num in set(nums[:3]):
            if len(num) < 2:
                continue
            for source, idx in indexes.items():
                bucket = idx.get((str(country), num), [])
                if len(bucket) <= max_bucket:
                    for cid in bucket:
                        _append(cands, str(eid), cid, source, "address_block")


def tfidf_block(s1, others, cands, field, method, top_k=20, batch_size=5000):
    corpus = []
    meta = []
    for source, df in others.items():
        for eid, country, text in df[["entity_id", "country", field]].itertuples(index=False):
            corpus.append(text or "")
            meta.append((str(eid), str(country), source))
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=250000, dtype=np.float32)
    right = vectorizer.fit_transform(corpus)
    left = vectorizer.transform((s1[field].fillna("").astype(str)).tolist())
    s1_meta = list(s1[["entity_id", "country"]].itertuples(index=False, name=None))
    for start in range(0, left.shape[0], batch_size):
        scores = left[start : start + batch_size] @ right.T
        for local_i in range(scores.shape[0]):
            row = scores.getrow(local_i)
            if row.nnz == 0:
                continue
            if row.nnz > top_k:
                pick = np.argpartition(row.data, -top_k)[-top_k:]
            else:
                pick = np.arange(row.nnz)
            order = pick[np.argsort(row.data[pick])[::-1]]
            s1_id, s1_country = s1_meta[start + local_i]
            rank = 0
            for pos in order:
                cid, country, source = meta[row.indices[pos]]
                if country != str(s1_country):
                    continue
                rank += 1
                _append(cands, str(s1_id), cid, source, method, float(row.data[pos]), rank)


def generate_candidates(s1, s2, s3, config) -> tuple[pd.DataFrame, dict]:
    t0 = time.time()
    cands = {}
    others = {"S2": s2, "S3": s3}
    exact_block(s1, others, "clean_name", "clean_name", "exact_name_block", cands)
    exact_block(s1, others, "suffix_normalized_name", "suffix_normalized_name", "suffix_name_block", cands)
    rare_token_block(s1, others, cands)
    address_block(s1, others, cands)
    tfidf_block(s1, others, cands, "clean_name", "name_tfidf_block", top_k=config["name_top_k"])
    tfidf_block(s1, others, cands, "clean_address", "address_tfidf_block", top_k=config["address_top_k"])
    out = pd.DataFrame(cands.values())
    if len(out):
        block_cols = [c for c in out.columns if c.endswith("_block")]
        out["blocking_method_count"] = out[block_cols].sum(axis=1)
    stats = {"candidate_generation_runtime_sec": round(time.time() - t0, 3)}
    return out, stats


def candidate_stats(candidates: pd.DataFrame, total_s1: int, total_s2s3: int, true_pairs: set[tuple[str, str]]) -> dict:
    found = set(zip(candidates["s1_id"].astype(str), candidates["candidate_id"].astype(str))) if len(candidates) else set()
    per_s1 = candidates.groupby("s1_id").size() if len(candidates) else pd.Series(dtype=int)
    recovered = len(true_pairs & found)
    return {
        "true_links_recovered": int(recovered),
        "candidate_recall": recovered / len(true_pairs) if true_pairs else 0.0,
        "total_candidate_pairs": int(len(candidates)),
        "mean_candidates_per_s1": float(per_s1.mean()) if len(per_s1) else 0.0,
        "median_candidates_per_s1": float(per_s1.median()) if len(per_s1) else 0.0,
        "p95_candidates_per_s1": float(per_s1.quantile(0.95)) if len(per_s1) else 0.0,
        "max_candidates_per_s1": int(per_s1.max()) if len(per_s1) else 0,
        "candidate_reduction_ratio": 1.0 - (len(candidates) / (total_s1 * total_s2s3)),
    }
