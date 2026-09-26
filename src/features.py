from __future__ import annotations

from difflib import SequenceMatcher
import re

import numpy as np
import pandas as pd


def jaccard(a, b) -> float:
    sa, sb = set(a or ()), set(b or ())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 0.0


def overlap(a, b) -> float:
    sa, sb = set(a or ()), set(b or ())
    denom = min(len(sa), len(sb))
    return len(sa & sb) / denom if denom else 0.0


def ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a or "", b or "").ratio()


def token_sort_ratio(a, b) -> float:
    return ratio(" ".join(sorted(a or ())), " ".join(sorted(b or ())))


def build_features(candidates, s1, s2, s3, gt_pairs=None) -> tuple[pd.DataFrame, list[str]]:
    left = s1.add_prefix("s1_")
    left = left.rename(columns={"s1_entity_id": "s1_id"})
    right = pd.concat([s2, s3], ignore_index=True).add_prefix("cand_")
    right = right.rename(columns={"cand_entity_id": "candidate_id"})
    df = candidates.merge(left, on="s1_id", how="left").merge(right, on="candidate_id", how="left")
    rows = []
    for r in df.itertuples(index=False):
        name_tokens_l = tuple(t for t in str(r.s1_clean_name).split() if len(t) > 1)
        name_tokens_r = tuple(t for t in str(r.cand_clean_name).split() if len(t) > 1)
        addr_tokens_l = tuple(t for t in str(r.s1_clean_address).split() if len(t) > 1)
        addr_tokens_r = tuple(t for t in str(r.cand_clean_address).split() if len(t) > 1)
        num_l = tuple(re.findall(r"\d+", str(r.s1_business_address)))
        num_r = tuple(re.findall(r"\d+", str(r.cand_business_address)))
        postal_l = tuple(re.findall(r"\b[a-zA-Z]?\d[a-zA-Z0-9 -]{2,10}\d\b", str(r.s1_business_address)))
        postal_r = tuple(re.findall(r"\b[a-zA-Z]?\d[a-zA-Z0-9 -]{2,10}\d\b", str(r.cand_business_address)))
        row = {
            "name_exact_equal": float(r.s1_original_name == r.cand_original_name),
            "name_clean_equal": float(r.s1_clean_name == r.cand_clean_name and bool(r.s1_clean_name)),
            "name_suffix_equal": float(r.s1_suffix_normalized_name == r.cand_suffix_normalized_name and bool(r.s1_suffix_normalized_name)),
            "name_tfidf_similarity": float(r.name_tfidf_score),
            "name_token_jaccard": jaccard(name_tokens_l, name_tokens_r),
            "name_token_overlap": overlap(name_tokens_l, name_tokens_r),
            "name_ratio": ratio(r.s1_clean_name, r.cand_clean_name),
            "name_token_sort_ratio": token_sort_ratio(name_tokens_l, name_tokens_r),
            "name_len_diff": abs(len(r.s1_clean_name or "") - len(r.cand_clean_name or "")),
            "name_token_count_diff": abs(len(name_tokens_l or ()) - len(name_tokens_r or ())),
            "address_clean_equal": float(r.s1_clean_address == r.cand_clean_address and bool(r.s1_clean_address)),
            "address_tfidf_similarity": float(r.address_tfidf_score),
            "address_token_jaccard": jaccard(addr_tokens_l, addr_tokens_r),
            "address_token_overlap": overlap(addr_tokens_l, addr_tokens_r),
            "address_ratio": ratio(r.s1_clean_address, r.cand_clean_address),
            "numeric_token_overlap": overlap(num_l, num_r),
            "street_number_agree": float(bool(num_l and num_r and num_l[0] == num_r[0])),
            "postal_agree": float(bool(set(postal_l) & set(postal_r))),
            "address_len_diff": abs(len(r.s1_clean_address or "") - len(r.cand_clean_address or "")),
            "address_token_count_diff": abs(len(addr_tokens_l or ()) - len(addr_tokens_r or ())),
            "country_equal": float(r.s1_country == r.cand_country and bool(r.s1_country)),
            "country_missing": float(not r.s1_country or not r.cand_country),
            "candidate_is_s2": float(r.candidate_source == "S2"),
            "name_missing": float(not r.s1_clean_name or not r.cand_clean_name),
            "address_missing": float(not r.s1_clean_address or not r.cand_clean_address),
            "blocking_method_count": float(r.blocking_method_count),
            "name_tfidf_rank": float(r.name_tfidf_rank),
            "address_tfidf_rank": float(r.address_tfidf_rank),
        }
        for col in ["exact_name_block", "suffix_name_block", "rare_token_block", "address_block", "name_tfidf_block", "address_tfidf_block"]:
            row[col] = float(getattr(r, col))
        rows.append(row)
    features = pd.DataFrame(rows).replace([np.inf, -np.inf], 0).fillna(0)
    out = pd.concat([df[["s1_id", "candidate_id", "candidate_source"]].reset_index(drop=True), features], axis=1)
    if gt_pairs is not None:
        out["label"] = [int((a, b) in gt_pairs) for a, b in zip(out["s1_id"], out["candidate_id"])]
    return out, list(features.columns)
