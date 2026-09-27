from __future__ import annotations

import argparse
import inspect
import json
import re
from pathlib import Path

import pandas as pd

from . import blocking
from .config import make_paths
from .data_loader import ENTITY_COLUMNS, read_tsv
from .evaluation import truth_map
from .pipeline import normalize_frame, save_json


def _pct(series: pd.Series) -> float:
    return float(series.mean() * 100) if len(series) else 0.0


def _source_for_id(entity_id: object) -> str:
    text = str(entity_id)
    if text.startswith("S2-"):
        return "S2"
    if text.startswith("S3-"):
        return "S3"
    return "UNKNOWN"


def _truth_pairs(gt: pd.DataFrame) -> pd.DataFrame:
    truth = truth_map(gt)
    rows = [
        {"s1_id": str(s1_id), "candidate_id": str(match), "candidate_source": _source_for_id(match)}
        for s1_id, matches in truth.items()
        for match in matches
    ]
    return pd.DataFrame(rows, columns=["s1_id", "candidate_id", "candidate_source"])


def _prepare_entities(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    out = normalize_frame(df)
    keep = ["entity_id", "country", "clean_name", "suffix_normalized_name", "clean_address"]
    out = out[keep].rename(columns={"entity_id": id_col})
    return out


def _join_truth_links(pairs: pd.DataFrame, s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame) -> pd.DataFrame:
    pairs = pairs.merge(s1, on="s1_id", how="left")
    right = pd.concat([s2, s3], ignore_index=True)
    pairs = pairs.merge(right, on="candidate_id", how="left", suffixes=("_s1", "_cand"))
    return pairs


def _add_match_flags(links: pd.DataFrame) -> pd.DataFrame:
    links["country_equal"] = links["country_s1"].astype(str) == links["country_cand"].astype(str)
    links["clean_name_equal"] = (links["clean_name_s1"].astype(str) != "") & (
        links["clean_name_s1"].astype(str) == links["clean_name_cand"].astype(str)
    )
    links["suffix_name_equal"] = (links["suffix_normalized_name_s1"].astype(str) != "") & (
        links["suffix_normalized_name_s1"].astype(str) == links["suffix_normalized_name_cand"].astype(str)
    )

    s1_name_tokens = links["clean_name_s1"].fillna("").astype(str).str.split().map(lambda xs: {x for x in xs if len(x) > 1})
    cand_name_tokens = links["clean_name_cand"].fillna("").astype(str).str.split().map(lambda xs: {x for x in xs if len(x) > 1})
    links["shared_name_token"] = [bool(a & b) for a, b in zip(s1_name_tokens, cand_name_tokens)]

    s1_addr_nums = links["clean_address_s1"].fillna("").astype(str).map(lambda x: set(re.findall(r"\d+", x)))
    cand_addr_nums = links["clean_address_cand"].fillna("").astype(str).map(lambda x: set(re.findall(r"\d+", x)))
    links["shared_address_number"] = [bool(a & b) for a, b in zip(s1_addr_nums, cand_addr_nums)]

    links["missing_name"] = (links["clean_name_s1"].fillna("").astype(str) == "") | (
        links["clean_name_cand"].fillna("").astype(str) == ""
    )
    links["missing_address"] = (links["clean_address_s1"].fillna("").astype(str) == "") | (
        links["clean_address_cand"].fillna("").astype(str) == ""
    )
    return links


def _examples(links: pd.DataFrame, limit: int) -> list[dict]:
    mask = (
        ~links["country_equal"]
        | ~links["clean_name_equal"]
        | ~links["suffix_name_equal"]
        | ~links["shared_name_token"]
        | ~links["shared_address_number"]
        | links["missing_name"]
        | links["missing_address"]
    )
    cols = [
        "s1_id",
        "candidate_id",
        "candidate_source",
        "country_s1",
        "country_cand",
        "clean_name_s1",
        "clean_name_cand",
        "suffix_normalized_name_s1",
        "suffix_normalized_name_cand",
        "clean_address_s1",
        "clean_address_cand",
        "country_equal",
        "clean_name_equal",
        "suffix_name_equal",
        "shared_name_token",
        "shared_address_number",
        "missing_name",
        "missing_address",
    ]
    return links.loc[mask, cols].head(limit).fillna("").to_dict("records")


def _static_blocking_audit() -> dict:
    exact_source = inspect.getsource(blocking.exact_block)
    rare_source = inspect.getsource(blocking.rare_token_block)
    address_source = inspect.getsource(blocking.address_block)
    tfidf_source = inspect.getsource(blocking.tfidf_block)

    tfidf_topk_pos = tfidf_source.find("argpartition")
    tfidf_country_filter_pos = tfidf_source.find("country !=")
    topk_before_country = (
        tfidf_topk_pos != -1 and tfidf_country_filter_pos != -1 and tfidf_topk_pos < tfidf_country_filter_pos
    )

    hard_filters = {
        "exact_name_block": "on=[\"country\", \"_key\"]" in exact_source,
        "suffix_name_block": "on=[\"country\", \"_key\"]" in exact_source,
        "rare_token_block": "idx.get((str(country), token)" in rare_source,
        "address_block": "idx.get((str(country), num)" in address_source,
        "name_tfidf_block": "country != str(s1_country)" in tfidf_source,
        "address_tfidf_block": "country != str(s1_country)" in tfidf_source,
    }

    reasons = []
    if any(hard_filters.values()):
        reasons.append("Country is a hard filter in blocking methods, so true links with country mismatches are excluded.")
    if topk_before_country:
        reasons.append("TF-IDF selects global Top-K before country filtering, so same-country true links can be excluded by higher-scoring out-of-country rows.")
    reasons.extend(
        [
            "Exact and suffix blocks require normalized name equality.",
            "Rare-token blocking ignores very common tokens and buckets above max_bucket.",
            "Address blocking requires shared numeric address tokens and ignores buckets above max_bucket.",
            "TF-IDF Top-K can miss true links when name/address text similarity is low or missing.",
        ]
    )

    return {
        "country_hard_filter_by_method": hard_filters,
        "tfidf_top_k_selected_before_country_filter": bool(topk_before_country),
        "obvious_reasons_true_links_can_be_excluded": reasons,
    }


def run(args: argparse.Namespace) -> dict:
    paths = make_paths(args.data_dir, args.artifacts_dir)
    s1 = read_tsv(paths.train_dir / "train_source1.tsv", ENTITY_COLUMNS)
    s2 = read_tsv(paths.train_dir / "train_source2.tsv", ENTITY_COLUMNS)
    s3 = read_tsv(paths.train_dir / "train_source3.tsv", ENTITY_COLUMNS)
    gt = read_tsv(paths.train_dir / "train_ground_truth.tsv")

    pairs = _truth_pairs(gt)
    s1 = _prepare_entities(s1, "s1_id")
    s2 = _prepare_entities(s2, "candidate_id")
    s3 = _prepare_entities(s3, "candidate_id")
    links = _add_match_flags(_join_truth_links(pairs, s1, s2, s3))

    report = {
        "true_links_total": int(len(links)),
        "source_distribution": {str(k): int(v) for k, v in links["candidate_source"].value_counts(dropna=False).items()},
        "metrics_percent": {
            "country_equal": _pct(links["country_equal"]),
            "clean_name_equal": _pct(links["clean_name_equal"]),
            "suffix_name_equal": _pct(links["suffix_name_equal"]),
            "shared_name_token": _pct(links["shared_name_token"]),
            "shared_address_number": _pct(links["shared_address_number"]),
            "missing_name": _pct(links["missing_name"]),
            "missing_address": _pct(links["missing_address"]),
        },
        "join_quality": {
            "missing_s1_records": int(links["country_s1"].isna().sum()),
            "missing_candidate_records": int(links["country_cand"].isna().sum()),
        },
        "mismatched_true_link_examples": _examples(links, args.examples),
        "static_blocking_audit": _static_blocking_audit(),
    }

    out_path = paths.metrics_dir / "quick_gt_diagnostics.json"
    save_json(out_path, report)
    print(json.dumps({"output": str(out_path), "true_links_total": report["true_links_total"]}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--examples", type=int, default=100)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
