from __future__ import annotations

import argparse
import gc
import inspect
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .blocking import address_block, exact_block, rare_token_block, tfidf_block
from .config import RANDOM_SEED, make_paths
from .data_loader import load_training
from .evaluation import truth_map
from .pipeline import normalize_frame, save_json


BLOCK_METHODS = [
    "exact_name_block",
    "suffix_name_block",
    "rare_token_block",
    "address_block",
    "name_tfidf_block",
    "address_tfidf_block",
]


def _method_report(cands: dict[tuple[str, str], dict], true_pairs: set[tuple[str, str]]) -> tuple[dict, set[tuple[str, str]]]:
    recovered = {
        (str(s1_id), str(cid))
        for s1_id, cid in cands.keys()
        if (str(s1_id), str(cid)) in true_pairs
    }
    return {
        "true_links_recovered": int(len(recovered)),
        "true_links_total": int(len(true_pairs)),
        "candidate_recall": float(len(recovered) / len(true_pairs)) if true_pairs else 0.0,
        "total_candidate_pairs": int(len(cands)),
    }, recovered


def _gt_parse_report(gt: pd.DataFrame, s1_ids: set[str], other_ids: set[str]) -> dict:
    required = {"source1_entity_id", "matched_entity_ids"}
    missing_columns = sorted(required - set(gt.columns))
    truth = truth_map(gt)
    parsed_pairs = {(str(s1_id), str(match)) for s1_id, matches in truth.items() for match in matches}
    bad_s1 = sorted(str(s1_id) for s1_id in truth if str(s1_id) not in s1_ids)
    bad_match_ids = sorted({match for _, match in parsed_pairs if match not in other_ids})
    raw_match_values = gt["matched_entity_ids"].astype(str) if "matched_entity_ids" in gt.columns else pd.Series(dtype=str)
    return {
        "missing_required_columns": missing_columns,
        "ground_truth_rows": int(len(gt)),
        "truth_map_entities": int(len(truth)),
        "parsed_true_links": int(len(parsed_pairs)),
        "blank_match_rows": int((raw_match_values.str.strip() == "").sum()) if len(raw_match_values) else 0,
        "comma_separated_rows": int(raw_match_values.str.contains(",", regex=False).sum()) if len(raw_match_values) else 0,
        "source1_ids_not_found_count": int(len(bad_s1)),
        "source1_ids_not_found_examples": bad_s1[:20],
        "matched_ids_not_found_count": int(len(bad_match_ids)),
        "matched_ids_not_found_examples": bad_match_ids[:20],
    }


def _pairwise_tfidf_scores(left: list[str], right: list[str]) -> list[float]:
    scores = []
    for a, b in zip(left, right):
        if not a or not b:
            scores.append(0.0)
            continue
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, dtype=np.float32)
        matrix = vectorizer.fit_transform([a, b])
        scores.append(float((matrix[0] @ matrix[1].T)[0, 0]))
    return scores


def _distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    arr = np.array(values, dtype=np.float32)
    return {
        "count": int(len(values)),
        "min": float(np.min(arr)),
        "p25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.5)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.9)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _missed_link_causes(
    missed_pairs: set[tuple[str, str]],
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    sample_size: int,
) -> dict:
    rng = random.Random(RANDOM_SEED)
    sample = sorted(missed_pairs)
    if len(sample) > sample_size:
        sample = rng.sample(sample, sample_size)

    s1_lookup = s1.set_index("entity_id").to_dict("index")
    other_lookup = pd.concat([s2, s3], ignore_index=True).set_index("entity_id").to_dict("index")
    rows = []
    name_left, name_right, addr_left, addr_right = [], [], [], []
    counters = Counter()

    for s1_id, cid in sample:
        left = s1_lookup.get(s1_id)
        right = other_lookup.get(cid)
        if left is None or right is None:
            counters["missing_lookup"] += 1
            continue

        left_name = str(left.get("clean_name", ""))
        right_name = str(right.get("clean_name", ""))
        left_suffix = str(left.get("suffix_normalized_name", ""))
        right_suffix = str(right.get("suffix_normalized_name", ""))
        left_addr = str(left.get("clean_address", ""))
        right_addr = str(right.get("clean_address", ""))
        left_name_tokens = {t for t in left_name.split() if len(t) > 1}
        right_name_tokens = {t for t in right_name.split() if len(t) > 1}
        left_addr_nums = set(re.findall(r"\d+", left_addr))
        right_addr_nums = set(re.findall(r"\d+", right_addr))

        row = {
            "same_country": str(left.get("country", "")) == str(right.get("country", "")),
            "clean_name_equal": bool(left_name and left_name == right_name),
            "suffix_name_equal": bool(left_suffix and left_suffix == right_suffix),
            "shared_name_tokens": len(left_name_tokens & right_name_tokens),
            "shared_address_numbers": len(left_addr_nums & right_addr_nums),
            "missing_name": not left_name or not right_name,
            "missing_address": not left_addr or not right_addr,
        }
        rows.append(row)
        name_left.append(left_name)
        name_right.append(right_name)
        addr_left.append(left_addr)
        addr_right.append(right_addr)

    name_scores = _pairwise_tfidf_scores(name_left, name_right)
    address_scores = _pairwise_tfidf_scores(addr_left, addr_right)
    for row, name_score, address_score in zip(rows, name_scores, address_scores):
        row["name_tfidf_similarity"] = name_score
        row["address_tfidf_similarity"] = address_score

    df = pd.DataFrame(rows)
    if df.empty:
        return {"sampled_missed_links": 0, "lookup_failures": int(counters["missing_lookup"])}

    return {
        "sampled_missed_links": int(len(df)),
        "lookup_failures": int(counters["missing_lookup"]),
        "same_country": {str(k): int(v) for k, v in df["same_country"].value_counts(dropna=False).items()},
        "clean_name_exact_equality": {str(k): int(v) for k, v in df["clean_name_equal"].value_counts(dropna=False).items()},
        "suffix_name_equality": {str(k): int(v) for k, v in df["suffix_name_equal"].value_counts(dropna=False).items()},
        "name_tfidf_similarity_distribution": _distribution(df["name_tfidf_similarity"].tolist()),
        "address_tfidf_similarity_distribution": _distribution(df["address_tfidf_similarity"].tolist()),
        "shared_name_tokens_distribution": _distribution(df["shared_name_tokens"].astype(float).tolist()),
        "shared_address_numbers_distribution": _distribution(df["shared_address_numbers"].astype(float).tolist()),
        "missing_name": {str(k): int(v) for k, v in df["missing_name"].value_counts(dropna=False).items()},
        "missing_address": {str(k): int(v) for k, v in df["missing_address"].value_counts(dropna=False).items()},
    }


def _tfidf_audit() -> dict:
    source = inspect.getsource(tfidf_block)
    argpartition_pos = source.find("argpartition")
    country_filter_pos = source.find("country !=")
    topk_before_country = argpartition_pos != -1 and country_filter_pos != -1 and argpartition_pos < country_filter_pos
    return {
        "top_k_selected_before_country_filter": bool(topk_before_country),
        "risk": (
            "BUG/RISK: tfidf_block selects global top-k matches before filtering by country, "
            "so true same-country links can be dropped if out-of-country rows occupy the top-k."
            if topk_before_country
            else "No top-k-before-country-filter pattern detected by source audit."
        ),
    }


def run(args: argparse.Namespace) -> dict:
    paths = make_paths(args.data_dir, args.artifacts_dir)
    out_path = paths.metrics_dir / "blocking_diagnostics.json"
    started = time.time()
    print("Loading training data...", flush=True)
    s1, s2, s3, gt = load_training(paths)
    s1_ids = set(s1["entity_id"].astype(str))
    other_ids = set(s2["entity_id"].astype(str)) | set(s3["entity_id"].astype(str))
    gt_report = _gt_parse_report(gt, s1_ids, other_ids)
    truth = truth_map(gt)
    true_pairs = {(str(s1_id), str(match)) for s1_id, matches in truth.items() for match in matches}
    del gt, truth
    gc.collect()

    print("Normalizing frames...", flush=True)
    s1 = normalize_frame(s1)
    s2 = normalize_frame(s2)
    s3 = normalize_frame(s3)
    others = {"S2": s2, "S3": s3}

    method_results = {}
    union_recovered_pairs: set[tuple[str, str]] = set()

    def save_partial(status: str) -> None:
        partial_missed = int(len(true_pairs) - len(union_recovered_pairs))
        partial_report = {
            "status": status,
            "runtime_sec": round(time.time() - started, 3),
            "ground_truth_parsing": gt_report,
            "per_method_true_link_recall": method_results,
            "union": {
                "true_links_recovered": int(len(union_recovered_pairs)),
                "true_links_total": int(len(true_pairs)),
                "candidate_recall": float(len(union_recovered_pairs) / len(true_pairs)) if true_pairs else 0.0,
                "missed_true_links": partial_missed,
                "note": "Partial union tracks recovered true links only; candidate-pair union is intentionally not materialized.",
            },
            "tfidf_block_audit": _tfidf_audit(),
        }
        save_json(out_path, partial_report)
        print(f"Saved partial diagnostics to {out_path}", flush=True)

    method_specs = [
        ("exact_name_block", lambda c: exact_block(s1, others, "clean_name", "clean_name", "exact_name_block", c)),
        ("suffix_name_block", lambda c: exact_block(s1, others, "suffix_normalized_name", "suffix_normalized_name", "suffix_name_block", c)),
        ("rare_token_block", lambda c: rare_token_block(s1, others, c)),
        ("address_block", lambda c: address_block(s1, others, c)),
        ("name_tfidf_block", lambda c: tfidf_block(s1, others, c, "clean_name", "name_tfidf_block", top_k=args.name_top_k)),
        ("address_tfidf_block", lambda c: tfidf_block(s1, others, c, "clean_address", "address_tfidf_block", top_k=args.address_top_k)),
    ]

    for method, runner in method_specs:
        print(f"Running diagnostic method: {method}", flush=True)
        t0 = time.time()
        cands: dict[tuple[str, str], dict] = {}
        runner(cands)
        print(f"Computing true-link recall for {method}...", flush=True)
        method_results[method], recovered = _method_report(cands, true_pairs)
        method_results[method]["runtime_sec"] = round(time.time() - t0, 3)
        union_recovered_pairs.update(recovered)
        save_partial(f"completed_{method}")
        print(f"Completed diagnostic method: {method}", flush=True)
        del recovered
        del cands
        gc.collect()

    missed_pairs = true_pairs - union_recovered_pairs
    print(f"Sampling missed true links: {min(args.missed_sample_size, len(missed_pairs))}", flush=True)
    report = {
        "status": "complete",
        "runtime_sec": round(time.time() - started, 3),
        "ground_truth_parsing": gt_report,
        "per_method_true_link_recall": method_results,
        "union": {
            "true_links_recovered": int(len(union_recovered_pairs)),
            "true_links_total": int(len(true_pairs)),
            "candidate_recall": float(len(union_recovered_pairs) / len(true_pairs)) if true_pairs else 0.0,
            "missed_true_links": int(len(missed_pairs)),
            "note": "Union tracks recovered true links only; candidate-pair union is intentionally not materialized.",
        },
        "missed_true_link_cause_statistics": _missed_link_causes(missed_pairs, s1, s2, s3, args.missed_sample_size),
        "tfidf_block_audit": _tfidf_audit(),
    }
    save_json(out_path, report)
    print(f"Saved {out_path}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--name-top-k", type=int, default=12)
    parser.add_argument("--address-top-k", type=int, default=8)
    parser.add_argument("--missed-sample-size", type=int, default=1000)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"output": str(Path(args.artifacts_dir) / "metrics" / "blocking_diagnostics.json"), "union": report["union"]}, indent=2))


if __name__ == "__main__":
    main()
