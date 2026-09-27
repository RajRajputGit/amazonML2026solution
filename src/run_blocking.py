from __future__ import annotations

import argparse
import gc
import json
import time

import pandas as pd

from .blocking import candidate_stats, generate_candidates
from .config import make_paths
from .data_loader import load_training
from .evaluation import truth_map
from .pipeline import normalize_frame, save_json


def exact_baseline_fast(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame) -> pd.DataFrame:
    left = s1[["entity_id", "country", "clean_name"]].rename(columns={"entity_id": "s1_id"})
    left = left[left["clean_name"].astype(str) != ""]
    frames = []
    for source, right_df in [("S2", s2), ("S3", s3)]:
        right = right_df[["entity_id", "country", "clean_name"]].rename(columns={"entity_id": "candidate_id"})
        right = right[right["clean_name"].astype(str) != ""]
        merged = left.merge(right, on=["country", "clean_name"], how="inner")
        merged["candidate_source"] = source
        frames.append(merged[["s1_id", "candidate_id", "candidate_source"]])
    if not frames:
        return pd.DataFrame(columns=["s1_id", "candidate_id", "candidate_source"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(["s1_id", "candidate_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--mode", choices=["exact", "multipass"], required=True)
    parser.add_argument("--name-top-k", type=int, default=12)
    parser.add_argument("--address-top-k", type=int, default=8)
    args = parser.parse_args()

    paths = make_paths(args.data_dir, args.artifacts_dir)
    print(f"Loading training data for {args.mode} blocking...", flush=True)
    s1, s2, s3, gt = load_training(paths)
    truth = truth_map(gt)
    true_pairs = {(s1_id, match) for s1_id, matches in truth.items() for match in matches}
    del gt, truth
    gc.collect()

    print("Normalizing input frames...", flush=True)
    s1 = normalize_frame(s1)
    s2 = normalize_frame(s2)
    s3 = normalize_frame(s3)

    if args.mode == "exact":
        print("Running exact-name baseline only...", flush=True)
        t0 = time.time()
        exact = exact_baseline_fast(s1, s2, s3)
        exact_stats = candidate_stats(exact, len(s1), len(s2) + len(s3), true_pairs)
        exact_stats["runtime_sec"] = round(time.time() - t0, 3)
        save_json(paths.metrics_dir / "exact_baseline.json", exact_stats)
        print(f"Saved {paths.metrics_dir / 'exact_baseline.json'}", flush=True)
        print(json.dumps(exact_stats, indent=2))
        return

    total_s1 = len(s1)
    total_s2s3 = len(s2) + len(s3)

    def checkpoint(stage: str, cands: dict) -> None:
        stage_candidates = pd.DataFrame(cands.values())
        stats = candidate_stats(stage_candidates, total_s1, total_s2s3, true_pairs)
        stats["stage"] = stage
        stats["elapsed_sec"] = round(time.time() - t0, 3)
        out_path = paths.metrics_dir / f"blocking_{stage}_checkpoint.json"
        save_json(out_path, stats)
        print(f"Saved {out_path}", flush=True)
        del stage_candidates
        gc.collect()

    print("Running multipass blocking with stage checkpoints...", flush=True)
    t0 = time.time()
    candidates, runtime_stats = generate_candidates(
        s1,
        s2,
        s3,
        {"name_top_k": args.name_top_k, "address_top_k": args.address_top_k},
        checkpoint=checkpoint,
    )
    final_stats = candidate_stats(candidates, len(s1), len(s2) + len(s3), true_pairs)
    final_stats["runtime_sec"] = round(time.time() - t0, 3)
    final_stats.update(runtime_stats)
    save_json(paths.metrics_dir / "blocking_results_v2.json", {"final": final_stats})
    candidates.to_pickle(paths.metrics_dir / "blocking_multipass_candidates.pkl")
    print(f"Saved {paths.metrics_dir / 'blocking_results_v2.json'}", flush=True)
    print(f"Saved {paths.metrics_dir / 'blocking_multipass_candidates.pkl'}", flush=True)

    print(json.dumps({"final": final_stats}, indent=2))


if __name__ == "__main__":
    main()
