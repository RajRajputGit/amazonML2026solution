from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import zipfile
from pathlib import Path


INVALID_STRINGS = {"", "null", "none", "nan", "<na>"}


def clean_ids(value: str) -> list[str]:
    seen = set()
    cleaned = []
    for raw in (value or "").split(","):
        item = raw.strip()
        if item.lower() in INVALID_STRINGS:
            continue
        if not item.startswith(("S2-", "S3-")):
            continue
        if item in seen:
            continue
        seen.add(item)
        cleaned.append(item)
    return cleaned


def read_tsv(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        if len(header) != 2:
            raise ValueError(f"{path}: expected exactly 2 columns, got {header}")
        rows: dict[str, list[str]] = {}
        for row in reader:
            if not row:
                continue
            s1_id = row[0].strip()
            if not s1_id:
                continue
            ids = clean_ids(row[1] if len(row) > 1 else "")
            if s1_id not in rows:
                rows[s1_id] = ids
    return header, rows


def write_tsv(path: Path, header: list[str], rows: dict[str, list[str]], order: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        for s1_id in order:
            writer.writerow([s1_id, ",".join(rows.get(s1_id, []))])


def merge_order(*maps: dict[str, list[str]]) -> list[str]:
    seen = set()
    order = []
    for mapping in maps:
        for s1_id in mapping:
            if s1_id not in seen:
                seen.add(s1_id)
                order.append(s1_id)
    return order


def union_preserve_order(left: list[str], right: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in left + right:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def run_validator(args: argparse.Namespace, check_ids: bool = False) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "utils/validate_submission.py",
        "--test-dir",
        args.test_dir,
        "--matching",
        args.matching,
        "--candidate",
        args.candidate,
    ]
    if check_ids:
        cmd.append("--check-ids")
    return subprocess.run(cmd, text=True, capture_output=True)


def print_result(label: str, result: subprocess.CompletedProcess[str]) -> None:
    print(label, flush=True)
    if result.stdout:
        print(result.stdout, flush=True)
    if result.stderr:
        print(result.stderr, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matching", default="output/matching_results.tsv")
    parser.add_argument("--candidate", default="output/candidate_pairs.tsv")
    parser.add_argument("--test-dir", default="dataset/test")
    parser.add_argument("--zip-path", default="output/submission.zip")
    args = parser.parse_args()

    matching_path = Path(args.matching)
    candidate_path = Path(args.candidate)
    match_header, matches = read_tsv(matching_path)
    candidate_header, candidates = read_tsv(candidate_path)
    order = merge_order(matches, candidates)

    fixed_matches = {s1_id: clean_ids(",".join(matches.get(s1_id, []))) for s1_id in order}
    fixed_candidates = {
        s1_id: union_preserve_order(clean_ids(",".join(candidates.get(s1_id, []))), fixed_matches.get(s1_id, []))
        for s1_id in order
    }

    write_tsv(matching_path, match_header, fixed_matches, order)
    write_tsv(candidate_path, candidate_header, fixed_candidates, order)

    result = run_validator(args)
    print_result("validator", result)
    if result.returncode != 0:
        return result.returncode

    check_result = run_validator(args, check_ids=True)
    print_result("validator --check-ids", check_result)

    zip_path = Path(args.zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(matching_path, "output/matching_results.tsv")
        zf.write(candidate_path, "output/candidate_pairs.tsv")
    print(f"final ZIP path {zip_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
