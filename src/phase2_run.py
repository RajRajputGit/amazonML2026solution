import pandas as pd
import numpy as np
import pyarrow.csv as parse_csv
import pyarrow as pa
import os
import json
import time
from collections import defaultdict
from blocking import MultiStrategyBlocker, load_tsv, normalize_name

DATA_DIR = "/Users/apple/Desktop/amazonMLsolution/student_resource/dataset/train"
OUTPUT_DIR = "/Users/apple/Desktop/amazonMLsolution/output"
SCRATCH_DIR = "/Users/apple/Desktop/amazonMLsolution/scratch"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SCRATCH_DIR, exist_ok=True)

start_time = time.time()
print("=== Running Phase 2 Candidate Generation Pipeline & Validation ===")

print("[Step 1] Loading Training Data...")
df_s1 = load_tsv(os.path.join(DATA_DIR, "train_source1.tsv"))
df_s2 = load_tsv(os.path.join(DATA_DIR, "train_source2.tsv"))
df_s3 = load_tsv(os.path.join(DATA_DIR, "train_source3.tsv"))
df_gt = load_tsv(os.path.join(DATA_DIR, "train_ground_truth.tsv"))

df_gt['matched_entity_ids'] = df_gt['matched_entity_ids'].replace({'nan': '', 'None': '', '<NA>': ''}).fillna('')
gt_pairs_dict = {}
total_gt_pairs = 0

for s1_id, m_str in zip(df_gt['source1_entity_id'], df_gt['matched_entity_ids']):
    if m_str:
        m_set = set(x for x in m_str.split(',') if x)
        gt_pairs_dict[s1_id] = m_set
        total_gt_pairs += len(m_set)

total_s1 = len(df_s1)
max_possible_pairs = float(total_s1 * (len(df_s2) + len(df_s3)))

# ---------------------------------------------------------
# Run MultiStrategyBlocker
# ---------------------------------------------------------
print("\n[Step 2] Running MultiStrategyBlocker...")
blocker = MultiStrategyBlocker()
cand_dict = blocker.generate_candidates(df_s1, df_s2, df_s3)

# ---------------------------------------------------------
# Compute Metrics & Recovered Examples
# ---------------------------------------------------------
print("\n[Step 3] Measuring Metrics & Recovered Examples...")

# Build fast exact-name lookup to identify matches baseline missed
exact_idx_s2 = defaultdict(list)
exact_idx_s3 = defaultdict(list)
for eid, c, name in zip(df_s2['entity_id'], df_s2['country'], df_s2['business_name']):
    n = normalize_name(name)
    if n: exact_idx_s2[(c, n)].append(eid)
for eid, c, name in zip(df_s3['entity_id'], df_s3['country'], df_s3['business_name']):
    n = normalize_name(name)
    if n: exact_idx_s3[(c, n)].append(eid)

retrieved_gt = 0
total_candidates = 0
recovered_examples = []

s1_lookup = df_s1.set_index('entity_id').to_dict(orient='index')
s2_lookup = df_s2.set_index('entity_id').to_dict(orient='index')
s3_lookup = df_s3.set_index('entity_id').to_dict(orient='index')

def get_record(eid):
    if eid.startswith('S1-'): return s1_lookup.get(eid)
    if eid.startswith('S2-'): return s2_lookup.get(eid)
    if eid.startswith('S3-'): return s3_lookup.get(eid)
    return None

for s1_id, cands in cand_dict.items():
    num_c = len(cands)
    total_candidates += num_c
    
    gt_set = gt_pairs_dict.get(s1_id)
    if gt_set:
        cand_set = set(cands)
        hits = gt_set.intersection(cand_set)
        retrieved_gt += len(hits)
        
        # Check baseline exact hits
        s1_rec = s1_lookup[s1_id]
        c = s1_rec['country']
        n = normalize_name(s1_rec['business_name'])
        b0_cands = set(exact_idx_s2.get((c, n), [])) | set(exact_idx_s3.get((c, n), []))
        b0_hits = gt_set.intersection(b0_cands)
        
        recovered_hits = hits - b0_hits
        if recovered_hits and len(recovered_examples) < 15:
            rec_recs = []
            for m_id in list(recovered_hits)[:3]:
                m_r = get_record(m_id)
                if m_r:
                    rec_recs.append({"entity_id": m_id, **m_r})
            recovered_examples.append({
                "s1_id": s1_id,
                "s1_name": s1_rec['business_name'],
                "s1_address": s1_rec['business_address'],
                "country": c,
                "recovered_matches": rec_recs
            })

candidate_recall = float(retrieved_gt / total_gt_pairs)
avg_candidates_per_s1 = float(total_candidates / total_s1)
candidate_reduction_ratio = float(1.0 - (total_candidates / max_possible_pairs))

print(f"\n--- PHASE 2 FINAL BLOCKING METRICS ---")
print(f"Candidate Recall: {candidate_recall * 100:.2f}% ({retrieved_gt:,} / {total_gt_pairs:,} GT pairs)")
print(f"Average Candidates per S1: {avg_candidates_per_s1:.2f}")
print(f"Total Candidates Generated: {total_candidates:,}")
print(f"Candidate Reduction Ratio: {candidate_reduction_ratio * 100:.8f}%")

# Save candidate_pairs.tsv in output directory
print("\n[Step 4] Writing candidate_pairs.tsv...")
tsv_lines = ["source1_entity_id\tcandidate_entity_ids"]
for s1_id in df_s1['entity_id']:
    cands = cand_dict.get(s1_id, [])
    c_str = ",".join(cands)
    tsv_lines.append(f"{s1_id}\t{c_str}")

with open(os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"), "w") as f:
    f.write("\n".join(tsv_lines) + "\n")

print(f"Saved candidate_pairs.tsv ({len(tsv_lines)-1:,} rows) in output/ folder.")

summary_json = {
    "total_s1_entities": total_s1,
    "total_gt_pairs": total_gt_pairs,
    "retrieved_gt_pairs": retrieved_gt,
    "candidate_recall_pct": round(candidate_recall * 100, 2),
    "avg_candidates_per_s1": round(avg_candidates_per_s1, 2),
    "total_candidates": total_candidates,
    "reduction_ratio_pct": round(candidate_reduction_ratio * 100, 8),
    "recovered_examples": recovered_examples
}

with open(os.path.join(SCRATCH_DIR, "phase2_final_summary.json"), "w") as f:
    json.dump(summary_json, f, indent=2)

print(f"\nPhase 2 completed successfully in {time.time() - start_time:.2f}s!")
