import pandas as pd
import numpy as np
import pyarrow.csv as parse_csv
import pyarrow as pa
import re
import unicodedata
import os
import json
import time
from collections import Counter

DATA_DIR = "/Users/apple/Desktop/amazonMLsolution/student_resource/dataset/train"
OUTPUT_DIR = "/Users/apple/Desktop/amazonMLsolution/scratch"
os.makedirs(OUTPUT_DIR, exist_ok=True)

start_time = time.time()
print("=== Phase 1: Business Entity Resolution Data Profiling & Baseline Blocking ===")

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
print("\n[Step 1] Loading Datasets using PyArrow...")

def load_tsv(filepath):
    read_options = parse_csv.ReadOptions(block_size=1048576*16)
    parse_options = parse_csv.ParseOptions(delimiter='\t', quote_char=False)
    table = parse_csv.read_csv(filepath, read_options=read_options, parse_options=parse_options)
    return table.to_pandas().astype(str)

t0 = time.time()
df_s1 = load_tsv(os.path.join(DATA_DIR, "train_source1.tsv"))
df_s2 = load_tsv(os.path.join(DATA_DIR, "train_source2.tsv"))
df_s3 = load_tsv(os.path.join(DATA_DIR, "train_source3.tsv"))
df_gt = load_tsv(os.path.join(DATA_DIR, "train_ground_truth.tsv"))

print(f"Data loading completed in {time.time() - t0:.2f}s")
print(f"  df_s1 shape: {df_s1.shape}")
print(f"  df_s2 shape: {df_s2.shape}")
print(f"  df_s3 shape: {df_s3.shape}")
print(f"  df_gt shape: {df_gt.shape}")

for df in [df_s1, df_s2, df_s3]:
    for col in ['business_name', 'business_address', 'country']:
        df[col] = df[col].replace({'nan': None, 'None': None, '<NA>': None, '': None})

df_gt['matched_entity_ids'] = df_gt['matched_entity_ids'].replace({'nan': '', 'None': '', '<NA>': ''}).fillna('')

# ---------------------------------------------------------
# 2. Data Profiling (Fast)
# ---------------------------------------------------------
print("\n[Step 2] Data Profiling (Optimized)...")

def profile_dataset(df, name):
    t_prof = time.time()
    total = int(len(df))
    missing = {k: int(v) for k, v in df.isnull().sum().to_dict().items()}
    
    unique_ids = int(df['entity_id'].nunique())
    dup_ids = total - unique_ids
    
    name_non_null = df['business_name'].dropna()
    unique_names = int(name_non_null.nunique())
    dup_names = int(len(name_non_null)) - unique_names
    
    addr_non_null = df['business_address'].dropna()
    unique_addrs = int(addr_non_null.nunique())
    dup_addrs = int(len(addr_non_null)) - unique_addrs
    
    country_counts = df['country'].value_counts(dropna=False).to_dict()
    country_dist = {str(k): int(v) for k, v in country_counts.items()}
    
    name_str = df['business_name'].fillna('')
    addr_str = df['business_address'].fillna('')
    
    name_len = name_str.str.len().to_numpy()
    name_words = name_str.str.split().str.len().fillna(0).to_numpy()
    
    addr_len = addr_str.str.len().to_numpy()
    addr_words = addr_str.str.split().str.len().fillna(0).to_numpy()
    
    def calc_stats(arr):
        return {
            "mean": round(float(arr.mean()), 2),
            "std": round(float(arr.std()), 2),
            "min": int(arr.min()),
            "median": round(float(np.median(arr)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "p99": round(float(np.percentile(arr, 99)), 2),
            "max": int(arr.max())
        }
        
    res = {
        "name": name,
        "rows": total,
        "missing_values": missing,
        "unique_entity_ids": unique_ids,
        "duplicate_entity_ids": dup_ids,
        "unique_names": unique_names,
        "duplicate_names": dup_names,
        "unique_addresses": unique_addrs,
        "duplicate_addresses": dup_addrs,
        "country_distribution": country_dist,
        "name_char_len": calc_stats(name_len),
        "name_word_cnt": calc_stats(name_words),
        "addr_char_len": calc_stats(addr_len),
        "addr_word_cnt": calc_stats(addr_words)
    }
    print(f"  Profiled {name} in {time.time() - t_prof:.2f}s")
    return res

profile_s1 = profile_dataset(df_s1, "Source 1")
profile_s2 = profile_dataset(df_s2, "Source 2")
profile_s3 = profile_dataset(df_s3, "Source 3")

profiles = {"S1": profile_s1, "S2": profile_s2, "S3": profile_s3}
with open(os.path.join(OUTPUT_DIR, "dataset_profiling.json"), "w") as f:
    json.dump(profiles, f, indent=2)

# ---------------------------------------------------------
# 3. Ground Truth Analysis (Fast)
# ---------------------------------------------------------
print("\n[Step 3] Ground Truth Analysis...")
t_gt = time.time()

gt_s1_ids = df_gt['source1_entity_id'].to_numpy()
gt_matched_str = df_gt['matched_entity_ids'].to_numpy()

total_s1 = int(len(df_gt))
singletons = 0
s2_only_cnt = 0
s3_only_cnt = 0
s2_s3_cnt = 0

gt_pairs_dict = {}
total_gt_pairs = 0
s1_to_s2_gt_pairs = 0
s1_to_s3_gt_pairs = 0
match_count_counter = Counter()

for s1_id, m_str in zip(gt_s1_ids, gt_matched_str):
    if not m_str:
        singletons += 1
        match_count_counter[0] += 1
    else:
        m_list = [x for x in m_str.split(',') if x]
        mc = len(m_list)
        match_count_counter[mc] += 1
        gt_pairs_dict[s1_id] = set(m_list)
        
        has_s2 = False
        has_s3 = False
        for m in m_list:
            total_gt_pairs += 1
            if m.startswith('S2-'):
                has_s2 = True
                s1_to_s2_gt_pairs += 1
            elif m.startswith('S3-'):
                has_s3 = True
                s1_to_s3_gt_pairs += 1
                
        if has_s2 and has_s3:
            s2_s3_cnt += 1
        elif has_s2:
            s2_only_cnt += 1
        elif has_s3:
            s3_only_cnt += 1

singleton_pct = float(singletons / total_s1 * 100)

gt_stats = {
    "total_s1_entities": total_s1,
    "singletons": singletons,
    "singleton_percentage": round(singleton_pct, 4),
    "matched_s1_entities": total_s1 - singletons,
    "match_category_breakdown": {
        "Singleton (0 matches)": singletons,
        "S2 Only Match": s2_only_cnt,
        "S3 Only Match": s3_only_cnt,
        "S2 + S3 Match": s2_s3_cnt
    },
    "match_count_distribution": {int(k): int(v) for k, v in sorted(match_count_counter.items())},
    "total_ground_truth_pairs": total_gt_pairs,
    "s1_s2_gt_pairs": s1_to_s2_gt_pairs,
    "s1_s3_gt_pairs": s1_to_s3_gt_pairs
}

with open(os.path.join(OUTPUT_DIR, "ground_truth_stats.json"), "w") as f:
    json.dump(gt_stats, f, indent=2)

print(f"  Ground truth processed in {time.time() - t_gt:.2f}s")
print(f"  Total S1 Entities: {total_s1:,}")
print(f"  Singletons (0 matches): {singletons:,} ({singleton_pct:.2f}%)")
print(f"  Matched S1 Entities: {total_s1 - singletons:,} ({100 - singleton_pct:.2f}%)")
print(f"  Match Category Breakdown: S2-only={s2_only_cnt:,}, S3-only={s3_only_cnt:,}, S2+S3={s2_s3_cnt:,}")
print(f"  Total GT Pairs: {total_gt_pairs:,} (S1-S2: {s1_to_s2_gt_pairs:,}, S1-S3: {s1_to_s3_gt_pairs:,})")

# ---------------------------------------------------------
# 4. Sampling Ground Truth Matching Examples
# ---------------------------------------------------------
print("\n[Step 4] Fast Lookup & Sampling 25 GT Examples...")

s1_lookup = {row[1]: {'business_name': row[2], 'business_address': row[3], 'country': row[4]} for row in df_s1.itertuples()}
s2_lookup = {row[1]: {'business_name': row[2], 'business_address': row[3], 'country': row[4]} for row in df_s2.itertuples()}
s3_lookup = {row[1]: {'business_name': row[2], 'business_address': row[3], 'country': row[4]} for row in df_s3.itertuples()}

def get_record(eid):
    if eid.startswith('S1-'): return s1_lookup.get(eid)
    if eid.startswith('S2-'): return s2_lookup.get(eid)
    if eid.startswith('S3-'): return s3_lookup.get(eid)
    return None

examples_data = []

# Select diverse examples
sample_keys = list(gt_pairs_dict.keys())[:50]
for s1_id in sample_keys:
    m_set = gt_pairs_dict[s1_id]
    s1_rec = s1_lookup.get(s1_id)
    if not s1_rec: continue
    
    m_recs = []
    for m_id in m_set:
        m_r = get_record(m_id)
        if m_r:
            m_r_c = dict(m_r)
            m_r_c['entity_id'] = m_id
            m_recs.append(m_r_c)
            
    has_s2 = any(m.startswith('S2-') for m in m_set)
    has_s3 = any(m.startswith('S3-') for m in m_set)
    cat = "S2+S3" if (has_s2 and has_s3) else ("S2-only" if has_s2 else "S3-only")
    
    examples_data.append({
        "s1_id": s1_id,
        "country": s1_rec['country'],
        "match_category": cat,
        "s1_record": {"entity_id": s1_id, **s1_rec},
        "matched_records": m_recs
    })
    
    if len(examples_data) >= 30:
        break

with open(os.path.join(OUTPUT_DIR, "sampled_gt_examples.json"), "w") as f:
    json.dump(examples_data, f, indent=2)

print(f"  Sampled {len(examples_data)} examples saved to scratch/sampled_gt_examples.json")

# ---------------------------------------------------------
# 5. Normalization Functions & Application
# ---------------------------------------------------------
print("\n[Step 5] Fast Normalization...")

LEGAL_SUFFIX_RE = re.compile(r'\b(inc|incorporated|corp|corporation|ltd|limited|llc|co|company|pvt|private|pvt ltd|private limited|plc|gmbh|sa|bv|srl|llp|holdings|group|enterprises|enterprise|services|solutions|tech|technologies|international|intl)\b', re.IGNORECASE)
NON_ALPHANUM_RE = re.compile(r'[^a-z0-9\s]')
SPACE_RE = re.compile(r'\s+')

def normalize_text_single(text):
    if not text or text is None:
        return ""
    text = unicodedata.normalize('NFKD', str(text)).encode('ASCII', 'ignore').decode('utf-8')
    text = text.lower().replace('&', ' and ')
    text = NON_ALPHANUM_RE.sub(' ', text)
    text = SPACE_RE.sub(' ', text).strip()
    return text

def normalize_business_name_single(name):
    norm = normalize_text_single(name)
    if not norm:
        return ""
    norm = LEGAL_SUFFIX_RE.sub(' ', norm)
    norm = SPACE_RE.sub(' ', norm).strip()
    return norm

ADDRESS_SUB_MAP = {
    r'\brd\b': 'road',
    r'\bst\b': 'street',
    r'\bave\b': 'avenue',
    r'\bblvd\b': 'boulevard',
    r'\bdr\b': 'drive',
    r'\bln\b': 'lane',
    r'\bct\b': 'court',
    r'\bpl\b': 'place',
    r'\bsq\b': 'square',
    r'\bhwy\b': 'highway',
    r'\bpkwy\b': 'parkway',
    r'\bste\b': 'suite',
    r'\bapt\b': 'apartment',
    r'\bfl\b': 'floor',
    r'\bbldg\b': 'building',
    r'\bctr\b': 'center',
    r'\bno\b': 'number',
    r'\bpo box\b': 'pobox'
}
ADDR_COMPILED = [(re.compile(pat, re.IGNORECASE), rep) for pat, rep in ADDRESS_SUB_MAP.items()]

def normalize_business_address_single(addr):
    norm = normalize_text_single(addr)
    if not norm:
        return ""
    for pat_re, rep in ADDR_COMPILED:
        norm = pat_re.sub(rep, norm)
    norm = SPACE_RE.sub(' ', norm).strip()
    return norm

print("  Applying normalization to S1, S2, S3 via python list comprehension...")
for df_curr, name in [(df_s1, "S1"), (df_s2, "S2"), (df_s3, "S3")]:
    t0 = time.time()
    df_curr['norm_business_name'] = [normalize_business_name_single(x) for x in df_curr['business_name']]
    df_curr['norm_business_address'] = [normalize_business_address_single(x) for x in df_curr['business_address']]
    print(f"  Normalized {name} in {time.time() - t0:.2f}s")

# ---------------------------------------------------------
# 6. Baseline Exact-Normalized-Name Blocking Strategy
# ---------------------------------------------------------
print("\n[Step 6] Fast Baseline Blocking Evaluation...")
t0 = time.time()

s2_index = {}
for eid, c, nbn in zip(df_s2['entity_id'], df_s2['country'], df_s2['norm_business_name']):
    if nbn:
        key = (c, nbn)
        if key not in s2_index:
            s2_index[key] = []
        s2_index[key].append(eid)

s3_index = {}
for eid, c, nbn in zip(df_s3['entity_id'], df_s3['country'], df_s3['norm_business_name']):
    if nbn:
        key = (c, nbn)
        if key not in s3_index:
            s3_index[key] = []
        s3_index[key].append(eid)

print(f"  Index built in {time.time() - t0:.2f}s")

t0 = time.time()
retrieved_gt_pairs = 0
total_retrieved_candidates = 0

for s1_id, c, nbn in zip(df_s1['entity_id'], df_s1['country'], df_s1['norm_business_name']):
    if not nbn:
        continue
    key = (c, nbn)
    cands_s2 = s2_index.get(key)
    cands_s3 = s3_index.get(key)
    
    num_cand = (len(cands_s2) if cands_s2 else 0) + (len(cands_s3) if cands_s3 else 0)
    if num_cand == 0:
        continue
        
    total_retrieved_candidates += num_cand
    
    gt_set = gt_pairs_dict.get(s1_id)
    if gt_set:
        all_cands = set()
        if cands_s2: all_cands.update(cands_s2)
        if cands_s3: all_cands.update(cands_s3)
        retrieved_gt_pairs += len(gt_set.intersection(all_cands))

query_time = time.time() - t0
print(f"  Blocking query finished in {query_time:.2f}s")

candidate_recall = float(retrieved_gt_pairs / total_gt_pairs) if total_gt_pairs > 0 else 0.0
avg_candidates_per_s1 = float(total_retrieved_candidates / total_s1)
max_possible_pairs = float(total_s1 * (len(df_s2) + len(df_s3)))
candidate_reduction_ratio = float(1.0 - (total_retrieved_candidates / max_possible_pairs))

blocking_results = {
    "blocking_strategy": "Exact Normalized Business Name + Country",
    "total_s1_entities": total_s1,
    "total_gt_pairs": total_gt_pairs,
    "retrieved_gt_pairs": int(retrieved_gt_pairs),
    "candidate_recall": round(candidate_recall, 6),
    "candidate_recall_percentage": round(candidate_recall * 100, 2),
    "total_retrieved_candidates": int(total_retrieved_candidates),
    "avg_candidates_per_s1": round(avg_candidates_per_s1, 4),
    "max_possible_pairs": max_possible_pairs,
    "candidate_reduction_ratio": round(candidate_reduction_ratio, 10),
    "candidate_reduction_percentage": round(candidate_reduction_ratio * 100, 8)
}

with open(os.path.join(OUTPUT_DIR, "baseline_blocking_results.json"), "w") as f:
    json.dump(blocking_results, f, indent=2)

print("\n--- BASELINE BLOCKING RESULTS ---")
print(f"Candidate Recall: {candidate_recall * 100:.2f}% ({retrieved_gt_pairs:,} / {total_gt_pairs:,} GT pairs)")
print(f"Average Candidates per S1: {avg_candidates_per_s1:.4f}")
print(f"Total Candidate Pairs Generated: {total_retrieved_candidates:,}")
print(f"Candidate Reduction Ratio: {candidate_reduction_ratio * 100:.8f}%")

print(f"\nPhase 1 Pipeline finished successfully in {time.time() - start_time:.2f}s!")
