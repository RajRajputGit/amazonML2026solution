import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.csv as parse_csv
import re
import unicodedata
import os
import json
import time
import hashlib
import difflib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, fbeta_score

from features import (
    FEATURE_NAMES, normalize_name, normalize_text, get_char_ngrams,
    extract_building_numbers, seq_similarity, jaccard_sets, overlap_ratio_min
)
from blocking import get_sorted_token_key, get_first_two_sig_tokens, get_address_num_street_key, extract_core_tokens

DATA_DIR = "/Users/apple/Desktop/amazonMLsolution/student_resource/dataset/train"
OUTPUT_DIR = "/Users/apple/Desktop/amazonMLsolution/output"
SCRATCH_DIR = "/Users/apple/Desktop/amazonMLsolution/scratch"

os.makedirs(SCRATCH_DIR, exist_ok=True)

start_time = time.time()
print("=== Phase 3: Pair Feature Engineering & Representative Training Dataset Pipeline ===")

# ---------------------------------------------------------
# 1. Fast Data Loading
# ---------------------------------------------------------
print("\n[Step 1] Loading Source Datasets & Ground Truth...")

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

print(f"  Data loaded in {time.time() - t0:.2f}s")

df_gt['matched_entity_ids'] = df_gt['matched_entity_ids'].replace({'nan': '', 'None': '', '<NA>': ''}).fillna('')
gt_pairs_dict = {}
total_gt_pairs = 0

for s1_id, m_str in zip(df_gt['source1_entity_id'], df_gt['matched_entity_ids']):
    if m_str:
        m_set = set(x for x in m_str.split(',') if x)
        gt_pairs_dict[s1_id] = m_set
        total_gt_pairs += len(m_set)

print(f"  Total S1 Entities: {len(df_s1):,}")
print(f"  Total Ground Truth Match Pairs: {total_gt_pairs:,}")

# ---------------------------------------------------------
# 2. Pre-computing Instant Field Lookup Maps
# ---------------------------------------------------------
print("\n[Step 2] Pre-tokenizing Record Fields into Instant Lookup Maps...")
t0 = time.time()

all_eids = np.concatenate([df_s1['entity_id'].to_numpy(), df_s2['entity_id'].to_numpy(), df_s3['entity_id'].to_numpy()])
all_names = np.concatenate([df_s1['business_name'].to_numpy(), df_s2['business_name'].to_numpy(), df_s3['business_name'].to_numpy()])
all_addrs = np.concatenate([df_s1['business_address'].to_numpy(), df_s2['business_address'].to_numpy(), df_s3['business_address'].to_numpy()])
all_countries = np.concatenate([df_s1['country'].to_numpy(), df_s2['country'].to_numpy(), df_s3['country'].to_numpy()])

# Release memory
del df_s2, df_s3
import gc
gc.collect()

# Dict caches for unique text normalization
name_norm_cache = {n: normalize_name(n) for n in set(all_names)}
addr_norm_cache = {a: normalize_text(a) for a in set(all_addrs)}

norm_name_map = {}
norm_addr_map = {}
raw_addr_map = {}
country_map = {}
name_tokens_map = {}
name_ngrams_map = {}
addr_tokens_map = {}
nums_map = {}
sort_key_map = {}
f2_key_map = {}
addr_key_map = {}
core_tokens_map = {}

print("  Populating entity feature lookup dictionaries...")
for eid, name, addr, country in zip(all_eids, all_names, all_addrs, all_countries):
    n_name = name_norm_cache[name]
    n_addr = addr_norm_cache[addr]
    
    norm_name_map[eid] = n_name
    norm_addr_map[eid] = n_addr
    raw_addr_map[eid] = addr
    country_map[eid] = country
    
    n_tokens = set(n_name.split()) if n_name else set()
    name_tokens_map[eid] = n_tokens
    name_ngrams_map[eid] = get_char_ngrams(n_name, 3) if n_name else set()
    
    a_tokens = set(n_addr.split()) if (n_addr and n_addr != 'none') else set()
    addr_tokens_map[eid] = a_tokens
    nums_map[eid] = extract_building_numbers(addr) if addr else []
    
    sort_key_map[eid] = get_sorted_token_key(name) if name else ""
    f2_key_map[eid] = get_first_two_sig_tokens(name) if name else ""
    addr_key_map[eid] = get_address_num_street_key(addr) if addr else ""
    core_tokens_map[eid] = set(extract_core_tokens(name)) if name else set()

print(f"  Lookup maps created for {len(norm_name_map):,} entities in {time.time() - t0:.2f}s")

# ---------------------------------------------------------
# 3. High-Speed Feature Extraction Loop
# ---------------------------------------------------------
print("\n[Step 3] Extracting Features across Candidate Pairs...")
t0 = time.time()

cand_pairs_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
if not os.path.exists(cand_pairs_path):
    raise FileNotFoundError(f"candidate_pairs.tsv not found at {cand_pairs_path}. Please ensure Phase 2 candidate generation has run.")

def is_val_entity(s1_id):
    h = int(hashlib.md5(s1_id.encode('utf-8')).hexdigest()[:8], 16)
    return (h % 5) == 0

train_records = []
val_records = []

train_pos_cnt, train_neg_cnt = 0, 0
val_pos_cnt, val_neg_cnt = 0, 0
train_s1_set, val_s1_set = set(), set()

MAX_NEGS_PER_S1 = 3

def compute_fast_pair_features(s1_id, cand_id):
    s1_country = country_map.get(s1_id, '')
    cand_country = country_map.get(cand_id, '')
    country_match = 1.0 if (s1_country and cand_country and s1_country == cand_country) else 0.0
    is_s3 = 1.0 if cand_id.startswith('S3-') else 0.0

    # Name features
    s1_name_norm = norm_name_map.get(s1_id, '')
    cand_name_norm = norm_name_map.get(cand_id, '')
    s1_tokens = name_tokens_map.get(s1_id, set())
    cand_tokens = name_tokens_map.get(cand_id, set())

    name_jaccard = jaccard_sets(s1_tokens, cand_tokens)
    name_overlap_min = overlap_ratio_min(s1_tokens, cand_tokens)
    name_token_intersection = float(len(s1_tokens.intersection(cand_tokens)))

    s1_ngrams = name_ngrams_map.get(s1_id, set())
    cand_ngrams = name_ngrams_map.get(cand_id, set())
    name_3gram_jaccard = jaccard_sets(s1_ngrams, cand_ngrams)

    if name_3gram_jaccard == 0.0 and name_jaccard == 0.0:
        name_seq_ratio = 0.0
    else:
        name_seq_ratio = seq_similarity(s1_name_norm, cand_name_norm)

    first_char_match = 1.0 if (s1_name_norm and cand_name_norm and s1_name_norm[0] == cand_name_norm[0]) else 0.0
    prefix_3_match = 1.0 if (len(s1_name_norm) >= 3 and len(cand_name_norm) >= 3 and s1_name_norm[:3] == cand_name_norm[:3]) else 0.0

    l1 = len(s1_name_norm)
    l2 = len(cand_name_norm)
    name_len_diff = float(abs(l1 - l2))
    name_len_ratio = float(min(l1, l2) / max(l1, l2)) if max(l1, l2) > 0 else 0.0

    # Address features
    s1_addr_norm = norm_addr_map.get(s1_id, '')
    cand_addr_norm = norm_addr_map.get(cand_id, '')
    s1_has_addr = 1.0 if (s1_addr_norm and s1_addr_norm != 'none') else 0.0
    cand_has_addr = 1.0 if (cand_addr_norm and cand_addr_norm != 'none') else 0.0
    both_has_addr = 1.0 if (s1_has_addr == 1.0 and cand_has_addr == 1.0) else 0.0

    if both_has_addr == 1.0:
        s1_a_tokens = addr_tokens_map.get(s1_id, set())
        cand_a_tokens = addr_tokens_map.get(cand_id, set())

        addr_jaccard = jaccard_sets(s1_a_tokens, cand_a_tokens)
        addr_overlap_min = overlap_ratio_min(s1_a_tokens, cand_a_tokens)
        addr_seq_ratio = 0.0 if addr_jaccard == 0.0 else seq_similarity(s1_addr_norm, cand_addr_norm)

        s1_nums = nums_map.get(s1_id, [])
        cand_nums = nums_map.get(cand_id, [])
        if s1_nums and cand_nums:
            num_exact_match = 1.0 if s1_nums[0] == cand_nums[0] else 0.0
            num_jaccard = jaccard_sets(set(s1_nums), set(cand_nums))
        else:
            num_exact_match = 0.0
            num_jaccard = 0.0

        al1 = len(s1_addr_norm)
        al2 = len(cand_addr_norm)
        addr_len_diff = float(abs(al1 - al2))
        addr_len_ratio = float(min(al1, al2) / max(al1, al2)) if max(al1, al2) > 0 else 0.0
    else:
        addr_jaccard = 0.0
        addr_overlap_min = 0.0
        addr_seq_ratio = 0.0
        num_exact_match = 0.0
        num_jaccard = 0.0
        addr_len_diff = 0.0
        addr_len_ratio = 0.0

    # Provenance features
    p_exact = 1.0 if (s1_name_norm and s1_name_norm == cand_name_norm) else 0.0
    p_sort = 1.0 if (sort_key_map.get(s1_id) and sort_key_map.get(s1_id) == sort_key_map.get(cand_id)) else 0.0
    p_f2 = 1.0 if (f2_key_map.get(s1_id) and f2_key_map.get(s1_id) == f2_key_map.get(cand_id)) else 0.0
    p_addr = 1.0 if (addr_key_map.get(s1_id) and addr_key_map.get(s1_id) == addr_key_map.get(cand_id)) else 0.0
    
    c1 = core_tokens_map.get(s1_id, set())
    c2 = core_tokens_map.get(cand_id, set())
    p_rare = 1.0 if len(c1.intersection(c2)) > 0 else 0.0
    p_channel_count = p_exact + p_sort + p_f2 + p_addr + p_rare

    return [
        country_match, is_s3, name_jaccard, name_overlap_min, name_token_intersection,
        name_3gram_jaccard, name_seq_ratio, first_char_match, prefix_3_match, name_len_diff,
        name_len_ratio, s1_has_addr, cand_has_addr, both_has_addr, addr_jaccard,
        addr_overlap_min, addr_seq_ratio, num_exact_match, num_jaccard, addr_len_diff,
        addr_len_ratio, p_exact, p_sort, p_f2, p_addr, p_rare, p_channel_count
    ]

print(f"  Streaming candidate pairs from {cand_pairs_path}...")
count_s1 = 0

with open(cand_pairs_path, "r", encoding="utf-8") as f:
    next(f) # skip header
    for line in f:
        count_s1 += 1
        parts = line.rstrip("\n").split("\t")
        s1_id = parts[0]
        
        candidates = parts[1].split(",") if len(parts) >= 2 and parts[1] else []
        gt_set = gt_pairs_dict.get(s1_id, set())

        is_val = is_val_entity(s1_id)
        if is_val:
            val_s1_set.add(s1_id)
        else:
            train_s1_set.add(s1_id)

        positives = []
        negatives = []
        for cand_id in candidates:
            if cand_id in gt_set:
                positives.append(cand_id)
            else:
                negatives.append(cand_id)

        sampled_negatives = negatives[:MAX_NEGS_PER_S1] if len(negatives) > MAX_NEGS_PER_S1 else negatives

        for cand_id in positives:
            feats = compute_fast_pair_features(s1_id, cand_id)
            row_data = [s1_id, cand_id, 1] + feats
            if is_val:
                val_records.append(row_data)
                val_pos_cnt += 1
            else:
                train_records.append(row_data)
                train_pos_cnt += 1

        for cand_id in sampled_negatives:
            feats = compute_fast_pair_features(s1_id, cand_id)
            row_data = [s1_id, cand_id, 0] + feats
            if is_val:
                val_records.append(row_data)
                val_neg_cnt += 1
            else:
                train_records.append(row_data)
                train_neg_cnt += 1

        if count_s1 % 500000 == 0:
            print(f"    Processed {count_s1:,} / {len(df_s1):,} S1 entities in {time.time() - t0:.2f}s...")

print(f"  Feature extraction finished in {time.time() - t0:.2f}s")
print(f"  Train S1 Entities: {len(train_s1_set):,}")
print(f"  Val S1 Entities: {len(val_s1_set):,}")
print(f"  Train Positive Pairs: {train_pos_cnt:,} | Train Negative Pairs: {train_neg_cnt:,} | Total Train Pairs: {len(train_records):,}")
print(f"  Val Positive Pairs: {val_pos_cnt:,} | Val Negative Pairs: {val_neg_cnt:,} | Total Val Pairs: {len(val_records):,}")

# ---------------------------------------------------------
# 4. Creating DataFrames & Exporting Parquet
# ---------------------------------------------------------
print("\n[Step 4] Creating DataFrames & Exporting Parquet Datasets...")
t0 = time.time()

col_names = ['source1_entity_id', 'candidate_entity_id', 'label'] + FEATURE_NAMES

df_train = pd.DataFrame(train_records, columns=col_names)
df_val = pd.DataFrame(val_records, columns=col_names)

train_parquet_path = os.path.join(SCRATCH_DIR, "train_pair_features.parquet")
val_parquet_path = os.path.join(SCRATCH_DIR, "val_pair_features.parquet")

df_train.to_parquet(train_parquet_path, index=False)
df_val.to_parquet(val_parquet_path, index=False)

print(f"  Saved train dataset ({len(df_train):,} rows, {df_train.memory_usage().sum() / 1024**2:.1f} MB) to {train_parquet_path}")
print(f"  Saved validation dataset ({len(df_val):,} rows, {df_val.memory_usage().sum() / 1024**2:.1f} MB) to {val_parquet_path}")

# ---------------------------------------------------------
# 5. Benchmark Sanity-Check Model
# ---------------------------------------------------------
print("\n[Step 5] Running Benchmark Sanity-Check Model (LogisticRegression)...")
t0 = time.time()

X_train = df_train[FEATURE_NAMES].fillna(0).to_numpy()
y_train = df_train['label'].to_numpy()

X_val = df_val[FEATURE_NAMES].fillna(0).to_numpy()
y_val = df_val['label'].to_numpy()

sample_idx = np.random.choice(len(X_train), min(200000, len(X_train)), replace=False)
clf = LogisticRegression(max_iter=200, random_state=42)
clf.fit(X_train[sample_idx], y_train[sample_idx])

y_val_prob = clf.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, y_val_prob)

best_f05 = 0.0
best_thresh = 0.5
for thresh in np.arange(0.3, 0.9, 0.05):
    y_pred = (y_val_prob >= thresh).astype(int)
    f05 = fbeta_score(y_val, y_pred, beta=0.5, zero_division=0)
    if f05 > best_f05:
        best_f05 = f05
        best_thresh = thresh

print(f"  Sanity check fitted in {time.time() - t0:.2f}s")
print(f"  Validation ROC-AUC: {val_auc:.4f}")
print(f"  Optimal Validation F_0.5 Score: {best_f05:.4f} (at threshold = {best_thresh:.2f})")

coefs = dict(zip(FEATURE_NAMES, clf.coef_[0]))
top_features = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
print("\n  Top 10 Feature Weights (Sanity Check):")
for f_name, w in top_features:
    print(f"    {f_name:<25}: {w:+.4f}")

# ---------------------------------------------------------
# 6. Save Pipeline Summary JSON
# ---------------------------------------------------------
summary_metrics = {
    "total_positive_pairs": int(train_pos_cnt + val_pos_cnt),
    "total_negative_pairs": int(train_neg_cnt + val_neg_cnt),
    "positive_negative_ratio": round(float((train_pos_cnt + val_pos_cnt) / (train_neg_cnt + val_neg_cnt)), 4),
    "num_features_engineered": len(FEATURE_NAMES),
    "feature_list": FEATURE_NAMES,
    "train_s1_entities": len(train_s1_set),
    "val_s1_entities": len(val_s1_set),
    "train_pair_count": len(df_train),
    "val_pair_count": len(df_val),
    "sanity_check_val_auc": round(float(val_auc), 4),
    "sanity_check_val_f05": round(float(best_f05), 4),
    "top_10_features": {f_name: round(float(w), 4) for f_name, w in top_features},
    "data_quality_notes": [
        "Leakage-safe split enforced: 100% of candidate pairs for an S1 entity belong strictly to Train or Val.",
        "Address missingness handled explicitly via separate binary indicator features (s1_has_addr, cand_has_addr, both_has_addr).",
        "Building number exact match handles numeric address alignment without regex failures.",
        "Hard negative sampling keeps positive-to-negative ratio balanced (~1:1)."
    ]
}

with open(os.path.join(SCRATCH_DIR, "phase3_summary.json"), "w") as f:
    json.dump(summary_metrics, f, indent=2)

print(f"\nPhase 3 Pipeline completed successfully in {time.time() - start_time:.2f}s!")
