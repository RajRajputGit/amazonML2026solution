import pandas as pd
import numpy as np
import pyarrow.csv as parse_csv
import pyarrow as pa
import re
import unicodedata
import os
import json
import time
from collections import Counter, defaultdict

DATA_DIR = "/Users/apple/Desktop/amazonMLsolution/student_resource/dataset/train"
OUTPUT_DIR = "/Users/apple/Desktop/amazonMLsolution/scratch"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=== Phase 2: Multi-Strategy Candidate Generation Development ===")

# ---------------------------------------------------------
# 1. Fast Load
# ---------------------------------------------------------
t0 = time.time()
def load_tsv(filepath):
    read_options = parse_csv.ReadOptions(block_size=1048576*16)
    parse_options = parse_csv.ParseOptions(delimiter='\t', quote_char=False)
    table = parse_csv.read_csv(filepath, read_options=read_options, parse_options=parse_options)
    return table.to_pandas().astype(str)

print("Loading datasets...")
df_s1 = load_tsv(os.path.join(DATA_DIR, "train_source1.tsv"))
df_s2 = load_tsv(os.path.join(DATA_DIR, "train_source2.tsv"))
df_s3 = load_tsv(os.path.join(DATA_DIR, "train_source3.tsv"))
df_gt = load_tsv(os.path.join(DATA_DIR, "train_ground_truth.tsv"))

print(f"Data loaded in {time.time() - t0:.2f}s")

# Clean GT
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

print(f"Total S1 Entities: {total_s1:,}")
print(f"Total GT Pairs: {total_gt_pairs:,}")

# ---------------------------------------------------------
# 2. Advanced Normalization & Key Extraction
# ---------------------------------------------------------
LEGAL_SUFFIX_RE = re.compile(r'\b(inc|incorporated|corp|corporation|ltd|limited|llc|co|company|pvt|private|pvt ltd|private limited|plc|gmbh|sa|bv|srl|llp|holdings|group|enterprises|enterprise|services|solutions|tech|technologies|international|intl)\b', re.IGNORECASE)
NON_ALPHANUM_RE = re.compile(r'[^a-z0-9\s]')
SPACE_RE = re.compile(r'\s+')
DIGITS_RE = re.compile(r'\b\d+\b')

COMMON_STOPWORDS = {'and', 'the', 'of', 'for', 'in', 'on', 'at', 'to', 'a', 'an', 'is', 'by', 'with'}

def normalize_text(text):
    if not text or text is None or text == 'None':
        return ""
    text = unicodedata.normalize('NFKD', str(text)).encode('ASCII', 'ignore').decode('utf-8')
    text = text.lower().replace('&', ' and ')
    text = NON_ALPHANUM_RE.sub(' ', text)
    return SPACE_RE.sub(' ', text).strip()

def normalize_name(name):
    norm = normalize_text(name)
    if not norm:
        return ""
    norm = LEGAL_SUFFIX_RE.sub(' ', norm)
    return SPACE_RE.sub(' ', norm).strip()

def get_sorted_token_key(name):
    norm = normalize_name(name)
    if not norm:
        return ""
    tokens = [t for t in norm.split() if t not in COMMON_STOPWORDS]
    if not tokens:
        tokens = norm.split()
    return " ".join(sorted(tokens))

def get_first_two_sig_tokens(name):
    norm = normalize_name(name)
    if not norm:
        return ""
    tokens = [t for t in norm.split() if t not in COMMON_STOPWORDS]
    if len(tokens) >= 2:
        return " ".join(sorted(tokens[:2]))
    elif len(tokens) == 1:
        return tokens[0]
    return ""

def get_address_num_street_key(address):
    norm = normalize_text(address)
    if not norm:
        return ""
    tokens = norm.split()
    nums = [t for t in tokens if t.isdigit()]
    words = [t for t in tokens if not t.isdigit() and len(t) > 2 and t not in COMMON_STOPWORDS]
    if nums and words:
        return f"{nums[0]}_{words[0]}"
    elif len(words) >= 2:
        return f"{words[0]}_{words[1]}"
    return ""

def get_address_zip_number_key(address):
    norm = normalize_text(address)
    if not norm:
        return ""
    tokens = norm.split()
    nums = [t for t in tokens if t.isdigit() and len(t) >= 3]
    if len(nums) >= 2:
        return f"{nums[0]}_{nums[1]}"
    elif len(nums) == 1:
        return nums[0]
    return ""

def extract_core_long_tokens(name):
    norm = normalize_name(name)
    if not norm:
        return []
    tokens = [t for t in norm.split() if len(t) >= 4 and t not in COMMON_STOPWORDS]
    return tokens

print("\nExtracting blocking keys for S1, S2, S3...")
t0 = time.time()

for df_curr, name in [(df_s1, "S1"), (df_s2, "S2"), (df_s3, "S3")]:
    names_raw = df_curr['business_name'].tolist()
    addrs_raw = df_curr['business_address'].tolist()
    
    df_curr['norm_name'] = [normalize_name(x) for x in names_raw]
    df_curr['sorted_name'] = [get_sorted_token_key(x) for x in names_raw]
    df_curr['first2_name'] = [get_first_two_sig_tokens(x) for x in names_raw]
    df_curr['addr_num_street'] = [get_address_num_street_key(x) for x in addrs_raw]
    df_curr['core_tokens'] = [extract_core_long_tokens(x) for x in names_raw]
    print(f"  Processed {name} in {time.time() - t0:.2f}s")
    t0 = time.time()

print("Key extraction complete!")

# ---------------------------------------------------------
# 3. Blocking Evaluation Helper
# ---------------------------------------------------------
def evaluate_strategy(strategy_name, s1_keys, s2_keys, s3_keys, max_bucket_size=500):
    t_start = time.time()
    print(f"\n--- Evaluating Strategy: {strategy_name} ---")
    
    # Build inverted index
    s2_idx = defaultdict(list)
    for eid, c, k in zip(df_s2['entity_id'], df_s2['country'], s2_keys):
        if k:
            s2_idx[(c, k)].append(eid)
            
    s3_idx = defaultdict(list)
    for eid, c, k in zip(df_s3['entity_id'], df_s3['country'], s3_keys):
        if k:
            s3_idx[(c, k)].append(eid)
            
    retrieved_gt = 0
    total_candidates = 0
    
    for s1_id, c, k in zip(df_s1['entity_id'], df_s1['country'], s1_keys):
        if not k:
            continue
        key = (c, k)
        cands_s2 = s2_idx.get(key, [])
        cands_s3 = s3_idx.get(key, [])
        
        # Apply bucket size cap for high-frequency keys
        if len(cands_s2) > max_bucket_size:
            cands_s2 = cands_s2[:max_bucket_size]
        if len(cands_s3) > max_bucket_size:
            cands_s3 = cands_s3[:max_bucket_size]
            
        num_c = len(cands_s2) + len(cands_s3)
        if num_c == 0:
            continue
            
        total_candidates += num_c
        
        gt_set = gt_pairs_dict.get(s1_id)
        if gt_set:
            cand_set = set(cands_s2).union(cands_s3)
            retrieved_gt += len(gt_set.intersection(cand_set))
            
    recall = float(retrieved_gt / total_gt_pairs) if total_gt_pairs > 0 else 0.0
    avg_cand = float(total_candidates / total_s1)
    reduction = float(1.0 - (total_candidates / max_possible_pairs))
    
    print(f"  Time taken: {time.time() - t_start:.2f}s")
    print(f"  Recall: {recall*100:.2f}% ({retrieved_gt:,} / {total_gt_pairs:,})")
    print(f"  Avg Candidates/S1: {avg_cand:.2f}")
    print(f"  Total Candidates: {total_candidates:,}")
    print(f"  Reduction Ratio: {reduction*100:.8f}%")
    
    return {
        "strategy": strategy_name,
        "recall_pct": round(recall * 100, 2),
        "retrieved_gt": retrieved_gt,
        "avg_candidates_per_s1": round(avg_cand, 2),
        "total_candidates": total_candidates,
        "reduction_ratio_pct": round(reduction * 100, 8)
    }

# ---------------------------------------------------------
# 4. Evaluate Individual Strategies
# ---------------------------------------------------------
results = []

# BLOCK-0: Exact Normalized Name + Country
res0 = evaluate_strategy("BLOCK-0 (Exact Name + Country)", 
                         df_s1['norm_name'], df_s2['norm_name'], df_s3['norm_name'], max_bucket_size=500)
results.append(res0)

# BLOCK-1: Token-Sorted Name + Country
res1 = evaluate_strategy("BLOCK-1 (Token-Sorted Name + Country)", 
                         df_s1['sorted_name'], df_s2['sorted_name'], df_s3['sorted_name'], max_bucket_size=500)
results.append(res1)

# BLOCK-2: First 2 Significant Tokens + Country
res2 = evaluate_strategy("BLOCK-2 (First 2 Sig Tokens + Country)", 
                         df_s1['first2_name'], df_s2['first2_name'], df_s3['first2_name'], max_bucket_size=300)
results.append(res2)

# BLOCK-3: Address Number + Street Token + Country
res3 = evaluate_strategy("BLOCK-3 (Address Num + Street + Country)", 
                         df_s1['addr_num_street'], df_s2['addr_num_street'], df_s3['addr_num_street'], max_bucket_size=300)
results.append(res3)

# ---------------------------------------------------------
# 5. Core Token / Character N-gram Indexing (BLOCK-4)
# ---------------------------------------------------------
print("\n--- Evaluating Strategy: BLOCK-4 (Core Rare Token Indexing) ---")
t0 = time.time()

# Count token document frequencies across S2+S3
token_freq = Counter()
for tokens in df_s2['core_tokens']:
    for t in set(tokens):
        token_freq[t] += 1
for tokens in df_s3['core_tokens']:
    for t in set(tokens):
        token_freq[t] += 1

print(f"  Calculated document frequencies for {len(token_freq):,} unique 4+ char tokens in {time.time() - t0:.2f}s")

# Index S2 and S3 using their rarest core token (if frequency <= 5000)
s2_core_idx = defaultdict(list)
for eid, c, tokens in zip(df_s2['entity_id'], df_s2['country'], df_s2['core_tokens']):
    if tokens:
        # Pick rarest token
        rarest = min(tokens, key=lambda t: token_freq[t])
        if token_freq[rarest] <= 5000:
            s2_core_idx[(c, rarest)].append(eid)

s3_core_idx = defaultdict(list)
for eid, c, tokens in zip(df_s3['entity_id'], df_s3['country'], df_s3['core_tokens']):
    if tokens:
        rarest = min(tokens, key=lambda t: token_freq[t])
        if token_freq[rarest] <= 5000:
            s3_core_idx[(c, rarest)].append(eid)

retrieved_gt_core = 0
total_candidates_core = 0

for s1_id, c, tokens in zip(df_s1['entity_id'], df_s1['country'], df_s1['core_tokens']):
    if not tokens:
        continue
    # Query using rarest token
    rarest = min(tokens, key=lambda t: token_freq.get(t, 0))
    key = (c, rarest)
    
    cands_s2 = s2_core_idx.get(key, [])[:200]
    cands_s3 = s3_core_idx.get(key, [])[:200]
    
    num_c = len(cands_s2) + len(cands_s3)
    if num_c == 0:
        continue
        
    total_candidates_core += num_c
    
    gt_set = gt_pairs_dict.get(s1_id)
    if gt_set:
        cand_set = set(cands_s2).union(cands_s3)
        retrieved_gt_core += len(gt_set.intersection(cand_set))

recall_core = float(retrieved_gt_core / total_gt_pairs)
avg_cand_core = float(total_candidates_core / total_s1)
reduction_core = float(1.0 - (total_candidates_core / max_possible_pairs))

print(f"  Recall: {recall_core*100:.2f}% ({retrieved_gt_core:,} / {total_gt_pairs:,})")
print(f"  Avg Candidates/S1: {avg_cand_core:.2f}")
print(f"  Total Candidates: {total_candidates_core:,}")

res4 = {
    "strategy": "BLOCK-4 (Core Rare Token + Country)",
    "recall_pct": round(recall_core * 100, 2),
    "retrieved_gt": retrieved_gt_core,
    "avg_candidates_per_s1": round(avg_cand_core, 2),
    "total_candidates": total_candidates_core,
    "reduction_ratio_pct": round(reduction_core * 100, 8)
}
results.append(res4)

# ---------------------------------------------------------
# 6. Union Strategy Evaluation (UNION-BLOCK)
# ---------------------------------------------------------
print("\n--- Evaluating Strategy: UNION-BLOCK (Multi-Channel Combination) ---")
t0 = time.time()

# Re-build indexes for fast union lookup
idx0_s2 = defaultdict(list)
idx0_s3 = defaultdict(list)
for eid, c, k in zip(df_s2['entity_id'], df_s2['country'], df_s2['norm_name']):
    if k: idx0_s2[(c, k)].append(eid)
for eid, c, k in zip(df_s3['entity_id'], df_s3['country'], df_s3['norm_name']):
    if k: idx0_s3[(c, k)].append(eid)

idx1_s2 = defaultdict(list)
idx1_s3 = defaultdict(list)
for eid, c, k in zip(df_s2['entity_id'], df_s2['country'], df_s2['sorted_name']):
    if k: idx1_s2[(c, k)].append(eid)
for eid, c, k in zip(df_s3['entity_id'], df_s3['country'], df_s3['sorted_name']):
    if k: idx1_s3[(c, k)].append(eid)

idx2_s2 = defaultdict(list)
idx2_s3 = defaultdict(list)
for eid, c, k in zip(df_s2['entity_id'], df_s2['country'], df_s2['first2_name']):
    if k: idx2_s2[(c, k)].append(eid)
for eid, c, k in zip(df_s3['entity_id'], df_s3['country'], df_s3['first2_name']):
    if k: idx2_s3[(c, k)].append(eid)

idx3_s2 = defaultdict(list)
idx3_s3 = defaultdict(list)
for eid, c, k in zip(df_s2['entity_id'], df_s2['country'], df_s2['addr_num_street']):
    if k: idx3_s2[(c, k)].append(eid)
for eid, c, k in zip(df_s3['entity_id'], df_s3['country'], df_s3['addr_num_street']):
    if k: idx3_s3[(c, k)].append(eid)

print(f"  All channel indexes built in {time.time() - t0:.2f}s")
t0 = time.time()

retrieved_gt_union = 0
total_candidates_union = 0

recovered_examples = []

for s1_id, c, norm_n, sort_n, f2_n, addr_k, tokens in zip(
    df_s1['entity_id'], df_s1['country'], df_s1['norm_name'], 
    df_s1['sorted_name'], df_s1['first2_name'], df_s1['addr_num_street'], df_s1['core_tokens']
):
    cand_set = set()
    
    # Channel 0: Exact Name
    if norm_n:
        key = (c, norm_n)
        cand_set.update(idx0_s2.get(key, [])[:100])
        cand_set.update(idx0_s3.get(key, [])[:100])
        
    # Channel 1: Token Sorted Name
    if sort_n:
        key = (c, sort_n)
        cand_set.update(idx1_s2.get(key, [])[:100])
        cand_set.update(idx1_s3.get(key, [])[:100])
        
    # Channel 2: First 2 Significant Tokens
    if f2_n:
        key = (c, f2_n)
        cand_set.update(idx2_s2.get(key, [])[:50])
        cand_set.update(idx2_s3.get(key, [])[:50])

    # Channel 3: Address Num + Street Token
    if addr_k:
        key = (c, addr_k)
        cand_set.update(idx3_s2.get(key, [])[:50])
        cand_set.update(idx3_s3.get(key, [])[:50])
        
    # Channel 4: Core Rare Token
    if tokens:
        rarest = min(tokens, key=lambda t: token_freq.get(t, 0))
        if token_freq.get(rarest, 0) <= 5000:
            key = (c, rarest)
            cand_set.update(s2_core_idx.get(key, [])[:50])
            cand_set.update(s3_core_idx.get(key, [])[:50])
            
    num_c = len(cand_set)
    total_candidates_union += num_c
    
    gt_set = gt_pairs_dict.get(s1_id)
    if gt_set:
        hits = gt_set.intersection(cand_set)
        retrieved_gt_union += len(hits)
        
        # Check if baseline missed any hit that Union recovered!
        b0_cands = set(idx0_s2.get((c, norm_n), [])) | set(idx0_s3.get((c, norm_n), []))
        b0_hits = gt_set.intersection(b0_cands)
        new_hits = hits - b0_hits
        if new_hits and len(recovered_examples) < 20:
            recovered_examples.append({
                "s1_id": s1_id,
                "country": c,
                "s1_name": df_s1.loc[df_s1['entity_id']==s1_id, 'business_name'].values[0],
                "s1_address": df_s1.loc[df_s1['entity_id']==s1_id, 'business_address'].values[0],
                "recovered_hits": list(new_hits)
            })

query_time = time.time() - t0
recall_union = float(retrieved_gt_union / total_gt_pairs)
avg_cand_union = float(total_candidates_union / total_s1)
reduction_union = float(1.0 - (total_candidates_union / max_possible_pairs))

print(f"  Union query evaluated in {query_time:.2f}s")
print(f"  Recall: {recall_union*100:.2f}% ({retrieved_gt_union:,} / {total_gt_pairs:,})")
print(f"  Avg Candidates/S1: {avg_cand_union:.2f}")
print(f"  Total Candidates: {total_candidates_union:,}")

res_union = {
    "strategy": "UNION-BLOCK (Combined Multi-Channel)",
    "recall_pct": round(recall_union * 100, 2),
    "retrieved_gt": retrieved_gt_union,
    "avg_candidates_per_s1": round(avg_cand_union, 2),
    "total_candidates": total_candidates_union,
    "reduction_ratio_pct": round(reduction_union * 100, 8)
}
results.append(res_union)

# ---------------------------------------------------------
# Save Results
# ---------------------------------------------------------
with open(os.path.join(OUTPUT_DIR, "phase2_blocking_summary.json"), "w") as f:
    json.dump({
        "results": results,
        "recovered_examples": recovered_examples
    }, f, indent=2)

print("\n--- SUMMARY TABLE OF BLOCKING STRATEGIES ---")
print(f"{'Strategy':<42} | {'Recall %':<10} | {'Avg Cand/S1':<12} | {'Total Cand':<14} | {'Reduction %':<14}")
print("-" * 100)
for r in results:
    print(f"{r['strategy']:<42} | {r['recall_pct']:<10.2f} | {r['avg_candidates_per_s1']:<12.2f} | {r['total_candidates']:<14,} | {r['reduction_ratio_pct']:<14.6f}")

