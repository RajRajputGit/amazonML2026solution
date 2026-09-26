"""
Phase 2: Multi-Strategy Candidate Generation Framework
Amazon ML Challenge 2026 - Business Entity Resolution

This module implements multi-channel blocking strategies:
1. BLOCK-0: Exact Normalized Business Name + Country
2. BLOCK-1: Token-Sorted Business Name + Country
3. BLOCK-2: First 2 Significant Tokens + Country
4. BLOCK-3: Address Number + Street Token + Country
5. BLOCK-4: Rare Core Token + Country
6. UNION-BLOCK: High-Recall Multi-Channel Combination with Bucket Caps
"""

import pandas as pd
import numpy as np
import pyarrow.csv as parse_csv
import re
import unicodedata
import os
import json
import time
from collections import Counter, defaultdict

COMMON_STOPWORDS = {'and', 'the', 'of', 'for', 'in', 'on', 'at', 'to', 'a', 'an', 'is', 'by', 'with'}
LEGAL_SUFFIX_RE = re.compile(r'\b(inc|incorporated|corp|corporation|ltd|limited|llc|co|company|pvt|private|pvt ltd|private limited|plc|gmbh|sa|bv|srl|llp|holdings|group|enterprises|enterprise|services|solutions|tech|technologies|international|intl)\b', re.IGNORECASE)
NON_ALPHANUM_RE = re.compile(r'[^a-z0-9\s]')
SPACE_RE = re.compile(r'\s+')

def load_tsv(filepath):
    read_options = parse_csv.ReadOptions(block_size=1048576*16)
    parse_options = parse_csv.ParseOptions(delimiter='\t', quote_char=False)
    table = parse_csv.read_csv(filepath, read_options=read_options, parse_options=parse_options)
    return table.to_pandas().astype(str)

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

def extract_core_tokens(name):
    norm = normalize_name(name)
    if not norm:
        return []
    tokens = [t for t in norm.split() if len(t) >= 4 and t not in COMMON_STOPWORDS]
    return tokens

class MultiStrategyBlocker:
    def __init__(self, channel_caps=None):
        if channel_caps is None:
            self.channel_caps = {
                'exact_name': 100,
                'sorted_name': 100,
                'first2_tokens': 50,
                'address_key': 50,
                'rare_token': 50
            }
        else:
            self.channel_caps = channel_caps
            
    def generate_candidates(self, df_s1, df_s2, df_s3):
        t0 = time.time()
        print("[MultiStrategyBlocker] Preprocessing and extracting keys...")
        
        token_freq = Counter()
        s2_core_list = [extract_core_tokens(x) for x in df_s2['business_name']]
        s3_core_list = [extract_core_tokens(x) for x in df_s3['business_name']]
        
        for tokens in s2_core_list:
            for t in set(tokens): token_freq[t] += 1
        for tokens in s3_core_list:
            for t in set(tokens): token_freq[t] += 1

        idx_exact_s2 = defaultdict(list)
        idx_exact_s3 = defaultdict(list)
        idx_sort_s2 = defaultdict(list)
        idx_sort_s3 = defaultdict(list)
        idx_f2_s2 = defaultdict(list)
        idx_f2_s3 = defaultdict(list)
        idx_addr_s2 = defaultdict(list)
        idx_addr_s3 = defaultdict(list)
        idx_rare_s2 = defaultdict(list)
        idx_rare_s3 = defaultdict(list)

        print("[MultiStrategyBlocker] Building S2 inverted indexes...")
        for eid, c, name, addr, core_tokens in zip(
            df_s2['entity_id'], df_s2['country'], df_s2['business_name'], df_s2['business_address'], s2_core_list
        ):
            n_norm = normalize_name(name)
            n_sort = get_sorted_token_key(name)
            n_f2 = get_first_two_sig_tokens(name)
            a_key = get_address_num_street_key(addr)
            
            if n_norm: idx_exact_s2[(c, n_norm)].append(eid)
            if n_sort: idx_sort_s2[(c, n_sort)].append(eid)
            if n_f2: idx_f2_s2[(c, n_f2)].append(eid)
            if a_key: idx_addr_s2[(c, a_key)].append(eid)
            if core_tokens:
                rarest = min(core_tokens, key=lambda t: token_freq[t])
                if token_freq[rarest] <= 5000:
                    idx_rare_s2[(c, rarest)].append(eid)

        print("[MultiStrategyBlocker] Building S3 inverted indexes...")
        for eid, c, name, addr, core_tokens in zip(
            df_s3['entity_id'], df_s3['country'], df_s3['business_name'], df_s3['business_address'], s3_core_list
        ):
            n_norm = normalize_name(name)
            n_sort = get_sorted_token_key(name)
            n_f2 = get_first_two_sig_tokens(name)
            a_key = get_address_num_street_key(addr)
            
            if n_norm: idx_exact_s3[(c, n_norm)].append(eid)
            if n_sort: idx_sort_s3[(c, n_sort)].append(eid)
            if n_f2: idx_f2_s3[(c, n_f2)].append(eid)
            if a_key: idx_addr_s3[(c, a_key)].append(eid)
            if core_tokens:
                rarest = min(core_tokens, key=lambda t: token_freq[t])
                if token_freq[rarest] <= 5000:
                    idx_rare_s3[(c, rarest)].append(eid)

        print(f"[MultiStrategyBlocker] Indexes built in {time.time() - t0:.2f}s")
        t0 = time.time()
        print("[MultiStrategyBlocker] Querying candidate sets for S1 entities...")

        s1_candidate_dict = {}
        
        cap_exact = self.channel_caps['exact_name']
        cap_sort = self.channel_caps['sorted_name']
        cap_f2 = self.channel_caps['first2_tokens']
        cap_addr = self.channel_caps['address_key']
        cap_rare = self.channel_caps['rare_token']

        for s1_id, c, name, addr in zip(
            df_s1['entity_id'], df_s1['country'], df_s1['business_name'], df_s1['business_address']
        ):
            cand_set = set()
            n_norm = normalize_name(name)
            n_sort = get_sorted_token_key(name)
            n_f2 = get_first_two_sig_tokens(name)
            a_key = get_address_num_street_key(addr)
            core_tokens = extract_core_tokens(name)

            if n_norm:
                key = (c, n_norm)
                cand_set.update(idx_exact_s2.get(key, [])[:cap_exact])
                cand_set.update(idx_exact_s3.get(key, [])[:cap_exact])
                
            if n_sort:
                key = (c, n_sort)
                cand_set.update(idx_sort_s2.get(key, [])[:cap_sort])
                cand_set.update(idx_sort_s3.get(key, [])[:cap_sort])
                
            if n_f2:
                key = (c, n_f2)
                cand_set.update(idx_f2_s2.get(key, [])[:cap_f2])
                cand_set.update(idx_f2_s3.get(key, [])[:cap_f2])

            if a_key:
                key = (c, a_key)
                cand_set.update(idx_addr_s2.get(key, [])[:cap_addr])
                cand_set.update(idx_addr_s3.get(key, [])[:cap_addr])

            if core_tokens:
                rarest = min(core_tokens, key=lambda t: token_freq.get(t, 0))
                if token_freq.get(rarest, 0) <= 5000:
                    key = (c, rarest)
                    cand_set.update(idx_rare_s2.get(key, [])[:cap_rare])
                    cand_set.update(idx_rare_s3.get(key, [])[:cap_rare])

            s1_candidate_dict[s1_id] = list(cand_set)

        print(f"[MultiStrategyBlocker] Candidate generation finished in {time.time() - t0:.2f}s")
        return s1_candidate_dict

if __name__ == "__main__":
    print("Running MultiStrategyBlocker validation test...")
    DATA_DIR = "/Users/apple/Desktop/amazonMLsolution/student_resource/dataset/train"
    df_s1 = load_tsv(os.path.join(DATA_DIR, "train_source1.tsv"))
    df_s2 = load_tsv(os.path.join(DATA_DIR, "train_source2.tsv"))
    df_s3 = load_tsv(os.path.join(DATA_DIR, "train_source3.tsv"))
    df_gt = load_tsv(os.path.join(DATA_DIR, "train_ground_truth.tsv"))

    df_gt['matched_entity_ids'] = df_gt['matched_entity_ids'].replace({'nan': '', 'None': '', '<NA>': ''}).fillna('')
    gt_pairs_dict = {}
    total_gt = 0
    for s1_id, m_str in zip(df_gt['source1_entity_id'], df_gt['matched_entity_ids']):
        if m_str:
            m_set = set(x for x in m_str.split(',') if x)
            gt_pairs_dict[s1_id] = m_set
            total_gt += len(m_set)

    blocker = MultiStrategyBlocker()
    cand_dict = blocker.generate_candidates(df_s1, df_s2, df_s3)

    retrieved = 0
    total_cands = 0
    for s1_id, cands in cand_dict.items():
        total_cands += len(cands)
        gt_set = gt_pairs_dict.get(s1_id)
        if gt_set:
            retrieved += len(gt_set.intersection(cands))

    recall = retrieved / total_gt
    avg_cands = total_cands / len(df_s1)
    max_poss = len(df_s1) * (len(df_s2) + len(df_s3))
    reduction = 1.0 - (total_cands / max_poss)

    print(f"\nFinal Blocker Recall: {recall*100:.2f}% ({retrieved:,} / {total_gt:,})")
    print(f"Final Avg Candidates/S1: {avg_cands:.2f}")
    print(f"Final Total Candidates: {total_cands:,}")
    print(f"Final Reduction Ratio: {reduction*100:.8f}%")
