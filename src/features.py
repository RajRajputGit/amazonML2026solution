"""
Phase 3: Pair Feature Engineering & Representative Dataset Generator
Amazon ML Challenge 2026 - Business Entity Resolution

This module handles:
1. Hard Negative Sampling per S1 entity
2. Multi-field Pair Feature Extraction:
   - Name Similarity (Jaccard, Char 3-gram Jaccard, Sequence Similarity, Token set overlap, Length ratio, First char match, Prefix-3 match)
   - Address Similarity (Token overlap, Sequence similarity, Building number exact match, Numeric token Jaccard, Address length features)
   - Country Match
   - Missingness & Length Features
   - Blocking Provenance Channels (exact, sorted, first2, address, rare_token, channel_count)
   - Source Type (S2 vs S3)
3. Leakage-safe Train/Validation Split at the S1 entity level.
"""

import numpy as np
import pandas as pd
import re
import unicodedata
import os
import json
import time
import difflib

# ---------------------------------------------------------
# Feature Helper Functions
# ---------------------------------------------------------
COMMON_STOPWORDS = {'and', 'the', 'of', 'for', 'in', 'on', 'at', 'to', 'a', 'an', 'is', 'by', 'with'}
NON_ALPHANUM_RE = re.compile(r'[^a-z0-9\s]')
SPACE_RE = re.compile(r'\s+')
DIGITS_RE = re.compile(r'\b\d+\b')
LEGAL_SUFFIX_RE = re.compile(r'\b(inc|incorporated|corp|corporation|ltd|limited|llc|co|company|pvt|private|pvt ltd|private limited|plc|gmbh|sa|bv|srl|llp|holdings|group|enterprises|enterprise|services|solutions|tech|technologies|international|intl)\b', re.IGNORECASE)

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

def get_char_ngrams(text, n=3):
    if not text or len(text) < n:
        return set([text]) if text else set()
    return set(text[i:i+n] for i in range(len(text) - n + 1))

def jaccard_sets(set1, set2):
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return float(intersection / union) if union > 0 else 0.0

def overlap_ratio_min(set1, set2):
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    min_len = min(len(set1), len(set2))
    return float(intersection / min_len) if min_len > 0 else 0.0

def extract_building_numbers(text):
    if not text:
        return []
    return DIGITS_RE.findall(str(text))

ngram_cache = {}
def cached_char_ngrams(text, n=3):
    if text not in ngram_cache:
        ngram_cache[text] = get_char_ngrams(text, n)
    return ngram_cache[text]

num_cache = {}
def cached_building_numbers(text):
    if text not in num_cache:
        num_cache[text] = extract_building_numbers(text)
    return num_cache[text]

def seq_similarity(str1, str2):
    if not str1 or not str2:
        return 0.0
    if str1 == str2:
        return 1.0
    l1, l2 = len(str1), len(str2)
    if min(l1, l2) / max(l1, l2) < 0.2:
        return 0.0
    return float(difflib.SequenceMatcher(None, str1, str2).quick_ratio())

def compute_pair_features(s1_rec, cand_rec, provenance_flags=None):
    """
    Computes a vector of 27 numerical features for a single (S1, Candidate) pair.
    """
    s1_name_raw = s1_rec.get('business_name', '') or ''
    cand_name_raw = cand_rec.get('business_name', '') or ''
    s1_addr_raw = s1_rec.get('business_address', '') or ''
    cand_addr_raw = cand_rec.get('business_address', '') or ''
    s1_country = s1_rec.get('country', '') or ''
    cand_country = cand_rec.get('country', '') or ''
    cand_id = cand_rec.get('entity_id', '') or ''

    # Normalized text
    s1_name_norm = s1_rec.get('norm_name', '') or normalize_name(s1_name_raw)
    cand_name_norm = cand_rec.get('norm_name', '') or normalize_name(cand_name_raw)
    s1_addr_norm = s1_rec.get('norm_addr', '') or normalize_text(s1_addr_raw)
    cand_addr_norm = cand_rec.get('norm_addr', '') or normalize_text(cand_addr_raw)

    # --- 1. Country & Source Features ---
    country_match = 1.0 if (s1_country and cand_country and s1_country == cand_country) else 0.0
    is_s3 = 1.0 if cand_id.startswith('S3-') else 0.0

    # --- 2. Name Features ---
    s1_name_tokens = set(s1_name_norm.split())
    cand_name_tokens = set(cand_name_norm.split())

    name_jaccard = jaccard_sets(s1_name_tokens, cand_name_tokens)
    name_overlap_min = overlap_ratio_min(s1_name_tokens, cand_name_tokens)
    name_token_intersection = float(len(s1_name_tokens.intersection(cand_name_tokens)))

    # 3-gram Jaccard
    s1_ngrams = cached_char_ngrams(s1_name_norm, 3)
    cand_ngrams = cached_char_ngrams(cand_name_norm, 3)
    name_3gram_jaccard = jaccard_sets(s1_ngrams, cand_ngrams)

    # Sequence Similarity with Pruning
    if name_3gram_jaccard == 0.0 and name_jaccard == 0.0:
        name_seq_ratio = 0.0
    else:
        name_seq_ratio = seq_similarity(s1_name_norm, cand_name_norm)

    # First char & prefix match
    first_char_match = 1.0 if (s1_name_norm and cand_name_norm and s1_name_norm[0] == cand_name_norm[0]) else 0.0
    prefix_3_match = 1.0 if (len(s1_name_norm) >= 3 and len(cand_name_norm) >= 3 and s1_name_norm[:3] == cand_name_norm[:3]) else 0.0

    # Name Length features
    l1 = len(s1_name_norm)
    l2 = len(cand_name_norm)
    name_len_diff = float(abs(l1 - l2))
    name_len_ratio = float(min(l1, l2) / max(l1, l2)) if max(l1, l2) > 0 else 0.0

    # --- 3. Address Features ---
    s1_has_addr = 1.0 if (s1_addr_norm and s1_addr_norm != 'none') else 0.0
    cand_has_addr = 1.0 if (cand_addr_norm and cand_addr_norm != 'none') else 0.0
    both_has_addr = 1.0 if (s1_has_addr == 1.0 and cand_has_addr == 1.0) else 0.0

    if both_has_addr == 1.0:
        s1_addr_tokens = set(s1_addr_norm.split())
        cand_addr_tokens = set(cand_addr_norm.split())

        addr_jaccard = jaccard_sets(s1_addr_tokens, cand_addr_tokens)
        addr_overlap_min = overlap_ratio_min(s1_addr_tokens, cand_addr_tokens)
        
        if addr_jaccard == 0.0:
            addr_seq_ratio = 0.0
        else:
            addr_seq_ratio = seq_similarity(s1_addr_norm, cand_addr_norm)

        # Building number matching
        s1_nums = cached_building_numbers(s1_addr_raw)
        cand_nums = cached_building_numbers(cand_addr_raw)

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

    # --- 4. Provenance Features ---
    if provenance_flags is None:
        provenance_flags = {}
    p_exact = float(provenance_flags.get('exact_name', 0))
    p_sort = float(provenance_flags.get('sorted_name', 0))
    p_f2 = float(provenance_flags.get('first2_tokens', 0))
    p_addr = float(provenance_flags.get('address_key', 0))
    p_rare = float(provenance_flags.get('rare_token', 0))
    p_channel_count = p_exact + p_sort + p_f2 + p_addr + p_rare

    return [
        country_match,
        is_s3,
        name_jaccard,
        name_overlap_min,
        name_token_intersection,
        name_3gram_jaccard,
        name_seq_ratio,
        first_char_match,
        prefix_3_match,
        name_len_diff,
        name_len_ratio,
        s1_has_addr,
        cand_has_addr,
        both_has_addr,
        addr_jaccard,
        addr_overlap_min,
        addr_seq_ratio,
        num_exact_match,
        num_jaccard,
        addr_len_diff,
        addr_len_ratio,
        p_exact,
        p_sort,
        p_f2,
        p_addr,
        p_rare,
        p_channel_count
    ]

FEATURE_NAMES = [
    'country_match',
    'is_s3',
    'name_jaccard',
    'name_overlap_min',
    'name_token_intersection',
    'name_3gram_jaccard',
    'name_seq_ratio',
    'first_char_match',
    'prefix_3_match',
    'name_len_diff',
    'name_len_ratio',
    's1_has_addr',
    'cand_has_addr',
    'both_has_addr',
    'addr_jaccard',
    'addr_overlap_min',
    'addr_seq_ratio',
    'num_exact_match',
    'num_jaccard',
    'addr_len_diff',
    'addr_len_ratio',
    'p_exact',
    'p_sort',
    'p_f2',
    'p_addr',
    'p_rare',
    'p_channel_count'
]

