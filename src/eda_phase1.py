import pandas as pd
import numpy as np
import re
import unicodedata
import os
import json
from collections import Counter

DATA_DIR = "/Users/apple/Desktop/amazonMLsolution/student_resource/dataset/train"

print("--- Step 1: Loading Data ---")
s1_path = os.path.join(DATA_DIR, "train_source1.tsv")
s2_path = os.path.join(DATA_DIR, "train_source2.tsv")
s3_path = os.path.join(DATA_DIR, "train_source3.tsv")
gt_path = os.path.join(DATA_DIR, "train_ground_truth.tsv")

df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str)
df_s2 = pd.read_csv(s2_path, sep="\t", dtype=str)
df_s3 = pd.read_csv(s3_path, sep="\t", dtype=str)
df_gt = pd.read_csv(gt_path, sep="\t", dtype=str)

print(f"Source 1 shape: {df_s1.shape}")
print(f"Source 2 shape: {df_s2.shape}")
print(f"Source 3 shape: {df_s3.shape}")
print(f"Ground Truth shape: {df_gt.shape}")

print("\n--- Step 2: Data Profiling ---")

def profile_df(df, name):
    print(f"\n--- Profile for {name} ---")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Unique entity_ids: {df['entity_id'].nunique()} / {len(df)}")
    print(f"Unique business_names: {df['business_name'].nunique()} / {len(df)}")
    print(f"Unique business_addresses: {df['business_address'].nunique()} / {len(df)}")
    print("Country distribution:")
    country_counts = df['country'].value_counts(dropna=False)
    for c, cnt in country_counts.items():
        print(f"  {c}: {cnt} ({cnt/len(df)*100:.2f}%)")
    
    # Text statistics
    name_len = df['business_name'].fillna('').apply(len)
    name_words = df['business_name'].fillna('').apply(lambda x: len(x.split()))
    addr_len = df['business_address'].fillna('').apply(len)
    addr_words = df['business_address'].fillna('').apply(lambda x: len(x.split()))
    
    print(f"Business Name Char Length: mean={name_len.mean():.1f}, median={name_len.median()}, min={name_len.min()}, max={name_len.max()}, p95={name_len.quantile(0.95):.1f}")
    print(f"Business Name Word Count: mean={name_words.mean():.1f}, median={name_words.median()}, min={name_words.min()}, max={name_words.max()}, p95={name_words.quantile(0.95):.1f}")
    print(f"Business Address Char Length: mean={addr_len.mean():.1f}, median={addr_len.median()}, min={addr_len.min()}, max={addr_len.max()}, p95={addr_len.quantile(0.95):.1f}")
    print(f"Business Address Word Count: mean={addr_words.mean():.1f}, median={addr_words.median()}, min={addr_words.min()}, max={addr_words.max()}, p95={addr_words.quantile(0.95):.1f}")

profile_df(df_s1, "Source 1 (S1)")
profile_df(df_s2, "Source 2 (S2)")
profile_df(df_s3, "Source 3 (S3)")

print("\n--- Step 3: Ground Truth Analysis ---")
df_gt['matched_entity_ids'] = df_gt['matched_entity_ids'].fillna('')

def parse_matches(matched_str):
    if not matched_str or pd.isna(matched_str) or matched_str.strip() == '':
        return []
    return [m.strip() for m in matched_str.split(',') if m.strip()]

df_gt['match_list'] = df_gt['matched_entity_ids'].apply(parse_matches)
df_gt['match_count'] = df_gt['match_list'].apply(len)

total_s1_gt = len(df_gt)
singletons = (df_gt['match_count'] == 0).sum()
matched_s1 = (df_gt['match_count'] > 0).sum()

print(f"Total Ground Truth S1 records: {total_s1_gt}")
print(f"Singletons (0 matches): {singletons} ({singletons/total_s1_gt*100:.2f}%)")
print(f"Matched S1 records (>0 matches): {matched_s1} ({matched_s1/total_s1_gt*100:.2f}%)")

# Categorize matches: S2 only, S3 only, S2+S3
def match_source_types(match_list):
    has_s2 = any(m.startswith('S2-') for m in match_list)
    has_s3 = any(m.startswith('S3-') for m in match_list)
    if has_s2 and has_s3:
        return 'S2+S3'
    elif has_s2:
        return 'S2-only'
    elif has_s3:
        return 'S3-only'
    else:
        return 'None'

df_gt['match_type'] = df_gt['match_list'].apply(match_source_types)

print("\nMatch Type Breakdown (among all S1):")
type_counts = df_gt['match_type'].value_counts()
for t, cnt in type_counts.items():
    print(f"  {t}: {cnt} ({cnt/total_s1_gt*100:.2f}%)")

print("\nMatch Count Distribution:")
match_dist = df_gt['match_count'].value_counts().sort_index()
for mc, cnt in match_dist.items():
    print(f"  {mc} matches: {cnt} ({cnt/total_s1_gt*100:.2f}%)")

# Total pairs
all_pairs = []
for idx, row in df_gt.iterrows():
    s1_id = row['source1_entity_id']
    for m in row['match_list']:
        all_pairs.append((s1_id, m))

print(f"\nTotal Ground Truth Pair Connections: {len(all_pairs)}")
s2_pairs = sum(1 for p in all_pairs if p[1].startswith('S2-'))
s3_pairs = sum(1 for p in all_pairs if p[1].startswith('S3-'))
print(f"  Ground Truth S1-S2 pairs: {s2_pairs}")
print(f"  Ground Truth S1-S3 pairs: {s3_pairs}")

# Save state for further analysis
output_dir = "/Users/apple/Desktop/amazonMLsolution/student_resource/scratch"
os.makedirs(output_dir, exist_ok=True)
df_gt.to_pickle(os.path.join(output_dir, "scratch_df_gt.pkl"))
df_s1.to_pickle(os.path.join(output_dir, "scratch_df_s1.pkl"))
df_s2.to_pickle(os.path.join(output_dir, "scratch_df_s2.pkl"))
df_s3.to_pickle(os.path.join(output_dir, "scratch_df_s3.pkl"))
print("Saved temporary dataframes to pickle.")
