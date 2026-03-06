"""
SCRIPT 1: PREPARE ARMD DATA
============================
Loads the real Stanford ARMD dataset, merges files, engineers features,
and creates train/test/forget/retain splits for machine unlearning.

REQUIRED FILES in armd_data/ folder:
  - microbiology_cultures_cohort.csv
  - microbiology_cultures_demographics.csv
  - microbiology_cultures_ward_info.csv

HOW TO RUN:
  python script1_prepare_data.py
"""

import pandas as pd
import numpy as np
import os

# Reproducibility
np.random.seed(42)

# Create output folder
os.makedirs("data", exist_ok=True)

print("=" * 60)
print("STEP 1: Loading ARMD dataset files...")
print("=" * 60)

# ── Load cohort.csv (core file: organism, antibiotic, susceptibility) ─────────
print("\nLoading cohort file...")
cohort = pd.read_csv("armd_data/microbiology_cultures_cohort.csv", low_memory=False)
print(f"  {len(cohort):,} rows | Columns: {list(cohort.columns)}")

# ── Load demographics.csv (age bins, gender) ──────────────────────────────────
print("Loading demographics file...")
demographics = pd.read_csv("armd_data/microbiology_cultures_demographics.csv", low_memory=False)
print(f"  {len(demographics):,} rows | Columns: {list(demographics.columns)}")

# ── Load ward_info.csv (ICU, ER, inpatient, outpatient flags) ────────────────
print("Loading ward info file...")
ward = pd.read_csv("armd_data/microbiology_cultures_ward_info.csv", low_memory=False)
print(f"  {len(ward):,} rows | Columns: {list(ward.columns)}")

print("\n" + "=" * 60)
print("STEP 2: Cleaning susceptibility labels...")
print("=" * 60)

# Show what susceptibility values exist
print(f"\nAll susceptibility values:\n{cohort['susceptibility'].value_counts().to_string()}")

# Keep only Resistant / Susceptible rows (drop Null, Inconclusive, etc.)
cohort_clean = cohort[cohort["susceptibility"].isin(["Resistant", "Susceptible"])].copy()
print(f"\nKept {len(cohort_clean):,} rows (Resistant or Susceptible only)")

# Create binary target: 1 = Resistant, 0 = Susceptible
cohort_clean["target"] = (cohort_clean["susceptibility"] == "Resistant").astype(int)
print(f"Resistant: {cohort_clean['target'].sum():,} ({cohort_clean['target'].mean():.1%})")
print(f"Susceptible: {(cohort_clean['target']==0).sum():,}")

# Drop rows missing organism or antibiotic name
cohort_clean = cohort_clean.dropna(subset=["organism", "antibiotic"])
print(f"After dropping missing organism/antibiotic: {len(cohort_clean):,} rows")

print("\n" + "=" * 60)
print("STEP 3: Merging demographics and ward info...")
print("=" * 60)

# Find the best linking key shared across files
possible_keys = ["order_proc_id_coded", "pat_enc_csn_id_coded", "anon_id"]
cohort_keys = [k for k in possible_keys if k in cohort_clean.columns]
demo_keys   = [k for k in possible_keys if k in demographics.columns]
ward_keys   = [k for k in possible_keys if k in ward.columns]

# Pick the most granular key available in all three files
merge_key = None
for k in possible_keys:
    if k in cohort_keys and k in demo_keys and k in ward_keys:
        merge_key = k
        break
if merge_key is None:
    # Fall back to best shared key between cohort + demographics
    for k in possible_keys:
        if k in cohort_keys and k in demo_keys:
            merge_key = k
            break
if merge_key is None:
    merge_key = cohort_keys[0]  # last resort

print(f"Linking key: '{merge_key}'")

# Merge demographics (take one row per encounter to avoid duplication)
if merge_key in demographics.columns:
    demo_cols = [merge_key]
    if "age"    in demographics.columns: demo_cols.append("age")
    if "gender" in demographics.columns: demo_cols.append("gender")
    demo_small = demographics[demo_cols].drop_duplicates(subset=[merge_key])
    merged = cohort_clean.merge(demo_small, on=merge_key, how="left")
    print(f"After demographics merge: {len(merged):,} rows")
else:
    merged = cohort_clean.copy()
    print("Skipped demographics (key not found)")

# Merge ward info
if merge_key in ward.columns:
    ward_need = [merge_key] + [c for c in ["hosp_ward_IP","hosp_ward_OP","hosp_ward_ER","hosp_ward_ICU"]
                                if c in ward.columns]
    ward_small = ward[ward_need].drop_duplicates(subset=[merge_key])
    merged = merged.merge(ward_small, on=merge_key, how="left")
    print(f"After ward info merge:    {len(merged):,} rows")
else:
    print("Skipped ward info (key not found)")

print("\n" + "=" * 60)
print("STEP 4: Feature engineering...")
print("=" * 60)

# ── Organism: keep top 20, group rest as "Other" ──────────────────────────────
top_organisms = merged["organism"].value_counts().head(20).index.tolist()
merged["org_clean"] = merged["organism"].apply(lambda x: x if x in top_organisms else "Other")
org_dummies = pd.get_dummies(merged["org_clean"], prefix="org")
print(f"Organism features: {org_dummies.shape[1]}")

# ── Antibiotic: keep top 30, group rest as "Other" ────────────────────────────
top_abx = merged["antibiotic"].value_counts().head(30).index.tolist()
merged["abx_clean"] = merged["antibiotic"].apply(lambda x: x if x in top_abx else "Other")
abx_dummies = pd.get_dummies(merged["abx_clean"], prefix="abx")
print(f"Antibiotic features: {abx_dummies.shape[1]}")

# ── Culture type: urine / blood / respiratory ─────────────────────────────────
if "culture_description" in merged.columns:
    top_cultures = merged["culture_description"].value_counts().head(5).index.tolist()
    merged["culture_clean"] = merged["culture_description"].apply(
        lambda x: x if x in top_cultures else "Other")
    culture_dummies = pd.get_dummies(merged["culture_clean"], prefix="culture")
    print(f"Culture type features: {culture_dummies.shape[1]}")
else:
    culture_dummies = pd.DataFrame(index=merged.index)

# ── Age bins → ordinal numbers ────────────────────────────────────────────────
age_map = {"18-24":1,"25-34":2,"35-44":3,"45-54":4,"55-64":5,"65-74":6,"75-84":7,"85-89":8,"90+":9}
merged["age_num"] = merged["age"].map(age_map).fillna(4) if "age" in merged.columns else 4

# ── Gender (already 0/1 in ARMD) ─────────────────────────────────────────────
merged["gender_num"] = merged["gender"].fillna(0) if "gender" in merged.columns else 0

# ── Ward type (binary flags) ──────────────────────────────────────────────────
ward_cols = [c for c in ["hosp_ward_IP","hosp_ward_OP","hosp_ward_ER","hosp_ward_ICU"]
             if c in merged.columns]
for c in ward_cols:
    merged[c] = merged[c].fillna(0).astype(int)
print(f"Ward features: {ward_cols}")

# ── Build final feature matrix ────────────────────────────────────────────────
base_cols = ["age_num", "gender_num"] + ward_cols
feature_df = pd.concat([
    merged[base_cols].reset_index(drop=True),
    org_dummies.reset_index(drop=True),
    abx_dummies.reset_index(drop=True),
    culture_dummies.reset_index(drop=True),
], axis=1)

feature_df["target"]  = merged["target"].values
feature_df["anon_id"] = merged["anon_id"].values if "anon_id" in merged.columns else np.arange(len(merged))

feature_cols = [c for c in feature_df.columns if c not in ["target", "anon_id"]]
with open("data/feature_columns.txt", "w") as f:
    f.write("\n".join(feature_cols))

print(f"\nTotal features: {len(feature_cols)}")
print(f"Total samples:  {len(feature_df):,}")

print("\n" + "=" * 60)
print("STEP 5: Sampling, splitting, and saving...")
print("=" * 60)

# Sample up to 200k rows so scripts run fast on a laptop
MAX_ROWS = 200_000
if len(feature_df) > MAX_ROWS:
    feature_df = feature_df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)
    print(f"Sampled {MAX_ROWS:,} rows for speed (full dataset = {len(merged):,})")

# 80/20 train-test split, stratified by target
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(
    feature_df, test_size=0.2, random_state=42, stratify=feature_df["target"])

# Forget set = 500 random training patients requesting deletion
N_FORGET = 500
forget_set = train_df.sample(n=N_FORGET, random_state=42)
retain_set = train_df.drop(forget_set.index)

# Save
train_df.to_csv("data/train_data.csv",   index=False)
test_df.to_csv( "data/test_data.csv",    index=False)
forget_set.to_csv("data/forget_set.csv", index=False)
retain_set.to_csv("data/retain_set.csv", index=False)

print(f"\n  Train:      {len(train_df):,} rows  → data/train_data.csv")
print(f"  Test:       {len(test_df):,} rows  → data/test_data.csv")
print(f"  Forget set: {len(forget_set):,} rows  → data/forget_set.csv")
print(f"  Retain set: {len(retain_set):,} rows  → data/retain_set.csv")
print(f"  Features:   {len(feature_cols)}         → data/feature_columns.txt")

print("\n" + "=" * 60)
print("✓  SCRIPT 1 DONE — Run script2_train_model.py next")
print("=" * 60)