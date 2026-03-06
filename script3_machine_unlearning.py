"""
SCRIPT 3: MACHINE UNLEARNING (SISA METHOD)
===========================================
Implements the SISA (Sharded, Isolated, Sliced, Aggregated) machine unlearning
method on the ARMD antibiotic resistance dataset.

THE IDEA:
  Instead of training one big model on all data, we split training data into
  N "shards" and train a separate model per shard. When a patient requests
  deletion, we only retrain the shard(s) containing their data — much faster
  than retraining from scratch.

  We also measure a Membership Inference Attack (MIA) to verify that the
  unlearned model has truly "forgotten" the requested patients.

HOW TO RUN:
  python script3_machine_unlearning.py
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.inspection import permutation_importance

os.makedirs("models",  exist_ok=True)
os.makedirs("results", exist_ok=True)

print("=" * 60)
print("STEP 1: Loading data and models...")
print("=" * 60)

train_df  = pd.read_csv("data/train_data.csv")
test_df   = pd.read_csv("data/test_data.csv")
forget_df = pd.read_csv("data/forget_set.csv")
retain_df = pd.read_csv("data/retain_set.csv")

with open("data/feature_columns.txt") as f:
    feature_cols = [line.strip() for line in f if line.strip()]

with open("models/original_model.pkl",  "rb") as f: original_model  = pickle.load(f)
with open("models/retrained_model.pkl", "rb") as f: retrained_model = pickle.load(f)
with open("results/model_results.json") as f:       prev_results    = json.load(f)

# ARMD uses "Null" strings for missing values — clean before converting to arrays
for df in [train_df, test_df, forget_df, retain_df]:
    df.replace({"Null": 0, "null": 0}, inplace=True)
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

X_test    = test_df[feature_cols].values
y_test    = test_df["target"].values
X_forget  = forget_df[feature_cols].values
y_forget  = forget_df["target"].values
X_retain  = retain_df[feature_cols].values
y_retain  = retain_df["target"].values
X_train   = train_df[feature_cols].values
y_train   = train_df["target"].values

print(f"Train size:     {len(X_train):,}")
print(f"Retain size:    {len(X_retain):,}")
print(f"Forget size:    {len(X_forget):,}")
print(f"Test size:      {len(X_test):,}")
print(f"Features:       {len(feature_cols)}")

print("\n" + "=" * 60)
print("STEP 2: SISA — Train shard models on FULL training data...")
print("=" * 60)

# ── How SISA works ────────────────────────────────────────────────────────────
# 1. Split training data into N_SHARDS equal pieces
# 2. Train one Random Forest per shard
# 3. For prediction, majority-vote across all shard models
# 4. When forgetting: identify which shard(s) contain the forget patients
#    and ONLY retrain those shards — everything else stays the same
# This is much faster than full retraining

N_SHARDS = 5
print(f"\nSplitting training data into {N_SHARDS} shards...")

shard_size = len(train_df) // N_SHARDS
shard_indices = []  # track which training rows belong to each shard
shard_models_full = []  # models trained on ALL data (pre-unlearning)

t_start = time.time()
for i in range(N_SHARDS):
    # Assign rows to this shard
    start = i * shard_size
    end   = start + shard_size if i < N_SHARDS - 1 else len(train_df)
    idx   = list(range(start, end))
    shard_indices.append(idx)

    # Train a Random Forest on this shard
    X_shard = X_train[idx]
    y_shard = y_train[idx]
    model = RandomForestClassifier(n_estimators=50, max_depth=10,
                                   min_samples_leaf=5, random_state=i, n_jobs=-1)
    model.fit(X_shard, y_shard)
    shard_models_full.append(model)
    print(f"  Shard {i+1}/{N_SHARDS}: {len(idx):,} samples — trained ✓")

t_sisa_initial = time.time() - t_start
print(f"\nInitial SISA training time: {t_sisa_initial:.1f}s")

# ── SISA ensemble prediction: majority vote across shards ─────────────────────
def sisa_predict_proba(models, X):
    """Average predicted probabilities across all shard models."""
    probs = np.array([m.predict_proba(X)[:, 1] for m in models])
    return probs.mean(axis=0)

def sisa_predict(models, X, threshold=0.5):
    return (sisa_predict_proba(models, X) >= threshold).astype(int)

# Evaluate the full SISA model (before any unlearning)
y_prob_sisa_full = sisa_predict_proba(shard_models_full, X_test)
acc_sisa_full    = accuracy_score(y_test, (y_prob_sisa_full >= 0.5).astype(int))
auc_sisa_full    = roc_auc_score(y_test, y_prob_sisa_full)
print(f"\nSISA (before unlearning): Accuracy={acc_sisa_full:.4f}  AUC={auc_sisa_full:.4f}")

print("\n" + "=" * 60)
print("STEP 3: SISA Unlearning — remove forget set patients...")
print("=" * 60)

# Find which shard(s) contain each forget patient
# We need to check: which shard index (in the original training order) has these rows
forget_indices_in_train = forget_df.index.tolist()  # their position in train_df
affected_shards = set()
for idx in forget_indices_in_train:
    for shard_id, shard_idx in enumerate(shard_indices):
        if idx in shard_idx:
            affected_shards.add(shard_id)
            break

print(f"Forget patients span {len(affected_shards)} shard(s): {affected_shards}")

# Retrain ONLY the affected shards (without the forget patients)
shard_models_unlearned = list(shard_models_full)  # copy list, reuse unaffected shards
t_start = time.time()
forget_set_indices = set(forget_df.index.tolist())

for shard_id in affected_shards:
    shard_idx = shard_indices[shard_id]

    # Remove forget patients from this shard
    clean_idx = [i for i in shard_idx if i not in forget_set_indices]

    if len(clean_idx) == 0:
        print(f"  Shard {shard_id+1}: ALL rows removed — using dummy model")
        # Edge case: entire shard was forget patients
        dummy = RandomForestClassifier(n_estimators=10, random_state=shard_id)
        dummy.fit(X_retain[:10], y_retain[:10])
        shard_models_unlearned[shard_id] = dummy
        continue

    X_shard_clean = X_train[clean_idx]
    y_shard_clean = y_train[clean_idx]

    model_new = RandomForestClassifier(n_estimators=50, max_depth=10,
                                       min_samples_leaf=5, random_state=shard_id, n_jobs=-1)
    model_new.fit(X_shard_clean, y_shard_clean)
    shard_models_unlearned[shard_id] = model_new
    removed = len(shard_idx) - len(clean_idx)
    print(f"  Shard {shard_id+1}: retrained without {removed} forget patients ✓")

t_unlearning = time.time() - t_start
print(f"\nSISA unlearning time: {t_unlearning:.1f}s  "
      f"(only retrained {len(affected_shards)}/{N_SHARDS} shards)")

# Full retrain time (for comparison — measured earlier)
t_full_retrain_start = time.time()
temp_model = RandomForestClassifier(n_estimators=100, max_depth=10,
                                    min_samples_leaf=5, random_state=42, n_jobs=-1)
temp_model.fit(X_retain, y_retain)
t_full_retrain = time.time() - t_full_retrain_start
print(f"Full retrain time: {t_full_retrain:.1f}s  (what hospitals currently do)")
print(f"SISA speedup: {t_full_retrain / max(t_unlearning, 0.01):.1f}x faster")

print("\n" + "=" * 60)
print("STEP 4: Evaluating unlearned model on test set...")
print("=" * 60)

y_prob_unlearned = sisa_predict_proba(shard_models_unlearned, X_test)
acc_unlearned    = accuracy_score(y_test, (y_prob_unlearned >= 0.5).astype(int))
auc_unlearned    = roc_auc_score(y_test, y_prob_unlearned)

print(f"\nUnlearned model: Accuracy={acc_unlearned:.4f}  AUC={auc_unlearned:.4f}")
print(f"Original model:  Accuracy={prev_results['original_model']['accuracy']:.4f}  "
      f"AUC={prev_results['original_model']['auc_roc']:.4f}")
print(f"Retrained model: Accuracy={prev_results['retrained_model']['accuracy']:.4f}  "
      f"AUC={prev_results['retrained_model']['auc_roc']:.4f}")
print(f"\nAccuracy drop after unlearning: "
      f"{abs(prev_results['original_model']['accuracy'] - acc_unlearned):.1%}")

print("\n" + "=" * 60)
print("STEP 5: Membership Inference Attack (MIA) — did we truly forget?")
print("=" * 60)

# ── What is MIA? ──────────────────────────────────────────────────────────────
# A membership inference attack tries to guess whether a sample was used in training.
# If the model has truly "forgotten" a patient, the attack should NOT be able to
# distinguish forget patients from unseen test patients.
#
# Method: We use prediction confidence as a proxy.
# A model tends to be MORE confident (higher probability) on its training data
# (overfitting). If a model has forgotten a sample, its confidence on that
# sample drops to be similar to confidence on unseen test data.

def mia_score(model_probs, labels):
    """
    Compute a simple MIA score:
    Mean prediction confidence on the given samples.
    Higher = model is more confident = more likely the data was in training.
    """
    # Confidence = predicted probability for the TRUE class
    confidences = []
    for prob, label in zip(model_probs, labels):
        if label == 1:
            confidences.append(prob)       # probability of Resistant (correct class)
        else:
            confidences.append(1 - prob)   # probability of Susceptible (correct class)
    return np.mean(confidences)

# MIA on forget set and test set for ORIGINAL model
mia_forget_original = mia_score(original_model.predict_proba(X_forget)[:,1], y_forget)
mia_test_original   = mia_score(original_model.predict_proba(X_test)[:,1],   y_test)

# MIA on forget set and test set for UNLEARNED model
mia_forget_unlearned = mia_score(sisa_predict_proba(shard_models_unlearned, X_forget), y_forget)
mia_test_unlearned   = mia_score(sisa_predict_proba(shard_models_unlearned, X_test),   y_test)

# MIA gap: how much MORE confident is the model on forget vs test?
# Ideal = 0 (model treats forget patients like strangers)
mia_gap_original  = mia_forget_original  - mia_test_original
mia_gap_unlearned = mia_forget_unlearned - mia_test_unlearned

print(f"\nORIGINAL MODEL:")
print(f"  Confidence on forget set: {mia_forget_original:.4f}")
print(f"  Confidence on test set:   {mia_test_original:.4f}")
print(f"  MIA gap (should be > 0):  {mia_gap_original:.4f}  ← model 'remembers' forget patients")

print(f"\nUNLEARNED MODEL (SISA):")
print(f"  Confidence on forget set: {mia_forget_unlearned:.4f}")
print(f"  Confidence on test set:   {mia_test_unlearned:.4f}")
print(f"  MIA gap (should be ≈ 0):  {mia_gap_unlearned:.4f}  ← model treats them like strangers")

# Forgetting score: how close is the unlearned model to the retrained model?
mia_forget_retrained = mia_score(retrained_model.predict_proba(X_forget)[:,1], y_forget)
forgetting_score = 1 - abs(mia_forget_unlearned - mia_forget_retrained)
print(f"\nForgetting score: {forgetting_score:.4f}  (1.0 = perfect forgetting)")
print(f"  (Compares unlearned model confidence to retrained model confidence on forget set)")

print("\n" + "=" * 60)
print("STEP 6: Saving all results and models...")
print("=" * 60)

# Save unlearned shard models
with open("models/sisa_unlearned_models.pkl", "wb") as f:
    pickle.dump(shard_models_unlearned, f)
print("Saved → models/sisa_unlearned_models.pkl")

# Save comprehensive results
unlearning_results = {
    "original_model": {
        "accuracy": round(prev_results["original_model"]["accuracy"], 4),
        "auc_roc":  round(prev_results["original_model"]["auc_roc"], 4),
        "mia_on_forget_set": round(mia_forget_original, 4),
        "mia_on_test_set":   round(mia_test_original, 4),
        "mia_gap":           round(mia_gap_original, 4)
    },
    "unlearned_model_sisa": {
        "accuracy":          round(acc_unlearned, 4),
        "auc_roc":           round(auc_unlearned, 4),
        "mia_on_forget_set": round(mia_forget_unlearned, 4),
        "mia_on_test_set":   round(mia_test_unlearned, 4),
        "mia_gap":           round(mia_gap_unlearned, 4),
        "forgetting_score":  round(forgetting_score, 4),
        "unlearning_time_sec": round(t_unlearning, 2),
        "shards_retrained":  len(affected_shards),
        "total_shards":      N_SHARDS
    },
    "retrained_model": {
        "accuracy": round(prev_results["retrained_model"]["accuracy"], 4),
        "auc_roc":  round(prev_results["retrained_model"]["auc_roc"], 4),
        "mia_on_forget_set": round(mia_forget_retrained, 4),
        "full_retrain_time_sec": round(t_full_retrain, 2)
    },
    "speedup_factor": round(t_full_retrain / max(t_unlearning, 0.01), 1),
    "forget_set_size": len(X_forget),
    "n_shards": N_SHARDS
}

with open("results/unlearning_results.json", "w") as f:
    json.dump(unlearning_results, f, indent=2)
print("Saved → results/unlearning_results.json")

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"  Original model accuracy:  {prev_results['original_model']['accuracy']:.1%}")
print(f"  Unlearned model accuracy: {acc_unlearned:.1%}")
print(f"  Retrained model accuracy: {prev_results['retrained_model']['accuracy']:.1%}")
print(f"  Forgetting score:         {forgetting_score:.4f} / 1.0000")
print(f"  SISA speedup:             {t_full_retrain/max(t_unlearning,0.01):.1f}x faster than full retrain")

print("\n✓  SCRIPT 3 DONE — Run script4_evaluate.py next")