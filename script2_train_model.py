"""
SCRIPT 2: TRAIN MODELS
=======================
Trains two Random Forest classifiers on the ARMD dataset:
  1. ORIGINAL MODEL  — trained on ALL training data (including future forget set)
  2. RETRAINED MODEL — trained only on the retain set (gold standard for comparison)

These two models are the benchmarks. In Script 3, we apply machine unlearning
and compare the unlearned model against both of these.

HOW TO RUN:
  python script2_train_model.py
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             classification_report, confusion_matrix)

os.makedirs("models",  exist_ok=True)
os.makedirs("results", exist_ok=True)

print("=" * 60)
print("STEP 1: Loading prepared data...")
print("=" * 60)

# Load the splits created by Script 1
train_df  = pd.read_csv("data/train_data.csv")
test_df   = pd.read_csv("data/test_data.csv")
retain_df = pd.read_csv("data/retain_set.csv")

# Load the feature column names
with open("data/feature_columns.txt") as f:
    feature_cols = [line.strip() for line in f if line.strip()]

print(f"Training samples (full):   {len(train_df):,}")
print(f"Training samples (retain): {len(retain_df):,}")
print(f"Test samples:              {len(test_df):,}")
print(f"Number of features:        {len(feature_cols)}")

# ARMD uses the string "Null" for missing values — replace with 0
# This must be done before converting to numeric arrays
for df in [train_df, test_df, retain_df]:
    df.replace("Null", 0, inplace=True)
    df.replace("null", 0, inplace=True)
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# Separate features (X) and labels (y)
X_train  = train_df[feature_cols].values
y_train  = train_df["target"].values
X_retain = retain_df[feature_cols].values
y_retain = retain_df["target"].values
X_test   = test_df[feature_cols].values
y_test   = test_df["target"].values

print(f"\nResistance rate in test set: {y_test.mean():.1%}")

print("\n" + "=" * 60)
print("STEP 2: Training ORIGINAL model (all training data)...")
print("=" * 60)

# Random Forest with 100 trees
# n_jobs=-1 means use all CPU cores for speed
original_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
original_model.fit(X_train, y_train)

# Evaluate on test set
y_pred_orig  = original_model.predict(X_test)
y_prob_orig  = original_model.predict_proba(X_test)[:, 1]  # probability of Resistant
acc_orig     = accuracy_score(y_test, y_pred_orig)
auc_orig     = roc_auc_score(y_test, y_prob_orig)

print(f"\n  Accuracy: {acc_orig:.4f}  ({acc_orig*100:.1f}%)")
print(f"  AUC-ROC:  {auc_orig:.4f}")
print(f"\n  Classification Report:\n{classification_report(y_test, y_pred_orig, target_names=['Susceptible','Resistant'])}")

# Save the model
with open("models/original_model.pkl", "wb") as f:
    pickle.dump(original_model, f)
print("  Saved → models/original_model.pkl")

print("\n" + "=" * 60)
print("STEP 3: Training RETRAINED model (retain set only)...")
print("=" * 60)

# This is the gold standard — a model that was NEVER trained on the forget patients
# In practice this is expensive (requires full retraining), which is why machine
# unlearning is valuable: it achieves the same result much faster
retrained_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
retrained_model.fit(X_retain, y_retain)

y_pred_ret = retrained_model.predict(X_test)
y_prob_ret = retrained_model.predict_proba(X_test)[:, 1]
acc_ret    = accuracy_score(y_test, y_pred_ret)
auc_ret    = roc_auc_score(y_test, y_prob_ret)

print(f"\n  Accuracy: {acc_ret:.4f}  ({acc_ret*100:.1f}%)")
print(f"  AUC-ROC:  {auc_ret:.4f}")
print(f"\n  Classification Report:\n{classification_report(y_test, y_pred_ret, target_names=['Susceptible','Resistant'])}")

with open("models/retrained_model.pkl", "wb") as f:
    pickle.dump(retrained_model, f)
print("  Saved → models/retrained_model.pkl")

print("\n" + "=" * 60)
print("STEP 4: Feature importance analysis...")
print("=" * 60)

# Which features matter most for predicting resistance?
importances = pd.Series(original_model.feature_importances_, index=feature_cols)
top_features = importances.sort_values(ascending=False).head(15)

print("\nTop 15 most important features for AMR prediction:")
for feat, imp in top_features.items():
    bar = "█" * int(imp * 300)
    print(f"  {feat:<35} {imp:.4f}  {bar}")

# Save feature importance
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": original_model.feature_importances_
}).sort_values("importance", ascending=False)
importance_df.to_csv("results/feature_importance.csv", index=False)
print("\nSaved → results/feature_importance.csv")

print("\n" + "=" * 60)
print("STEP 5: Saving results...")
print("=" * 60)

results = {
    "original_model": {
        "accuracy": round(acc_orig, 4),
        "auc_roc":  round(auc_orig, 4),
        "training_samples": int(len(X_train))
    },
    "retrained_model": {
        "accuracy": round(acc_ret, 4),
        "auc_roc":  round(auc_ret, 4),
        "training_samples": int(len(X_retain))
    }
}

with open("results/model_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved → results/model_results.json")

print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
print(f"\n  Original  model  | Accuracy: {acc_orig:.1%}  | AUC: {auc_orig:.4f}")
print(f"  Retrained model  | Accuracy: {acc_ret:.1%}  | AUC: {auc_ret:.4f}")
print(f"\n  Accuracy drop from removing {len(train_df)-len(retain_df)} patients:")
print(f"  {abs(acc_orig - acc_ret):.1%} — this is the 'cost' of forgetting")

print("\n✓  SCRIPT 2 DONE — Run script3_machine_unlearning.py next")