"""
SCRIPT 4: EXTENDED EVALUATION EXPERIMENTS
==========================================
Runs 4 experiments to strengthen the paper:
  1. Effect of forget set SIZE on accuracy and forgetting quality
  2. Top resistance-predictive features (biological interpretation)
  3. Unlearning performance by organism type
  4. Cumulative deletion scenario over time

HOW TO RUN:
  python script4_evaluate.py
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

os.makedirs("results", exist_ok=True)

print("=" * 60)
print("Loading data and models...")
print("=" * 60)

train_df  = pd.read_csv("data/train_data.csv")
test_df   = pd.read_csv("data/test_data.csv")

with open("data/feature_columns.txt") as f:
    feature_cols = [line.strip() for line in f if line.strip()]

with open("models/original_model.pkl",  "rb") as f: original_model  = pickle.load(f)
with open("models/retrained_model.pkl", "rb") as f: retrained_model = pickle.load(f)
with open("models/sisa_unlearned_models.pkl", "rb") as f: unlearned_models = pickle.load(f)
with open("results/unlearning_results.json") as f: unlearn_results = json.load(f)

X_train = train_df[feature_cols].replace({"Null": 0, "null": 0})
X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0).values
y_train = train_df["target"].values
X_test  = test_df[feature_cols].replace({"Null": 0, "null": 0})
X_test  = X_test.apply(pd.to_numeric, errors="coerce").fillna(0).values
y_test  = test_df["target"].values
N_SHARDS = unlearn_results["n_shards"]

def sisa_predict_proba(models, X):
    probs = np.array([m.predict_proba(X)[:, 1] for m in models])
    return probs.mean(axis=0)

def mia_score(probs, labels):
    return np.mean([p if l == 1 else 1 - p for p, l in zip(probs, labels)])

def run_sisa_unlearning(train_X, train_y, forget_indices):
    forget_set = set(forget_indices)
    shard_size = len(train_X) // N_SHARDS
    shard_indices = []
    shard_models  = []
    for i in range(N_SHARDS):
        start = i * shard_size
        end   = start + shard_size if i < N_SHARDS - 1 else len(train_X)
        idx   = list(range(start, end))
        shard_indices.append(idx)
        m = RandomForestClassifier(n_estimators=50, max_depth=10,
                                   min_samples_leaf=5, random_state=i, n_jobs=-1)
        m.fit(train_X[idx], train_y[idx])
        shard_models.append(m)
    for shard_id, shard_idx in enumerate(shard_indices):
        overlap = [i for i in shard_idx if i in forget_set]
        if not overlap:
            continue
        clean = [i for i in shard_idx if i not in forget_set]
        if len(clean) < 5:
            continue
        m = RandomForestClassifier(n_estimators=50, max_depth=10,
                                   min_samples_leaf=5, random_state=shard_id, n_jobs=-1)
        m.fit(train_X[clean], train_y[clean])
        shard_models[shard_id] = m
    return shard_models


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXPERIMENT 1: Effect of forget set SIZE")
print("=" * 60)

forget_sizes = [50, 100, 200, 500, 1000, 2000]
forget_sizes = [s for s in forget_sizes if s <= int(len(train_df) * 0.3)]
exp1 = []
np.random.seed(42)

for n in forget_sizes:
    f_idx  = np.random.choice(len(train_df), size=n, replace=False)
    models = run_sisa_unlearning(X_train, y_train, f_idx)
    probs  = sisa_predict_proba(models, X_test)
    acc    = accuracy_score(y_test, (probs >= 0.5).astype(int))
    auc    = roc_auc_score(y_test, probs)
    probs_f= sisa_predict_proba(models, X_train[f_idx])
    probs_t= sisa_predict_proba(models, X_test)
    mia_gap= mia_score(probs_f, y_train[f_idx]) - mia_score(probs_t, y_test)
    exp1.append({"forget_size": n, "accuracy": round(acc, 4),
                 "auc_roc": round(auc, 4), "mia_gap": round(mia_gap, 4)})
    print(f"  n={n:5d} | Acc={acc:.3f} | AUC={auc:.3f} | MIA gap={mia_gap:.4f}")

pd.DataFrame(exp1).to_csv("results/exp1_forget_size.csv", index=False)
print("Saved → results/exp1_forget_size.csv")


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXPERIMENT 2: Top predictive features")
print("=" * 60)

feat_imp = pd.read_csv("results/feature_importance.csv")
print(feat_imp.head(15).to_string(index=False))

feat_comparison = pd.DataFrame({
    "feature": feature_cols,
    "importance_orig":     original_model.feature_importances_,
    "importance_retrained": retrained_model.feature_importances_,
}).sort_values("importance_orig", ascending=False)
feat_comparison.to_csv("results/exp2_feature_comparison.csv", index=False)
print("Saved → results/exp2_feature_comparison.csv")


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXPERIMENT 3: Performance by organism")
print("=" * 60)

org_cols = [c for c in feature_cols if c.startswith("org_")]
exp3 = []
for col in org_cols:
    mask = test_df[col].values == 1
    if mask.sum() < 20:
        continue
    X_org = X_test[mask]; y_org = y_test[mask]
    acc_orig = accuracy_score(y_org, original_model.predict(X_org))
    prob_unl = sisa_predict_proba(unlearned_models, X_org)
    acc_unl  = accuracy_score(y_org, (prob_unl >= 0.5).astype(int))
    org_name = col.replace("org_", "")
    exp3.append({"organism": org_name, "n_samples": int(mask.sum()),
                 "resistance_rate": round(y_org.mean(), 3),
                 "acc_original": round(acc_orig, 4), "acc_unlearned": round(acc_unl, 4),
                 "acc_drop": round(acc_orig - acc_unl, 4)})
    print(f"  {org_name:<40} n={mask.sum():4d} | "
          f"Orig={acc_orig:.3f} | Unl={acc_unl:.3f} | Drop={acc_orig-acc_unl:+.3f}")

pd.DataFrame(exp3).sort_values("n_samples", ascending=False).to_csv(
    "results/exp3_by_organism.csv", index=False)
print("Saved → results/exp3_by_organism.csv")


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXPERIMENT 4: Cumulative deletion over time")
print("=" * 60)

batch_sizes = [50, 50, 100, 100, 200]
cumulative  = []
available   = list(range(len(train_df)))
exp4        = [{"batch": 0, "cumulative_deleted": 0,
                "accuracy": unlearn_results["original_model"]["accuracy"],
                "auc_roc":  unlearn_results["original_model"]["auc_roc"]}]
np.random.seed(99)

for i, batch in enumerate(batch_sizes):
    if batch > len(available): break
    new = list(np.random.choice(available, size=batch, replace=False))
    cumulative.extend(new)
    available = [x for x in available if x not in new]
    models = run_sisa_unlearning(X_train, y_train, cumulative)
    probs  = sisa_predict_proba(models, X_test)
    acc    = accuracy_score(y_test, (probs >= 0.5).astype(int))
    auc    = roc_auc_score(y_test, probs)
    exp4.append({"batch": i+1, "cumulative_deleted": len(cumulative),
                 "accuracy": round(acc, 4), "auc_roc": round(auc, 4)})
    print(f"  Batch {i+1}: {len(cumulative):,} total deleted | "
          f"Acc={acc:.4f} | AUC={auc:.4f}")

pd.DataFrame(exp4).to_csv("results/exp4_cumulative_deletion.csv", index=False)
print("Saved → results/exp4_cumulative_deletion.csv")


print("\n" + "=" * 60)
print("✓  SCRIPT 4 DONE — Run script5_graphs.py next")
print("=" * 60)