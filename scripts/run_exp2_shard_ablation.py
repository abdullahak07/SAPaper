"""
run_exp2_shard_ablation.py
==========================
Shard ablation: k in {2, 3, 5, 10} for RF and XGBoost on both datasets.
Produces: results/exp2_shard_ablation.csv

Usage:
    py run_exp2_shard_ablation.py
"""

import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from utils import (load_config, load_armd, load_patric,
                   get_model, method_full_retrain, method_sisa,
                   safe_proba, evaluate)


def main():
    cfg      = load_config("configs/config.json")
    k_values = cfg["shard_k_values"]
    seed     = 42          # fixed seed for ablation
    rows     = []

    for dataset in ["ARMD", "PATRIC"]:
        print(f"\n{'='*55}\nShard ablation — {dataset}\n{'='*55}")

        if dataset == "ARMD":
            X_r, y_r, X_t, y_t, X_f, y_f = load_armd(cfg, seed=seed)
        else:
            X_r, y_r, X_t, y_t, X_f, y_f = load_patric(cfg, seed=seed)

        for model_type in ["RF", "XGB"]:
            # Reference: full retrain time
            orig = get_model(model_type, cfg, seed)
            orig.fit(X_r, y_r)
            orig_acc = accuracy_score(
                y_t, (safe_proba(orig, X_t)[:, 1] >= 0.5).astype(int))

            rt = method_full_retrain(model_type, cfg, X_r, y_r,
                                     X_t, y_t, X_f, seed, orig_acc)
            retrain_time = rt["time_sec"]

            for k in k_values:
                print(f"  {model_type} k={k}...", end=" ", flush=True)
                res = method_sisa(model_type, cfg, X_r, y_r,
                                  X_t, y_t, X_f, seed, orig_acc, k=k)
                speedup = retrain_time / res["time_sec"]
                rows.append({
                    "dataset": dataset, "model": model_type,
                    "k": k, "speedup": speedup,
                    **res
                })
                print(f"speedup={speedup:.1f}×  "
                      f"acc={res['accuracy']:.4f}  "
                      f"time={res['time_sec']:.3f}s")

    df = pd.DataFrame(rows)
    out = "results/exp2_shard_ablation.csv"
    os.makedirs("results", exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print(df[["dataset", "model", "k", "speedup",
              "accuracy", "acc_drop"]].to_string(index=False))


if __name__ == "__main__":
    main()
