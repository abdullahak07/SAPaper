"""
run_exp3_forget_size.py
=======================
Forget-size ablation: forget_n in {500, 1000, 5000, 10000}.
Compares SISA vs Full Retrain time and accuracy across deletion scales.
Produces: results/exp3_forget_size.csv

Usage:
    py run_exp3_forget_size.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from utils import (load_config, load_armd, load_patric,
                   get_model, method_full_retrain, method_sisa,
                   safe_proba)


def load_patric_custom_forget(cfg, forget_n, seed=42):
    """PATRIC loader that supports larger forget sets."""
    import pandas as _pd
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    df = _pd.read_csv(cfg["patric"]["raw_csv"])
    df["target"] = (df["resistant_phenotype"] == "Resistant").astype(int)
    cat_cols = ["antibiotic", "species", "laboratory_typing_method",
                "host_name", "isolation_country"]
    for c in cat_cols:
        df[c] = df[c].fillna("Unknown")
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
    df["measurement_value"] = np.log1p(
        pd.to_numeric(df["measurement_value"], errors="coerce").fillna(0))
    fc = cat_cols + ["measurement_value"]
    df = df[fc + ["target"]].dropna().reset_index(drop=True)

    # Stratified forget set
    n_each = min(forget_n // 2, len(df[df.target == 0]) // 10)
    fg0 = df[df.target == 0].sample(n_each, random_state=seed)
    fg1 = df[df.target == 1].sample(n_each, random_state=seed)
    forget = _pd.concat([fg0, fg1])
    remain = df.drop(forget.index).reset_index(drop=True)
    tr_i, te_i = train_test_split(remain.index, test_size=0.2,
                                   random_state=seed,
                                   stratify=remain["target"])
    return (remain.loc[tr_i, fc].values, remain.loc[tr_i, "target"].values,
            remain.loc[te_i, fc].values, remain.loc[te_i, "target"].values,
            forget[fc].values, forget["target"].values)


def main():
    cfg          = load_config("configs/config.json")
    forget_sizes = cfg["forget_sizes"]
    seed         = 42
    rows         = []

    for dataset in ["ARMD", "PATRIC"]:
        print(f"\n{'='*55}\nForget-size ablation — {dataset}\n{'='*55}")

        for forget_n in forget_sizes:
            print(f"\n  forget_n = {forget_n:,}")

            if dataset == "ARMD":
                data = load_armd(cfg, forget_n=forget_n, seed=seed)
            else:
                data = load_patric_custom_forget(cfg, forget_n, seed=seed)

            X_r, y_r, X_t, y_t, X_f, y_f = data

            for model_type in ["RF", "XGB"]:
                orig = get_model(model_type, cfg, seed)
                orig.fit(X_r, y_r)
                orig_acc = accuracy_score(
                    y_t, (safe_proba(orig, X_t)[:, 1] >= 0.5).astype(int))

                rt   = method_full_retrain(model_type, cfg, X_r, y_r,
                                           X_t, y_t, X_f, seed, orig_acc)
                sisa = method_sisa(model_type, cfg, X_r, y_r,
                                   X_t, y_t, X_f, seed, orig_acc, k=5)
                speedup = rt["time_sec"] / sisa["time_sec"]

                rows.append({
                    "dataset": dataset,
                    "model": model_type,
                    "forget_n": forget_n,
                    "retrain_time": rt["time_sec"],
                    "sisa_time": sisa["time_sec"],
                    "speedup": speedup,
                    "sisa_acc": sisa["accuracy"],
                    "sisa_acc_drop": sisa["acc_drop"],
                    "retrain_acc_drop": rt["acc_drop"],
                    "sisa_mia": sisa["mia_gap"],
                    "retrain_mia": rt["mia_gap"],
                })
                print(f"    {model_type}: retrain={rt['time_sec']:.2f}s  "
                      f"sisa={sisa['time_sec']:.3f}s  "
                      f"speedup={speedup:.1f}×  "
                      f"acc={sisa['accuracy']:.4f}")

    df = pd.DataFrame(rows)
    out = "results/exp3_forget_size.csv"
    os.makedirs("results", exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print(df[["dataset", "model", "forget_n",
              "speedup", "sisa_acc_drop"]].to_string(index=False))


if __name__ == "__main__":
    main()
