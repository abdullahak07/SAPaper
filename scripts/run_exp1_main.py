"""
run_exp1_main.py
================
Main results: RF + XGBoost, 3 seeds, both datasets (ARMD + PATRIC).
Produces: results/exp1_main_results.csv

Usage:
    py run_exp1_main.py                    # both datasets
    py run_exp1_main.py --dataset ARMD     # ARMD only
    py run_exp1_main.py --dataset PATRIC   # PATRIC only
"""

import argparse, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
from utils import (load_config, load_armd, load_patric,
                   run_all_methods, add_speedup)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ARMD", "PATRIC", "both"],
                        default="both")
    parser.add_argument("--config", default="configs/config.json")
    parser.add_argument("--out",    default="results/exp1_main_results.csv")
    args = parser.parse_args()

    cfg   = load_config(args.config)
    seeds = cfg["seeds"]
    rows  = []

    datasets = (["ARMD", "PATRIC"] if args.dataset == "both"
                else [args.dataset])

    for dataset in datasets:
        for seed in seeds:
            print(f"\n{'='*55}")
            print(f"  {dataset} | seed={seed}")
            print(f"{'='*55}")

            if dataset == "ARMD":
                data = load_armd(cfg, seed=seed)
            else:
                data = load_patric(cfg, seed=seed)

            X_r, y_r, X_t, y_t, X_f, y_f = data

            for model_type in ["RF", "XGB"]:
                r = run_all_methods(model_type, cfg,
                                    X_r, y_r, X_t, y_t,
                                    X_f, y_f, seed, dataset)
                rows.extend(r)

    df = pd.DataFrame(rows)
    df = add_speedup(df)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}")

    # Quick summary
    summary = (df.groupby(["dataset", "model", "method"])
               [["accuracy", "speedup", "acc_drop", "mia_gap"]]
               .mean().round(4))
    print("\nSummary (mean across seeds):")
    print(summary.to_string())


if __name__ == "__main__":
    main()
