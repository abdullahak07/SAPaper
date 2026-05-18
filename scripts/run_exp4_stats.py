"""
run_exp4_stats.py
=================
Computes summary statistics (mean ± SD) and paired t-tests across seeds.
Input:  results/exp1_main_results.csv
Output: results/exp4_summary_stats.csv
        results/exp4_pvalues.csv

Usage:
    py run_exp4_stats.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from scipy import stats


def fmt(mean, std, decimals=4):
    """Format as 'mean ± std'."""
    fmt_str = f"{{:.{decimals}f}}"
    return f"{fmt_str.format(mean)} ± {fmt_str.format(std)}"


def main():
    inp = "results/exp1_main_results.csv"
    if not os.path.exists(inp):
        print(f"ERROR: {inp} not found. Run run_exp1_main.py first.")
        return

    df  = pd.read_csv(inp)
    os.makedirs("results", exist_ok=True)

    # ── 1. Summary table (mean ± SD across seeds) ─────────────────────────
    grp  = df.groupby(["dataset", "model", "method"])
    cols = ["accuracy", "auc", "mia_gap", "time_sec", "speedup", "acc_drop"]

    summary_rows = []
    for (dataset, model, method), g in grp:
        row = {"dataset": dataset, "model": model, "method": method}
        for c in cols:
            vals = g[c].dropna().values
            row[f"{c}_mean"] = vals.mean()
            row[f"{c}_std"]  = vals.std()
            row[f"{c}_fmt"]  = fmt(vals.mean(), vals.std(),
                                   decimals=4 if "acc" in c or "auc" in c
                                   else 3)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv("results/exp4_summary_stats.csv", index=False)
    print("Summary stats saved: results/exp4_summary_stats.csv")

    # ── 2. Paired t-tests: SISA vs other methods ──────────────────────────
    pval_rows = []
    for dataset in df["dataset"].unique():
        for model in df["model"].unique():
            sub = df[(df["dataset"] == dataset) & (df["model"] == model)]

            sisa_time = sub[sub["method"] == "SISA"]["time_sec"].values
            sisa_acc  = sub[sub["method"] == "SISA"]["accuracy"].values

            for method in sub["method"].unique():
                if method == "SISA":
                    continue
                m_time = sub[sub["method"] == method]["time_sec"].values
                m_acc  = sub[sub["method"] == method]["accuracy"].values

                # Need at least 2 paired observations
                n = min(len(sisa_time), len(m_time))
                if n < 2:
                    p_t = p_a = float("nan")
                else:
                    _, p_t = stats.ttest_rel(
                        sisa_time[:n], m_time[:n])
                    _, p_a = stats.ttest_rel(
                        sisa_acc[:n], m_acc[:n])

                pval_rows.append({
                    "dataset": dataset,
                    "model": model,
                    "comparison": f"SISA vs {method}",
                    "sisa_time_mean": sisa_time.mean(),
                    "other_time_mean": m_time.mean(),
                    "p_value_time": p_t,
                    "p_value_acc": p_a,
                    "time_sig": "✓" if p_t < 0.05 else "—",
                    "acc_sig":  "✓" if p_a < 0.05 else "—",
                    "n_seeds": n,
                    "note": ("indicative only (n=3 seeds)"
                             if n == 3 else "")
                })

    pvals = pd.DataFrame(pval_rows)
    pvals.to_csv("results/exp4_pvalues.csv", index=False)
    print("P-values saved: results/exp4_pvalues.csv")

    # ── 3. Print key results table ────────────────────────────────────────
    print("\n" + "="*65)
    print("KEY RESULTS — SISA mean ± SD across 3 seeds")
    print("="*65)
    sisa_summary = summary[summary["method"] == "SISA"][
        ["dataset", "model",
         "accuracy_fmt", "speedup_fmt", "acc_drop_fmt",
         "mia_gap_fmt", "time_sec_fmt"]]
    print(sisa_summary.to_string(index=False))

    print("\n" + "="*65)
    print("SIGNIFICANCE: SISA time vs Full Retrain")
    print("="*65)
    sig = pvals[pvals["comparison"] == "SISA vs Full Retrain"][
        ["dataset", "model",
         "sisa_time_mean", "other_time_mean",
         "p_value_time", "time_sig", "note"]]
    print(sig.to_string(index=False))


if __name__ == "__main__":
    main()
