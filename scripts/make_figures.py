"""
make_figures.py
===============
Generates all AIM upgrade figures from saved results CSVs.

Figures produced:
  Fig A — Main results: RF vs XGB speedup comparison (ARMD + PATRIC)
  Fig B — Multi-seed robustness: mean ± SD bars
  Fig C — Shard ablation: speedup and acc_drop vs k
  Fig D — Forget-size ablation: time and speedup vs forget_n
  Fig E — Clinical workflow: cumulative deletion time projection

Usage:
    py make_figures.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

COLORS = {
    "Full Retrain":          "#4C72B0",
    "SISA":                  "#2196F3",
    "Label-Flip Retraining": "#FF9800",
    "Influence Reweighting": "#9E9E9E",
    "Tree Pruning":          "#795548",
}
RF_COLOR  = "#2196F3"
XGB_COLOR = "#4CAF50"
THRESHOLD = 0.005   # 0.5% accuracy threshold

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


# ── Fig A: RF vs XGB speedup side-by-side ──────────────────────────────────
def fig_a_model_comparison():
    fpath = "results/exp1_main_results.csv"
    if not os.path.exists(fpath):
        print(f"  Skipping Fig A — {fpath} not found"); return

    df   = pd.read_csv(fpath)
    grp  = df.groupby(["dataset", "model", "method"])["speedup"].mean()
    methods = ["Full Retrain", "SISA", "Label-Flip Retraining",
               "Influence Reweighting", "Tree Pruning"]
    datasets = ["ARMD", "PATRIC"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x     = np.arange(len(methods))
    width = 0.35

    for ax, ds in zip(axes, datasets):
        rf_vals  = [grp.get((ds, "RF",  m), 0) for m in methods]
        xgb_vals = [grp.get((ds, "XGB", m), 0) for m in methods]
        ax.bar(x - width/2, rf_vals,  width, label="Random Forest",
               color=RF_COLOR,  alpha=0.85, edgecolor="white")
        ax.bar(x + width/2, xgb_vals, width, label="XGBoost",
               color=XGB_COLOR, alpha=0.85, edgecolor="white")
        ax.axhline(1, color="black", ls="--", lw=0.8, label="Baseline (1×)")
        ax.set_title(f"{ds} Dataset")
        ax.set_ylabel("Speedup vs Full Retrain (×)")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" ", "\n") for m in methods],
                           fontsize=8)
        ax.legend(fontsize=8)
        for i, (rv, xv) in enumerate(zip(rf_vals, xgb_vals)):
            if rv > 1.5:
                ax.text(i - width/2, rv + 0.2, f"{rv:.1f}×",
                        ha="center", fontsize=7)
            if xv > 1.5:
                ax.text(i + width/2, xv + 0.2, f"{xv:.1f}×",
                        ha="center", fontsize=7)

    fig.suptitle("SISA Speedup: Random Forest vs XGBoost", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT}/figA_model_comparison.png", bbox_inches="tight")
    plt.close()
    print("  Saved figA_model_comparison.png")


# ── Fig B: Multi-seed robustness (mean ± SD) ──────────────────────────────
def fig_b_multiseed():
    fpath = "results/exp4_summary_stats.csv"
    if not os.path.exists(fpath):
        print(f"  Skipping Fig B — {fpath} not found"); return

    df = pd.read_csv(fpath)
    datasets = ["ARMD", "PATRIC"]
    methods  = ["Full Retrain", "SISA", "Label-Flip Retraining",
                "Influence Reweighting", "Tree Pruning"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    for col_i, ds in enumerate(datasets):
        sub = df[(df["dataset"] == ds) & (df["model"] == "RF")]
        sub = sub[sub["method"].isin(methods)].set_index("method").reindex(methods)

        # Speedup
        ax = axes[0, col_i]
        means = sub["speedup_mean"].fillna(0).values
        stds  = sub["speedup_std"].fillna(0).values
        colors = [COLORS.get(m, "#607D8B") for m in methods]
        bars = ax.bar(range(len(methods)), means, yerr=stds,
                      color=colors, capsize=4, alpha=0.85, edgecolor="white")
        ax.axhline(1, color="black", ls="--", lw=0.8)
        ax.set_title(f"{ds} — Speedup (RF, mean±SD, n=3)")
        ax.set_ylabel("Speedup (×)")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace(" ","\n") for m in methods], fontsize=8)
        for i, (m_v, s_v) in enumerate(zip(means, stds)):
            if m_v > 1: ax.text(i, m_v + s_v + 0.3, f"{m_v:.1f}×",
                                ha="center", fontsize=7.5)

        # Accuracy drop
        ax2 = axes[1, col_i]
        means2 = sub["acc_drop_mean"].fillna(0).values * 100
        stds2  = sub["acc_drop_std"].fillna(0).values * 100
        ax2.bar(range(len(methods)), means2, yerr=stds2,
                color=colors, capsize=4, alpha=0.85, edgecolor="white")
        ax2.axhline(THRESHOLD * 100, color="red", ls="--", lw=1,
                    label="0.5% threshold")
        ax2.set_title(f"{ds} — Accuracy Drop (RF, mean±SD, n=3)")
        ax2.set_ylabel("Accuracy Drop (%)")
        ax2.set_xticks(range(len(methods)))
        ax2.set_xticklabels([m.replace(" ","\n") for m in methods], fontsize=8)
        ax2.legend(fontsize=8)

    fig.suptitle("Multi-Seed Robustness (3 seeds: 42, 123, 456)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT}/figB_multiseed_robustness.png", bbox_inches="tight")
    plt.close()
    print("  Saved figB_multiseed_robustness.png")


# ── Fig C: Shard ablation ──────────────────────────────────────────────────
def fig_c_shard_ablation():
    fpath = "results/exp2_shard_ablation.csv"
    if not os.path.exists(fpath):
        print(f"  Skipping Fig C — {fpath} not found"); return

    df = pd.read_csv(fpath)
    datasets = ["ARMD", "PATRIC"]
    models   = ["RF", "XGB"]
    ls_map   = {"RF": "-o", "XGB": "-s"}
    c_map    = {"RF": RF_COLOR, "XGB": XGB_COLOR}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for col_i, ds in enumerate(datasets):
        sub = df[df["dataset"] == ds]

        ax_s = axes[0, col_i]
        ax_a = axes[1, col_i]

        for mt in models:
            m_sub = sub[sub["model"] == mt].sort_values("k")
            ax_s.plot(m_sub["k"], m_sub["speedup"],
                      ls_map[mt], color=c_map[mt],
                      label=mt, linewidth=2, markersize=7)
            ax_a.plot(m_sub["k"], m_sub["acc_drop"] * 100,
                      ls_map[mt], color=c_map[mt],
                      label=mt, linewidth=2, markersize=7)

        ax_s.set_title(f"{ds} — Speedup vs Shard Count")
        ax_s.set_xlabel("Number of shards (k)")
        ax_s.set_ylabel("Speedup vs Full Retrain (×)")
        ax_s.set_xticks(df["k"].unique())
        ax_s.legend()
        ax_s.axhline(1, color="gray", ls="--", lw=0.8)

        ax_a.axhline(THRESHOLD * 100, color="red", ls="--", lw=1,
                     label="0.5% threshold")
        ax_a.set_title(f"{ds} — Accuracy Drop vs Shard Count")
        ax_a.set_xlabel("Number of shards (k)")
        ax_a.set_ylabel("Accuracy Drop (%)")
        ax_a.set_xticks(df["k"].unique())
        ax_a.legend()

    fig.suptitle("Shard Count Ablation (k ∈ {2, 3, 5, 10})",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT}/figC_shard_ablation.png", bbox_inches="tight")
    plt.close()
    print("  Saved figC_shard_ablation.png")


# ── Fig D: Forget-size ablation ────────────────────────────────────────────
def fig_d_forget_size():
    fpath = "results/exp3_forget_size.csv"
    if not os.path.exists(fpath):
        print(f"  Skipping Fig D — {fpath} not found"); return

    df = pd.read_csv(fpath)
    datasets = ["ARMD", "PATRIC"]
    models   = ["RF", "XGB"]
    c_map    = {"RF": RF_COLOR, "XGB": XGB_COLOR}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for col_i, ds in enumerate(datasets):
        sub = df[df["dataset"] == ds]

        ax_t = axes[0, col_i]
        ax_s = axes[1, col_i]

        for mt in models:
            m_sub = sub[sub["model"] == mt].sort_values("forget_n")
            ax_t.plot(m_sub["forget_n"] / 1000, m_sub["sisa_time"],
                      "-o", color=c_map[mt], label=f"SISA {mt}", lw=2, ms=7)
            ax_t.plot(m_sub["forget_n"] / 1000, m_sub["retrain_time"],
                      "--", color=c_map[mt], label=f"Retrain {mt}",
                      lw=1.5, ms=5, alpha=0.6)
            ax_s.plot(m_sub["forget_n"] / 1000, m_sub["speedup"],
                      "-o", color=c_map[mt], label=mt, lw=2, ms=7)

        ax_t.set_title(f"{ds} — Unlearning Time vs Forget Size")
        ax_t.set_xlabel("Forget set size (×1000 records)")
        ax_t.set_ylabel("Unlearning time (s)")
        ax_t.legend(fontsize=8)

        ax_s.axhline(1, color="gray", ls="--", lw=0.8)
        ax_s.set_title(f"{ds} — SISA Speedup vs Forget Size")
        ax_s.set_xlabel("Forget set size (×1000 records)")
        ax_s.set_ylabel("Speedup vs Full Retrain (×)")
        ax_s.legend(fontsize=8)

    fig.suptitle("Forget-Set Size Ablation (500 → 10,000 records)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT}/figD_forget_size.png", bbox_inches="tight")
    plt.close()
    print("  Saved figD_forget_size.png")


# ── Fig E: Clinical workflow projection ────────────────────────────────────
def fig_e_clinical_workflow():
    """Projects cumulative deletion cost at hospital scale."""
    # Monthly deletion volumes: 50 (original), 100, 250, 500
    monthly = np.array([50, 100, 250, 500])
    months  = 12

    # Times from ARMD results (RF): retrain ~9.5s, SISA ~1.3s per deletion
    armd_retrain = 9.5
    armd_sisa    = 1.3

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: annual hours vs monthly deletion volume
    ax = axes[0]
    retrain_hours = armd_retrain * monthly * months / 3600
    sisa_hours    = armd_sisa    * monthly * months / 3600
    ax.plot(monthly, retrain_hours, "-o", color="#4C72B0",
            label="Full Retrain", lw=2, ms=8)
    ax.plot(monthly, sisa_hours, "-o", color=RF_COLOR,
            label="SISA (proposed)", lw=2, ms=8)
    ax.fill_between(monthly, sisa_hours, retrain_hours,
                    alpha=0.15, color=RF_COLOR, label="Time saved")
    ax.set_xlabel("Monthly deletion requests")
    ax.set_ylabel("Annual unlearning time (hours)")
    ax.set_title("Annual Compliance Burden vs Deletion Volume\n(ARMD, RF)")
    ax.legend()
    for m_v, r_h, s_h in zip(monthly, retrain_hours, sisa_hours):
        ax.annotate(f"{r_h:.1f}h", (m_v, r_h), textcoords="offset points",
                    xytext=(5, 5), fontsize=8, color="#4C72B0")
        ax.annotate(f"{s_h:.2f}h", (m_v, s_h), textcoords="offset points",
                    xytext=(5, -12), fontsize=8, color=RF_COLOR)

    # Right: cumulative time per month (500 deletions/month scenario)
    ax2 = axes[1]
    month_range = np.arange(1, 13)
    m500_rt = armd_retrain * 500 * month_range / 3600
    m500_s  = armd_sisa    * 500 * month_range / 3600
    ax2.plot(month_range, m500_rt, "-o", color="#4C72B0",
             label="Full Retrain", lw=2, ms=6)
    ax2.plot(month_range, m500_s,  "-o", color=RF_COLOR,
             label="SISA", lw=2, ms=6)
    ax2.fill_between(month_range, m500_s, m500_rt,
                     alpha=0.15, color=RF_COLOR)
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Cumulative unlearning time (hours)")
    ax2.set_title("Cumulative Burden at 500 Monthly Deletions\n(ARMD, RF)")
    ax2.legend()
    ax2.annotate(f"{m500_rt[-1]:.1f}h/year",
                 (12, m500_rt[-1]), textcoords="offset points",
                 xytext=(-40, 8), fontsize=9, color="#4C72B0", fontweight="bold")
    ax2.annotate(f"{m500_s[-1]:.2f}h/year",
                 (12, m500_s[-1]), textcoords="offset points",
                 xytext=(-50, -15), fontsize=9, color=RF_COLOR, fontweight="bold")

    fig.suptitle("Clinical Deployment Workflow: SISA vs Full Retraining",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT}/figE_clinical_workflow.png", bbox_inches="tight")
    plt.close()
    print("  Saved figE_clinical_workflow.png")


if __name__ == "__main__":
    print("Generating AIM figures...")
    fig_a_model_comparison()
    fig_b_multiseed()
    fig_c_shard_ablation()
    fig_d_forget_size()
    fig_e_clinical_workflow()
    print(f"\nAll figures saved to ./{OUT}/")
