"""
SCRIPT 5: GENERATE PUBLICATION-QUALITY FIGURES
================================================
Creates 6 figures for the paper using matplotlib/seaborn.
All figures are saved in the graphs/ folder as high-resolution PNG files.

FIGURES GENERATED:
  Fig 1: Model accuracy comparison (Original / SISA Unlearned / Retrained)
  Fig 2: Effect of forget set size on accuracy and MIA gap
  Fig 3: Top 15 most important features for AMR prediction
  Fig 4: Computation time comparison (SISA vs full retrain)
  Fig 5: Privacy verification — MIA confidence scores
  Fig 6: Performance by organism type

HOW TO RUN:
  python script5_graphs.py
"""

import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

os.makedirs("graphs", exist_ok=True)

# ── Global style settings ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         12,
    "axes.titlesize":    14,
    "axes.titleweight":  "bold",
    "axes.labelsize":    12,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

# Colour palette (colourblind-friendly)
COLORS = {
    "original":   "#2196F3",   # blue
    "unlearned":  "#FF9800",   # orange
    "retrained":  "#4CAF50",   # green
    "accent":     "#9C27B0",   # purple
    "danger":     "#F44336",   # red
}

# ── Load results ──────────────────────────────────────────────────────────────
with open("results/unlearning_results.json") as f:
    unlearn = json.load(f)

exp1_df = pd.read_csv("results/exp1_forget_size.csv")
exp2_df = pd.read_csv("results/exp2_feature_comparison.csv")
exp3_df = pd.read_csv("results/exp3_by_organism.csv")
exp4_df = pd.read_csv("results/exp4_cumulative_deletion.csv")
feat_df = pd.read_csv("results/feature_importance.csv")

orig_acc = unlearn["original_model"]["accuracy"]
unl_acc  = unlearn["unlearned_model_sisa"]["accuracy"]
ret_acc  = unlearn["retrained_model"]["accuracy"]
orig_auc = unlearn["original_model"]["auc_roc"]
unl_auc  = unlearn["unlearned_model_sisa"]["auc_roc"]
ret_auc  = unlearn["retrained_model"]["auc_roc"]

print("Generating figures...\n")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Model performance comparison (Accuracy + AUC)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Machine Unlearning vs Baseline Models\non ARMD Antibiotic Resistance Dataset",
             fontsize=14, fontweight="bold", y=1.02)

models      = ["Original\n(all data)", "SISA\n(unlearned)", "Retrained\n(gold standard)"]
accuracies  = [orig_acc, unl_acc, ret_acc]
aucs        = [orig_auc, unl_auc, ret_auc]
bar_colors  = [COLORS["original"], COLORS["unlearned"], COLORS["retrained"]]

# Accuracy subplot
ax = axes[0]
bars = ax.bar(models, accuracies, color=bar_colors, width=0.5, edgecolor="white", linewidth=1.5)
ax.set_ylim(max(0, min(accuracies) - 0.05), min(1.0, max(accuracies) + 0.06))
ax.set_ylabel("Accuracy")
ax.set_title("Prediction Accuracy")
ax.set_xlabel("Model")
for bar, val in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
ax.axhline(y=orig_acc, color=COLORS["original"], linestyle="--", alpha=0.4, linewidth=1)
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# AUC subplot
ax = axes[1]
bars = ax.bar(models, aucs, color=bar_colors, width=0.5, edgecolor="white", linewidth=1.5)
ax.set_ylim(max(0, min(aucs) - 0.05), min(1.0, max(aucs) + 0.06))
ax.set_ylabel("AUC-ROC")
ax.set_title("AUC-ROC Score")
ax.set_xlabel("Model")
for bar, val in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("graphs/fig1_model_comparison.png")
plt.close()
print("  Fig 1 saved → graphs/fig1_model_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Effect of forget set size
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Effect of Forget Set Size on Unlearning Performance",
             fontsize=14, fontweight="bold")

# Accuracy vs forget size
ax = axes[0]
ax.plot(exp1_df["forget_size"], exp1_df["accuracy"], "o-",
        color=COLORS["unlearned"], linewidth=2.5, markersize=8, label="Unlearned")
ax.axhline(y=orig_acc, color=COLORS["original"], linestyle="--",
           linewidth=2, label=f"Original ({orig_acc:.3f})")
ax.axhline(y=ret_acc,  color=COLORS["retrained"],  linestyle=":",
           linewidth=2, label=f"Retrained ({ret_acc:.3f})")
ax.set_xlabel("Number of Patients Forgotten")
ax.set_ylabel("Accuracy on Test Set")
ax.set_title("Accuracy vs Forget Set Size")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# MIA gap vs forget size
ax = axes[1]
ax.plot(exp1_df["forget_size"], exp1_df["mia_gap"], "s-",
        color=COLORS["danger"], linewidth=2.5, markersize=8)
ax.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)
ax.fill_between(exp1_df["forget_size"], exp1_df["mia_gap"], 0,
                alpha=0.15, color=COLORS["danger"])
ax.set_xlabel("Number of Patients Forgotten")
ax.set_ylabel("MIA Gap (lower = better privacy)")
ax.set_title("Privacy Leakage vs Forget Set Size\n(MIA Gap)")
ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("graphs/fig2_forget_size_analysis.png")
plt.close()
print("  Fig 2 saved → graphs/fig2_forget_size_analysis.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Feature importance (top 15)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))

top15 = feat_df.head(15).sort_values("importance")
# Clean feature names for display
def clean_name(name):
    if name.startswith("org_"):   return name[4:].replace("_", " ")
    if name.startswith("abx_"):   return name[4:].replace("_", " ")
    if name.startswith("culture_"): return "Culture: " + name[8:].replace("_", " ")
    return name.replace("_", " ").title()

labels = [clean_name(n) for n in top15["feature"]]
values = top15["importance"].values

# Colour bars by feature type
bar_colors_feat = []
for n in top15["feature"]:
    if n.startswith("org_"):       bar_colors_feat.append("#E91E63")
    elif n.startswith("abx_"):     bar_colors_feat.append("#2196F3")
    elif n.startswith("culture_"): bar_colors_feat.append("#FF9800")
    else:                          bar_colors_feat.append("#4CAF50")

bars = ax.barh(labels, values, color=bar_colors_feat, edgecolor="white", linewidth=0.8)
for bar, val in zip(bars, values):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)

# Legend
legend_elements = [
    mpatches.Patch(color="#E91E63", label="Organism type"),
    mpatches.Patch(color="#2196F3", label="Antibiotic"),
    mpatches.Patch(color="#FF9800", label="Culture type"),
    mpatches.Patch(color="#4CAF50", label="Clinical/demographic"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)")
ax.set_title("Top 15 Features for Antibiotic Resistance Prediction\n(Random Forest, ARMD dataset)")
ax.grid(axis="x", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("graphs/fig3_feature_importance.png")
plt.close()
print("  Fig 3 saved → graphs/fig3_feature_importance.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Computation time — SISA vs full retrain
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

t_sisa    = unlearn["unlearned_model_sisa"]["unlearning_time_sec"]
t_retrain = unlearn["retrained_model"]["full_retrain_time_sec"]
speedup   = unlearn["speedup_factor"]

methods = ["Full Retraining\n(current practice)", f"SISA Unlearning\n(proposed method)"]
times   = [t_retrain, t_sisa]
colors  = [COLORS["danger"], COLORS["unlearned"]]

bars = ax.bar(methods, times, color=colors, width=0.45,
              edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{val:.1f}s", ha="center", va="bottom", fontweight="bold", fontsize=13)

ax.set_ylabel("Time (seconds)")
ax.set_title(f"Computation Time: SISA vs Full Retraining\n"
             f"SISA is {speedup:.1f}× faster for patient deletion requests")
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Add speedup annotation arrow
ax.annotate(f"{speedup:.1f}× faster",
            xy=(1, t_sisa), xytext=(0.5, (t_retrain + t_sisa) / 2),
            arrowprops=dict(arrowstyle="->", color=COLORS["retrained"], lw=2),
            fontsize=13, color=COLORS["retrained"], fontweight="bold",
            ha="center")

plt.tight_layout()
plt.savefig("graphs/fig4_time_comparison.png")
plt.close()
print("  Fig 4 saved → graphs/fig4_time_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Privacy verification — MIA scores
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

categories  = ["Forget set\n(Original model)", "Test set\n(Original model)",
               "Forget set\n(SISA Unlearned)", "Test set\n(SISA Unlearned)",
               "Forget set\n(Retrained model)"]
mia_vals    = [
    unlearn["original_model"]["mia_on_forget_set"],
    unlearn["original_model"]["mia_on_test_set"],
    unlearn["unlearned_model_sisa"]["mia_on_forget_set"],
    unlearn["unlearned_model_sisa"]["mia_on_test_set"],
    unlearn["retrained_model"]["mia_on_forget_set"],
]
bar_colors_mia = [COLORS["original"], COLORS["original"],
                  COLORS["unlearned"], COLORS["unlearned"],
                  COLORS["retrained"]]

bars = ax.bar(categories, mia_vals, color=bar_colors_mia,
              width=0.55, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, mia_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)

ax.set_ylabel("Mean Prediction Confidence (MIA score)")
ax.set_title("Privacy Verification via Membership Inference Attack\n"
             "Lower gap between forget and test = better privacy protection")
ax.set_ylim(0, max(mia_vals) * 1.15)
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Add MIA gap annotations
gap_orig = unlearn["original_model"]["mia_gap"]
gap_unl  = unlearn["unlearned_model_sisa"]["mia_gap"]
ax.annotate(f"Gap={gap_orig:.4f}\n(privacy leak)", xy=(0.5, max(mia_vals[0], mia_vals[1])),
            fontsize=9, ha="center", color=COLORS["danger"])
ax.annotate(f"Gap={gap_unl:.4f}\n(minimal leak)", xy=(2.5, max(mia_vals[2], mia_vals[3])),
            fontsize=9, ha="center", color=COLORS["retrained"])

legend_handles = [
    mpatches.Patch(color=COLORS["original"],  label="Original model"),
    mpatches.Patch(color=COLORS["unlearned"], label="SISA unlearned"),
    mpatches.Patch(color=COLORS["retrained"], label="Retrained (gold standard)"),
]
ax.legend(handles=legend_handles, fontsize=10)
plt.tight_layout()
plt.savefig("graphs/fig5_privacy_verification.png")
plt.close()
print("  Fig 5 saved → graphs/fig5_privacy_verification.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Performance by organism + cumulative deletion
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Organism-Level Analysis and Cumulative Deletion Scenario",
             fontsize=14, fontweight="bold")

# Organism subplot
ax = axes[0]
if len(exp3_df) > 0:
    top_orgs = exp3_df.sort_values("n_samples", ascending=False).head(8)
    x   = np.arange(len(top_orgs))
    w   = 0.35
    ax.bar(x - w/2, top_orgs["acc_original"],  width=w, label="Original",
           color=COLORS["original"],  edgecolor="white")
    ax.bar(x + w/2, top_orgs["acc_unlearned"], width=w, label="SISA Unlearned",
           color=COLORS["unlearned"], edgecolor="white")
    ax.set_xticks(x)
    org_labels = [o[:20] for o in top_orgs["organism"].tolist()]
    ax.set_xticklabels(org_labels, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Organism Type")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
else:
    ax.text(0.5, 0.5, "Insufficient organism data", ha="center", transform=ax.transAxes)

# Cumulative deletion subplot
ax = axes[1]
ax.plot(exp4_df["cumulative_deleted"], exp4_df["accuracy"], "D-",
        color=COLORS["unlearned"], linewidth=2.5, markersize=8, label="SISA accuracy")
ax.axhline(y=orig_acc, color=COLORS["original"], linestyle="--",
           linewidth=2, label=f"Original ({orig_acc:.3f})")
ax.axhline(y=ret_acc,  color=COLORS["retrained"],  linestyle=":",
           linewidth=2, label=f"Retrained ({ret_acc:.3f})")
ax.set_xlabel("Total Patients Deleted (Cumulative)")
ax.set_ylabel("Accuracy on Test Set")
ax.set_title("Cumulative Deletion Scenario\n(Simulated hospital deletion requests over time)")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("graphs/fig6_organism_cumulative.png")
plt.close()
print("  Fig 6 saved → graphs/fig6_organism_cumulative.png")


print("\n" + "=" * 60)
print("ALL 6 FIGURES GENERATED!")
print("=" * 60)
for i in range(1, 7):
    names = ["model_comparison", "forget_size_analysis", "feature_importance",
             "time_comparison", "privacy_verification", "organism_cumulative"]
    print(f"  Figure {i}: graphs/fig{i}_{names[i-1]}.png")

print("\n✓  SCRIPT 5 DONE — All results and graphs are ready!")
print("\nNext step: Share the graphs/ and results/ folders with your girlfriend")
print("so she can write the biological interpretation section of the paper.")
