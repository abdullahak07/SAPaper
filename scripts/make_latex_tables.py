"""
make_latex_tables.py
====================
Generates LaTeX table code for the AIM paper from results CSVs.
Outputs .tex snippets to results/latex_tables/

Usage:
    py make_latex_tables.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd

OUT_DIR = "results/latex_tables"
os.makedirs(OUT_DIR, exist_ok=True)

THRESHOLD = 0.005  # 0.5%
METHOD_ORDER = [
    "Full Retrain", "SISA",
    "Label-Flip Retraining", "Influence Reweighting", "Tree Pruning"
]
METHOD_SHORT = {
    "Full Retrain":          "Full Retrain",
    "SISA":                  r"\textbf{SISA (proposed)}",
    "Label-Flip Retraining": "Label-Flip Retrain.",
    "Influence Reweighting": "Influence Rewt.",
    "Tree Pruning":          "Tree Pruning",
}


def fmt_mean_std(mean, std, pct=False, times=False, decimals=3):
    """Format mean ± std, optionally as % or ×."""
    if pct:
        return f"{mean*100:.2f} $\\pm$ {std*100:.2f}"
    if times:
        return f"{mean:.1f} $\\pm$ {std:.1f}"
    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"


# ── Table 2/3: Main results RF ─────────────────────────────────────────────
def table_main_rf():
    fpath = "results/exp4_summary_stats.csv"
    if not os.path.exists(fpath):
        print(f"Skipping main RF table — {fpath} not found"); return

    df = pd.read_csv(fpath)

    for dataset in ["ARMD", "PATRIC"]:
        sub = df[(df["dataset"] == dataset) & (df["model"] == "RF")]
        sub = sub.set_index("method")

        lines = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(
            r"\caption{Machine unlearning on "
            + dataset
            + r" (RF, $n=3$ seeds, mean$\,\pm\,$SD). "
            r"$^\dagger$Violates 0.5\% accuracy threshold. "
            r"Bold = proposed SISA. "
            r"$\downarrow$ lower is better; $\uparrow$ higher is better.}")
        lines.append(
            r"\label{tab:" + dataset.lower() + r"_rf}")
        lines.append(r"\renewcommand{\arraystretch}{1.25}")
        lines.append(
            r"\begin{tabular}{L{3.2cm} C{2.3cm} C{2.0cm} C{2.2cm} C{2.4cm} C{1.8cm}}")
        lines.append(r"\toprule")
        lines.append(
            r"\textbf{Method} & \textbf{Acc.\,$\uparrow$} & "
            r"\textbf{AUC\,$\uparrow$} & "
            r"\textbf{|MIA|$\times10^{-3}$\,$\downarrow$} & "
            r"\textbf{Time (s)\,$\downarrow$} & "
            r"\textbf{Speedup\,$\uparrow$} \\")
        lines.append(r"\midrule")

        for method in METHOD_ORDER:
            if method not in sub.index:
                continue
            row   = sub.loc[method]
            label = METHOD_SHORT[method]
            acc   = fmt_mean_std(row["accuracy_mean"], row["accuracy_std"], pct=True)
            auc   = fmt_mean_std(row["auc_mean"], row["auc_std"])
            mia   = fmt_mean_std(row["mia_gap_mean"]*1e3,
                                  row["mia_gap_std"]*1e3, decimals=3)
            t     = fmt_mean_std(row["time_sec_mean"], row["time_sec_std"])
            spd   = fmt_mean_std(row["speedup_mean"], row["speedup_std"],
                                  times=True)

            # Flag threshold violation
            flag = ""
            if (row["acc_drop_mean"] > THRESHOLD and
                    method not in ["Full Retrain"]):
                flag = r"$^\dagger$"

            # Color SISA row
            color = r"\rowcolor{sisa}" if method == "SISA" else ""
            if row["acc_drop_mean"] > THRESHOLD:
                color = r"\rowcolor{bad}"

            lines.append(
                f"{color}{label}{flag} & {acc} & {auc} & "
                f"{mia} & {t} & {spd} \\\\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        out = f"{OUT_DIR}/table_{dataset.lower()}_rf.tex"
        with open(out, "w") as f:
            f.write("\n".join(lines))
        print(f"Saved {out}")


# ── Table 4: XGBoost results ───────────────────────────────────────────────
def table_xgboost():
    fpath = "results/exp4_summary_stats.csv"
    if not os.path.exists(fpath):
        print(f"Skipping XGB table — {fpath} not found"); return

    df  = pd.read_csv(fpath)
    sub = df[df["model"] == "XGB"]

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{XGBoost machine unlearning results (mean$\,\pm\,$SD, "
        r"$n=3$ seeds). Tree Pruning is not applicable to XGBoost; "
        r"results shown for completeness with label-flip proxy.}")
    lines.append(r"\label{tab:xgboost}")
    lines.append(r"\renewcommand{\arraystretch}{1.25}")
    lines.append(
        r"\begin{tabular}{L{2.0cm} L{3.0cm} C{2.3cm} C{2.0cm} "
        r"C{2.2cm} C{1.6cm}}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Dataset} & \textbf{Method} & \textbf{Acc.\,$\uparrow$} & "
        r"\textbf{AUC\,$\uparrow$} & "
        r"\textbf{Time (s)\,$\downarrow$} & \textbf{Speedup\,$\uparrow$} \\")
    lines.append(r"\midrule")

    for dataset in ["ARMD", "PATRIC"]:
        ds_sub = sub[sub["dataset"] == dataset].set_index("method")
        first  = True
        for method in METHOD_ORDER:
            if method == "Tree Pruning":
                continue
            if method not in ds_sub.index:
                continue
            row   = ds_sub.loc[method]
            label = METHOD_SHORT[method]
            acc   = fmt_mean_std(row["accuracy_mean"], row["accuracy_std"], pct=True)
            auc   = fmt_mean_std(row["auc_mean"], row["auc_std"])
            t     = fmt_mean_std(row["time_sec_mean"], row["time_sec_std"])
            spd   = fmt_mean_std(row["speedup_mean"], row["speedup_std"], times=True)
            ds_str = dataset if first else ""
            first  = False
            color  = r"\rowcolor{sisa}" if method == "SISA" else ""
            lines.append(
                f"{color}{ds_str} & {label} & {acc} & {auc} & "
                f"{t} & {spd} \\\\")
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = f"{OUT_DIR}/table_xgboost.tex"
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {out}")


# ── Table 5: Statistical tests ─────────────────────────────────────────────
def table_stats():
    fpath = "results/exp4_pvalues.csv"
    if not os.path.exists(fpath):
        print(f"Skipping stats table — {fpath} not found"); return

    df = pd.read_csv(fpath)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Paired $t$-test results: SISA versus other methods "
        r"on unlearning time across three seeds. "
        r"$p < 0.05$ = statistically significant. "
        r"Note: indicative only (n=3 seeds).}")
    lines.append(r"\label{tab:stats}")
    lines.append(r"\renewcommand{\arraystretch}{1.25}")
    lines.append(
        r"\begin{tabular}{L{1.5cm} L{1.5cm} L{4.5cm} "
        r"C{2.5cm} C{2.5cm} C{1.5cm}}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Dataset} & \textbf{Model} & \textbf{Comparison} & "
        r"\textbf{SISA time (s)} & \textbf{Other time (s)} & "
        r"\textbf{$p$-value} \\")
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        p = row["p_value_time"]
        p_str = f"{p:.4f}" if not np.isnan(p) else "—"
        sig   = r" $^*$" if (not np.isnan(p) and p < 0.05) else ""
        lines.append(
            f"{row['dataset']} & {row['model']} & {row['comparison']} & "
            f"{row['sisa_time_mean']:.3f} & "
            f"{row['other_time_mean']:.3f} & "
            f"{p_str}{sig} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = f"{OUT_DIR}/table_stats.tex"
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {out}")


if __name__ == "__main__":
    print("Generating LaTeX tables...")
    table_main_rf()
    table_xgboost()
    table_stats()
    print(f"\nAll tables saved to ./{OUT_DIR}/")
