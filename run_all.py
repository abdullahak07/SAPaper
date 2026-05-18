"""
run_all.py
==========
Master script — runs all experiments sequentially then generates
figures and LaTeX tables.

Expected runtime on RTX 4090 / modern CPU:
  Exp 1 (main results): ~8-12 min
  Exp 2 (shard ablation): ~3-5 min
  Exp 3 (forget-size): ~4-6 min
  Exp 4 (stats): <1 min
  Figures + tables: <1 min
  Total: ~20-25 min

Usage:
    py run_all.py                      # full run
    py run_all.py --skip-patric        # ARMD only (faster, ~8 min)
    py run_all.py --dataset ARMD       # same as --skip-patric
"""

import argparse, os, subprocess, sys, time


def run(cmd, label):
    print(f"\n{'='*60}")
    print(f"  RUNNING: {label}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable] + cmd,
        capture_output=False
    )
    elapsed = time.time() - t0
    status = "✓ OK" if result.returncode == 0 else "✗ FAILED"
    print(f"\n  {status} ({elapsed:.0f}s)")
    if result.returncode != 0:
        print(f"  Error in {label} — continuing...")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ARMD", "PATRIC", "both"],
                        default="both")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--skip-tables",  action="store_true")
    args = parser.parse_args()

    scripts = "scripts"
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    results = {}

    # Exp 1: Main results
    results["exp1"] = run(
        [f"{scripts}/run_exp1_main.py", "--dataset", args.dataset],
        "Exp 1: Main results (RF + XGBoost, 3 seeds)")

    # Exp 2: Shard ablation
    results["exp2"] = run(
        [f"{scripts}/run_exp2_shard_ablation.py"],
        "Exp 2: Shard ablation (k = 2,3,5,10)")

    # Exp 3: Forget-size ablation
    results["exp3"] = run(
        [f"{scripts}/run_exp3_forget_size.py"],
        "Exp 3: Forget-size ablation (500 → 10k)")

    # Exp 4: Stats (requires exp1 output)
    if results.get("exp1"):
        results["exp4"] = run(
            [f"{scripts}/run_exp4_stats.py"],
            "Exp 4: Statistical tests")

    # Figures
    if not args.skip_figures:
        results["figs"] = run(
            [f"{scripts}/make_figures.py"],
            "Generate figures")

    # LaTeX tables
    if not args.skip_tables:
        results["tables"] = run(
            [f"{scripts}/make_latex_tables.py"],
            "Generate LaTeX tables")

    # Summary
    print(f"\n{'='*60}")
    print("  RUN COMPLETE — Summary")
    print(f"{'='*60}")
    for k, v in results.items():
        print(f"  {'✓' if v else '✗'} {k}")

    print("\nOutput files:")
    for root, dirs, files in os.walk("results"):
        for f in sorted(files):
            print(f"  results/{f}")
    for root, dirs, files in os.walk("figures"):
        for f in sorted(files):
            print(f"  figures/{f}")


if __name__ == "__main__":
    main()
