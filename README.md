# AMR Machine Unlearning — AIM Upgrade Experiments

Code for reproducing all experiments in:

> **"Efficient Machine Unlearning for Antimicrobial Resistance Prediction
> in Clinical and Genomic Data"**
> Saniya, Abdullah Ahmad Khan — *Artificial Intelligence in Medicine* (submitted)

---

## Setup

```bash
pip install -r requirements.txt
```

## Data

Place the ARMD and PATRIC data files as follows:

```
data/
  retain_set.csv
  test_data.csv
  forget_set.csv
  feature_columns.txt
patric_data/
  patric_amr_raw.csv
```

**ARMD** is available under a Data Use Agreement from MIT ClinicalML:
https://clinicalml.org/data/amr-dataset/

**PATRIC/BV-BRC** is freely available via the NIH REST API:
https://www.bv-brc.org/api/genome_amr/

---

## Run All Experiments

```bash
# Full run (~20-25 min on modern CPU)
py run_all.py

# ARMD only (~8-10 min)
py run_all.py --dataset ARMD

# PATRIC only
py run_all.py --dataset PATRIC
```

## Run Individual Experiments

```bash
# Exp 1: Main results (RF + XGBoost, 3 seeds)
py scripts/run_exp1_main.py --dataset ARMD
py scripts/run_exp1_main.py --dataset PATRIC

# Exp 2: Shard ablation (k = 2, 3, 5, 10)
py scripts/run_exp2_shard_ablation.py

# Exp 3: Forget-size ablation (500 → 10,000)
py scripts/run_exp3_forget_size.py

# Exp 4: Statistical tests (requires Exp 1 output)
py scripts/run_exp4_stats.py

# Generate figures
py scripts/make_figures.py

# Generate LaTeX tables
py scripts/make_latex_tables.py
```

---

## Output Structure

```
results/
  exp1_main_results.csv      ← RF + XGB, 3 seeds, all methods
  exp2_shard_ablation.csv    ← k sweep
  exp3_forget_size.csv       ← forget-size sweep
  exp4_summary_stats.csv     ← mean ± SD across seeds
  exp4_pvalues.csv           ← paired t-test results
  latex_tables/              ← .tex snippets for paper

figures/
  figA_model_comparison.png  ← RF vs XGB speedup
  figB_multiseed_robustness  ← mean ± SD bars
  figC_shard_ablation.png    ← k sweep plots
  figD_forget_size.png       ← forget-size plots
  figE_clinical_workflow.png ← hospital deployment projection
```

---

## Experiment Configuration

All parameters in `configs/config.json`:
- Seeds: `[42, 123, 456]`
- n_estimators: `100`
- SISA shards (main): `k=5`
- Shard ablation: `k ∈ {2, 3, 5, 10}`
- Forget sizes: `{500, 1000, 5000, 10000}`
