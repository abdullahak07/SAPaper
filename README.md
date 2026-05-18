# Efficient Machine Unlearning for Antimicrobial Resistance Prediction in Clinical and Genomic Data

<div align="center">

[![Journal Target](https://img.shields.io/badge/Journal%20Target-Artificial%20Intelligence%20in%20Medicine-1f6feb?style=for-the-badge)](#paper-at-a-glance)
[![Domain](https://img.shields.io/badge/Domain-AMR%20Prediction-0a7ea4?style=for-the-badge)](#overview)
[![Task](https://img.shields.io/badge/Task-Machine%20Unlearning-8a2be2?style=for-the-badge)](#overview)
[![Models](https://img.shields.io/badge/Models-RF%20%2B%20XGBoost-0b7285?style=for-the-badge)](#methods)
[![License](https://img.shields.io/badge/License-See%20repository%20license-6f42c1?style=for-the-badge)](#license)

**Official repository for the manuscript**

**“Efficient Machine Unlearning for Antimicrobial Resistance Prediction in Clinical and Genomic Data”**

[Overview](#overview) •
[Highlights](#highlights) •
[Datasets](#datasets) •
[Methods](#methods) •
[Results](#results) •
[Reproducibility](#reproducibility) •
[Citation](#citation)

</div>

---

## Overview

Clinical machine learning systems trained on patient data need practical mechanisms for handling **right-to-erasure** requests without repeatedly retraining entire models from scratch.

This repository supports a study of **Sharded, Isolated, Sliced, and Aggregated (SISA)**-style machine unlearning for **antimicrobial resistance (AMR) prediction** across two data modalities:

- **clinical electronic health record (EHR)** microbiology data, and
- **genomic surveillance** AMR phenotype data.

The study evaluates whether shard-isolated retraining can reduce deletion-time computation while preserving predictive performance for AMR prediction models.

The current manuscript evaluates:

- two large AMR datasets,
- two tree-ensemble model families,
- three deterministic seeds,
- multiple unlearning baselines,
- shard-count sensitivity, and
- forget-size scalability from 500 to 10,000 deletion records.

---

## Highlights

- **Two model families:** Random Forest and XGBoost are evaluated.
- **Two AMR data modalities:** clinical EHR data and genomic surveillance data.
- **Multi-seed evaluation:** primary experiments use seeds 42, 123, and 456.
- **SISA speedups:** under the recommended `k = 5` setting, SISA achieves **3.08×–7.76×** speedup over full retraining.
- **Accuracy preserved:** mean accuracy changes remain within a **0.5 percentage-point operational threshold**.
- **Deployment analysis:** shard-count and forget-size ablations quantify practical utility–efficiency trade-offs.
- **Clinical workflow framing:** annual deletion-handling projections estimate model-computation savings for repeated right-to-erasure requests.

---

## Paper at a Glance

| Item | Details |
|---|---|
| Title | **Efficient Machine Unlearning for Antimicrobial Resistance Prediction in Clinical and Genomic Data** |
| Authors | **Saniya**, **Abdullah Ahmad Khan** |
| Journal target | **Artificial Intelligence in Medicine** |
| Task | Machine unlearning for AMR prediction model maintenance |
| Main method | SISA / shard-isolated retraining |
| Model families | Random Forest and XGBoost |
| Main datasets | ARMD and BV-BRC/PATRIC |
| Primary shard count | `k = 5` |
| Seeds | 42, 123, 456 |

---

## Datasets

### 1. ARMD

**Antibiotic Resistance Microbiology Dataset**

- **Modality:** Clinical EHR / microbiology records
- **Size used in manuscript:** 1,245,767 records
- **Source:** Stanford Health Care / MIT ClinicalML
- **Access:** Data Use Agreement required

ARMD is **not redistributed** in this repository. Users must obtain access from the original data provider and follow the dataset terms of use.

### 2. BV-BRC / PATRIC

**Bacterial and Viral Bioinformatics Resource Center genomic AMR phenotype data**

- **Modality:** Genomic surveillance
- **Size used in manuscript:** 400,372 records
- **Source:** NIH BV-BRC / PATRIC
- **Access:** Public API-based access, subject to BV-BRC terms

---

## Methods

### Compared unlearning approaches

| Method | Description |
|---|---|
| **Full Retraining** | Gold-standard retraining on the retained data after deletion |
| **SISA / Shard-Isolated Retraining** | Partition data into shard models and retrain only affected shard models |
| **Label-Flip Retraining** | Heuristic baseline that relabels forget records before retraining |
| **Influence-Inspired Reweighting** | Practical down-weighting baseline using near-zero forget-sample weights |
| **Selective Tree Pruning** | Random-Forest-only baseline that removes trees based on forget-set behavior |

### Experimental configuration

```text
Datasets: ARMD, BV-BRC/PATRIC
Models: Random Forest, XGBoost
Primary SISA shard count: k = 5
Seeds: 42, 123, 456
Primary forget-set size: 500 records
Forget-size ablation: 500, 1,000, 5,000, 10,000 records
Shard-count ablation: k = 2, 3, 5, 10
```

### Metrics

- Accuracy
- AUC-ROC
- Membership-inference-attack confidence gap
- Wall-clock model retraining/unlearning time
- Speedup relative to full retraining
- Accuracy drop relative to full retraining
- Clinical workflow compute-time projection

> **Timing note:** reported runtimes measure model retraining/unlearning computation on prepared feature matrices. They do not include dataset download, disk I/O, data cleaning, preprocessing, clinical validation, governance review, or production redeployment overhead.

---

## Results

### Main SISA results at `k = 5`

Values are means across seeds 42, 123, and 456.

| Dataset | Model | SISA accuracy | SISA AUC | Speedup vs full retrain | Accuracy drop |
|---|---:|---:|---:|---:|---:|
| ARMD | Random Forest | 0.8475 | 0.8160 | **5.70×** | 0.0020 |
| ARMD | XGBoost | 0.8615 | 0.8465 | **3.08×** | 0.0013 |
| BV-BRC/PATRIC | Random Forest | 0.6862 | 0.7736 | **7.76×** | -0.0074 |
| BV-BRC/PATRIC | XGBoost | 0.6940 | 0.7783 | **4.25×** | -0.0005 |

Negative accuracy drop means the SISA model achieved marginally higher test accuracy than full retraining in that setting. These small negative drops should be interpreted as evidence that SISA did not degrade predictive utility, not as a guaranteed accuracy improvement.

### Statistical timing comparison

SISA timing reductions relative to full retraining were consistent across the four model–dataset settings. Because the analysis uses three deterministic seeds, p-values are treated as **indicative** rather than definitive inferential evidence.

| Dataset | Model | SISA time | Full retrain time | p-value |
|---|---:|---:|---:|---:|
| ARMD | Random Forest | 0.174 s | 0.991 s | 0.000480 |
| ARMD | XGBoost | 0.131 s | 0.406 s | 0.015025 |
| BV-BRC/PATRIC | Random Forest | 0.249 s | 1.928 s | 0.000153 |
| BV-BRC/PATRIC | XGBoost | 0.088 s | 0.374 s | 0.001071 |

### Shard-count ablation

The shard-count ablation evaluates `k = 2, 3, 5, 10`.

Main finding: larger `k` generally improves speedup, but very large shard counts may increase utility risk. In the manuscript, `k = 5` is used as the recommended default because it provides a stable balance between deletion-time speed and accuracy preservation.

### Forget-size ablation

The forget-size ablation evaluates deletion batches of `500`, `1,000`, `5,000`, and `10,000` records.

Main finding: SISA preserves accuracy within the 0.5 percentage-point operational threshold across all tested model–dataset–deletion-size settings.

---

## Repository Structure

The repository may contain some or all of the following directories depending on the current release state:

```text
.
├── README.md
├── paper/
│   └── manuscript files and LaTeX sources
├── figures/
│   └── generated figures for the manuscript
├── results/
│   └── CSV outputs, LaTeX tables, and summaries
├── experiments/
│   └── experiment scripts
├── data/
│   └── dataset loading or preprocessing utilities
└── requirements.txt
```

---

## Reproducibility

This repository is intended to support reproducibility of the manuscript’s experiments and figures.

Because ARMD is distributed under a Data Use Agreement, raw ARMD data are not included. Users must obtain ARMD independently and place it in the expected local data path before running ARMD experiments.

A typical workflow is:

```bash
# Install dependencies
python -m pip install -r requirements.txt

# Run all experiments, tables, and figures if the required datasets are available
python run_all.py
```

If the repository layout differs, check script-level comments or configuration files for expected input/output paths.

---

## Data Availability

- **ARMD:** available through MIT ClinicalML under a Data Use Agreement.
- **BV-BRC/PATRIC:** accessible through the BV-BRC platform and associated APIs.
- **Code and scripts:** provided in this repository for reproducibility, subject to dataset access restrictions.

Please follow all original dataset terms of use and access requirements.

---

## Limitations

The manuscript and repository focus on operational deletion-time efficiency and predictive utility. Current limitations include:

- privacy verification uses a lightweight confidence-gap proxy rather than certified deletion or full shadow-model MIA evaluation;
- experiments are limited to tree-based model families;
- real-world deletion requests may affect multiple shards unless patient-level shard routing is enforced;
- absolute runtime values depend on hardware, implementation, preprocessing, and validation workflows;
- clinical deployment requires governance, audit trails, validation, and model registry processes beyond the benchmark code.

---

## Intended Audience

This repository is relevant to researchers and practitioners working on:

- machine unlearning,
- clinical AI model maintenance,
- privacy-aware machine learning,
- health informatics,
- antimicrobial resistance prediction,
- right-to-erasure workflows,
- tabular clinical machine learning.

---

## Citation

If you use this repository or build on this work, please cite the manuscript once the final citation details are available.

### BibTeX placeholder

```bibtex
@article{saniya_khan_2026_amr_unlearning,
  title   = {Efficient Machine Unlearning for Antimicrobial Resistance Prediction in Clinical and Genomic Data},
  author  = {Saniya and Khan, Abdullah Ahmad},
  year    = {2026},
  note    = {Manuscript under review / preprint details to be updated}
}
```

---

## License

Please follow the repository license and the terms of use of the underlying datasets. Raw ARMD data are not redistributed because access is governed by the original Data Use Agreement.

---

## Contact

**Abdullah Ahmad Khan**  
School of Information Technology, Murdoch University  
Perth, Western Australia, Australia

For questions related to the manuscript or repository, please open an issue or contact the corresponding author.

---

<div align="center">

**If this repository is useful, please consider starring it.**

</div>
