# Machine Unlearning for GDPR Right-to-Erasure in Antimicrobial Resistance Prediction Models

<div align="center">

[![Preprint](https://img.shields.io/badge/Preprint-medRxiv%202026-b31b1b?style=for-the-badge)](#citation)
[![Status](https://img.shields.io/badge/Status-Submitted%20to%20JAMIA-1f6feb?style=for-the-badge)](#)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-6f42c1?style=for-the-badge)](#license)
[![Domain](https://img.shields.io/badge/Domain-AMR%20Prediction-0a7ea4?style=for-the-badge)](#overview)
[![Task](https://img.shields.io/badge/Task-Machine%20Unlearning-8a2be2?style=for-the-badge)](#overview)

**Official repository for the paper**
**"Machine Unlearning for GDPR Right-to-Erasure in Antimicrobial Resistance Prediction Models"**

*medRxiv Manuscript ID: **MEDRXIV/2026/347960***
*JAMIA Manuscript ID: **amiajnl-2026-019295***

[Overview](#overview) •
[Highlights](#highlights) •
[Datasets](#datasets) •
[Methods](#methods) •
[Results](#results) •
[Citation](#citation)

</div>

---

## Overview

Healthcare machine learning systems trained on patient data must support the **GDPR right to erasure**.
The default solution—**full retraining**—is exact but can become operationally inefficient when deletion requests arrive repeatedly.

This work evaluates **Sharded, Isolated, Sliced, and Aggregated (SISA)** training as a practical machine unlearning framework for **antimicrobial resistance (AMR) prediction**, using both:

* **clinical electronic health record (EHR)** data, and
* **genomic surveillance** data.

We compare SISA against multiple baselines and quantify the trade-off between:

* predictive performance,
* privacy-related behavior,
* unlearning efficiency, and
* cumulative compliance cost.

---

## Highlights

* **Fast unlearning:** SISA achieves **8.9x** speedup on ARMD and **9.8x** speedup on BV-BRC/PATRIC versus full retraining.
* **Clinical utility preserved:** accuracy degradation remains below the paper's **0.5% operational threshold**.
* **Cross-modality evaluation:** validated on both **EHR** and **genomic** AMR prediction settings.
* **Practical compliance framing:** models can be updated efficiently in response to repeated deletion requests.
* **Large-scale study:** experiments span **1.24M+** ARMD records and **400k+** BV-BRC/PATRIC records.

---

## Paper at a Glance

| Item               | Details                                                                                        |
| ------------------ | ---------------------------------------------------------------------------------------------- |
| Title              | **Machine Unlearning for GDPR Right-to-Erasure in Antimicrobial Resistance Prediction Models** |
| Authors            | **Saniya Saniya**, **Abdullah Ahmad Khan**                                                     |
| Preprint           | medRxiv 2026                                                                                   |
| Journal Submission | JAMIA                                                                                          |
| Model Family       | Random Forest                                                                                  |
| Primary Method     | **SISA Training**                                                                              |
| Application        | Antimicrobial Resistance Prediction                                                            |

---

## Datasets

### 1) ARMD

**Antibiotic Resistance Microbiology Dataset**

* **Modality:** Clinical EHR
* **Size:** 1,245,767 records
* **Source:** Stanford Health Care / MIT ClinicalML
* **Access:** Data Use Agreement

### 2) BV-BRC / PATRIC

**Bacterial and Viral Bioinformatics Resource Center**

* **Modality:** Genomic surveillance
* **Size:** 400,372 records
* **Source:** NIH BV-BRC / PATRIC
* **Access:** Public resource / API-based access

---

## Methods

### Compared Unlearning Approaches

| Method                     | Description                                        |
| -------------------------- | -------------------------------------------------- |
| **Full Retraining**        | Retrain the model from scratch after deletion      |
| **SISA Training**          | Shard-based training enabling localized retraining |
| **Label-Flip Retraining**  | Heuristic adversarial deletion baseline            |
| **Influence Reweighting**  | Down-weight forget samples during retraining       |
| **Selective Tree Pruning** | Remove trees based on forget-set behavior          |

### Experimental Configuration

```text
Model: Random Forest
n_estimators: 500
max_depth: 12
min_samples_leaf: 5
Forget set size: 500 records per dataset
Shard count (SISA): 5
```

### Metrics

* Accuracy
* AUC-ROC
* Membership Inference Attack (MIA) gap
* Wall-clock unlearning time
* 12-month cumulative deletion cost

---

## Results

### Core Performance Summary

| Dataset       | Method          |      Time |  Speedup | Accuracy Drop |
| ------------- | --------------- | --------: | -------: | ------------: |
| ARMD          | Full Retraining |    66.7 s |     1.0x |             — |
| ARMD          | **SISA**        | **7.5 s** | **8.9x** |    **0.024%** |
| BV-BRC/PATRIC | Full Retraining |    13.4 s |     1.0x |             — |
| BV-BRC/PATRIC | **SISA**        | **1.4 s** | **9.8x** |    **0.048%** |

### Cumulative Compliance Cost

At **50 deletion requests per month** over **12 months**:

| Dataset       | Full Retraining |     SISA |
| ------------- | --------------: | -------: |
| ARMD          |           800 s | **90 s** |
| BV-BRC/PATRIC |           160 s | **16 s** |

---

## Repository Structure

```text
.
├── README.md
├── paper/
│   └── manuscript files
├── figures/
│   └── plots and result visualizations
├── data/
│   └── preprocessing scripts / dataset loaders
├── experiments/
│   └── training and evaluation scripts
└── results/
    └── tables, logs, summary outputs
```

---

## Intended Audience

This repository is relevant to researchers working in:

* machine unlearning,
* health informatics,
* privacy-aware machine learning,
* antimicrobial resistance prediction,
* clinical machine learning deployment.

---

## Data Availability

* **ARMD:** available through MIT ClinicalML under a **Data Use Agreement**
* **BV-BRC/PATRIC:** accessible via the BV-BRC platform and associated APIs

Please follow the original dataset terms of use and access requirements.

---

## Citation

### BibTeX

```bibtex
@article{SaniyaKhan2026SISA,
  title   = {Machine Unlearning for GDPR Right-to-Erasure in Antimicrobial Resistance Prediction Models},
  author  = {Saniya Saniya and Abdullah Ahmad Khan},
  journal = {medRxiv},
  year    = {2026},
  note    = {Preprint, Manuscript ID MEDRXIV/2026/347960}
}
```

---

## License

This preprint repository is intended to follow the licensing terms selected for the medRxiv posting.
For manuscript reuse, redistribution, or derivative use, please follow the applicable preprint license and journal policies.

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
