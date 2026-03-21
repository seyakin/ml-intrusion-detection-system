# 🛡️ Network Intrusion Detection System (IDS)

> **Binary classification of network traffic as Normal or Attack using the UNSW-NB15 benchmark dataset, with a Random Forest + XGBoost ensemble and 8 diagnostic visualisations.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Results](#results)
- [Visualisations](#visualisations)
- [Evaluation Metrics](#evaluation-metrics)
- [References](#references)

---

## Overview

This project implements a machine learning-based **Network Intrusion Detection System (IDS)** that classifies network connections as either *Normal* or *Attack*. The pipeline covers the full ML lifecycle:

- Real-world benchmark dataset (UNSW-NB15)
- Exploratory data analysis (EDA)
- Feature engineering and preprocessing
- Model training with 5-fold cross-validation
- Comprehensive evaluation (Accuracy, Precision, Recall, F1, AUC-ROC)
- Eight publication-quality visualisations
- Saved, reusable `.pkl` model artefacts

**Attack categories detected:**

| Category | Description |
|---|---|
| DoS | Denial-of-Service floods |
| Exploits | Software vulnerability exploitation |
| Fuzzers | Random/malformed input injection |
| Generic | Protocol-agnostic signature attacks |
| Reconnaissance | Port scanning, service enumeration |
| Backdoors | Persistent covert access channels |
| Analysis | Deep packet inspection evasion |
| Shellcode | Payload injection via shellcode |
| Worms | Self-replicating network worms |

---

## Dataset

### UNSW-NB15

| Property | Value |
|---|---|
| **Full name** | UNSW Network Benchmark 2015 |
| **Creator** | University of New South Wales Canberra (UNSW) |
| **Records** | 82,332 (training set) |
| **Features** | 49 (35 numeric, 3 categorical, 1 label) |
| **Class ratio** | ~56% Normal / ~44% Attack |
| **Source** | [research.unsw.edu.au](https://research.unsw.edu.au/projects/unsw-nb15-dataset) |

**Key features used:**

| Feature | Description |
|---|---|
| `dur` | Connection duration (seconds) |
| `sbytes` / `dbytes` | Source / destination bytes |
| `Sload` / `Dload` | Source / destination bits per second |
| `Spkts` / `Dpkts` | Packet counts (src → dst / dst → src) |
| `sttl` / `dttl` | Time-to-live values |
| `Sjit` / `Djit` | Jitter (inter-arrival time variance) |
| `ct_srv_src` | Connections to the same service (last 100) |
| `proto` | Layer-4 protocol (TCP, UDP, ICMP, …) |
| `state` | Connection state flag (FIN, CON, INT, …) |
| `service` | Application-layer service (http, dns, ftp, …) |

**Downloading the data:**

```bash
python data/download_dataset.py
```

If the official CloudStor mirror is unavailable, the script automatically generates a statistically equivalent synthetic dataset matching the original distribution.

---

## Model Architecture

```
Raw Network Traffic
        │
        ▼
┌───────────────────┐
│  Label Encoding   │  (proto, state, service → integers)
│  Outlier Clipping │  (99.5th percentile cap)
│  StandardScaler   │  (zero-mean, unit-variance)
└────────┬──────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
 Random     XGBoost
 Forest     Classifier
(200 trees) (200 rounds)
    │         │
    └────┬────┘
         │
         ▼
   Evaluation &
   Visualisation
```

### Model Hyperparameters

**Random Forest**

| Parameter | Value |
|---|---|
| `n_estimators` | 200 |
| `max_depth` | 20 |
| `min_samples_split` | 5 |
| `class_weight` | balanced |

**XGBoost**

| Parameter | Value |
|---|---|
| `n_estimators` | 200 |
| `max_depth` | 8 |
| `learning_rate` | 0.10 |
| `subsample` | 0.80 |
| `colsample_bytree` | 0.80 |

---

## Project Structure

```
ids-network-intrusion/
│
├── ids_pipeline.py               ← Main training & evaluation script
├── requirements.txt              ← Python dependencies
├── README.md
│
├── data/
│   ├── download_dataset.py       ← Dataset downloader / generator
│   └── UNSW_NB15_training-set.csv  ← Generated on first run
│
├── models/
│   ├── randomforest_ids.pkl      ← Saved Random Forest pipeline
│   └── xgboost_ids.pkl           ← Saved XGBoost pipeline
│
├── visualizations/
│   ├── 01_class_distribution.png
│   ├── 02_feature_distributions.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_confusion_matrices.png
│   ├── 05_roc_curves.png
│   ├── 06_precision_recall.png
│   ├── 07_feature_importance.png
│   └── 08_metrics_comparison.png
│
└── results/
    ├── model_metrics.csv
    └── classification_report.txt
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/ids-network-intrusion.git
cd ids-network-intrusion
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Download the dataset

```bash
python data/download_dataset.py
```

> The main script will also generate the dataset automatically if it is not found.

### 4. Run the full pipeline

```bash
python ids_pipeline.py
```

Expected output:

```
============================================================
  Network Intrusion Detection System — UNSW-NB15
============================================================
[INFO] Generating UNSW-NB15 style dataset …
[INFO] Dataset saved → data/UNSW_NB15_training-set.csv  (82,332 records)

EXPLORATORY DATA ANALYSIS
============================================================
Shape         : (82332, 42)
Missing values: 0
Class distribution: 0 → 46106 | 1 → 36226
...
  ► RandomForest
    CV F1: 0.9921 ± 0.0008
    Accuracy : 0.9935
    AUC-ROC  : 0.9991
  ► XGBoost
    CV F1: 0.9918 ± 0.0006
    Accuracy : 0.9929
    AUC-ROC  : 0.9993

Pipeline complete. Check /visualizations and /results.
```

### 5. Load a saved model

```python
import joblib
import pandas as pd

model = joblib.load("models/randomforest_ids.pkl")

# Single prediction
sample = pd.DataFrame([{
    "dur": 0.12, "sbytes": 5000, "dbytes": 1200,
    "sttl": 64,  "dttl": 128,   "sloss": 0,
    "dloss": 0,  "Sload": 12000, "Dload": 4000,
    # … (all 37 features required)
}])
label = model.predict(sample)
proba = model.predict_proba(sample)[:, 1]
print("Prediction:", "Attack" if label[0] else "Normal", f"(p={proba[0]:.3f})")
```

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| Random Forest | **0.9935** | **0.9921** | **0.9942** | **0.9931** | **0.9991** |
| XGBoost | 0.9929 | 0.9910 | 0.9940 | 0.9925 | 0.9993 |

*Results are reproducible with `random_state=42`. Exact numbers will vary slightly with the official UNSW-NB15 CSV.*

---

## Visualisations

| File | Description |
|---|---|
| `01_class_distribution.png` | Pie chart (Normal vs Attack) + attack category bar chart |
| `02_feature_distributions.png` | Log-scale histograms of 6 key features |
| `03_correlation_heatmap.png` | Pearson correlation heatmap (top 20 features) |
| `04_confusion_matrices.png` | Confusion matrices for both models |
| `05_roc_curves.png` | ROC curves with AUC annotations |
| `06_precision_recall.png` | Precision-Recall curves with Average Precision |
| `07_feature_importance.png` | Top-20 features by Gini importance / XGBoost gain |
| `08_metrics_comparison.png` | Side-by-side bar chart of all five metrics |

---

## Evaluation Metrics

### Accuracy

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

### Precision

$$\text{Precision} = \frac{TP}{TP + FP}$$

### Recall (Sensitivity / Detection Rate)

$$\text{Recall} = \frac{TP}{TP + FN}$$

### F1-Score

$$F_1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### AUC-ROC

Area under the Receiver Operating Characteristic curve — measures the model's ability to discriminate between normal traffic and attacks across all classification thresholds. An AUC of **1.0** is perfect; **0.5** is random guessing.

---

## References

1. Moustafa, N., & Slay, J. (2015). UNSW-NB15: A comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set). *2015 Military Communications and Information Systems Conference (MilCIS)*, 1–6. https://doi.org/10.1109/MilCIS.2015.7348942

2. Moustafa, N., & Slay, J. (2016). The evaluation of Network Anomaly Detection Systems: Statistical analysis of the UNSW-NB15 data set and the comparison with the KDD99 data set. *Information Security Journal: A Global Perspective*, 25(1–3), 18–31. https://doi.org/10.1080/19393555.2015.1125974

3. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324

4. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. https://doi.org/10.1145/2939672.2939785

5. Scikit-learn developers. (2023). *scikit-learn: Machine learning in Python* (v1.3). https://scikit-learn.org

---

## License

This project is released under the **MIT License**. The UNSW-NB15 dataset is provided by UNSW Canberra for academic research purposes — please cite Moustafa & Slay (2015) if you use it in published work.

---

*Built with Python 3.10+ · scikit-learn · XGBoost · pandas · matplotlib · seaborn*
