# Network Intrusion Detection System — UNSW-NB15

This repository is a reproducible research prototype for binary network-intrusion classification. It trains Random Forest and XGBoost models independently on the official UNSW-NB15 training split and evaluates them once on the publisher's held-out testing split.

See [`PROVENANCE.md`](PROVENANCE.md) for the data lineage, experiment record, authorship boundary, and historical-artifact note.

## Dataset and provenance

UNSW-NB15 was created by the Australian Centre for Cyber Security at UNSW Canberra. The publisher reports 175,341 records in the training set and 82,332 records in the testing set. Obtain both CSVs from the [official UNSW-NB15 dataset page](https://research.unsw.edu.au/projects/unsw-nb15-dataset) and review its academic/commercial-use terms before redistribution.

Place the files here (they are intentionally ignored by Git):

```text
data/UNSW_NB15_training-set.csv
data/UNSW_NB15_testing-set.csv
```

The downloader validates the schema and split size, downloads atomically, and can verify a publisher-provided SHA-256 digest. It never generates replacement records:

```bash
python src/download_dataset.py \
  --url "<official-training-csv-url>" \
  --output data/UNSW_NB15_training-set.csv \
  --expected-rows 175341 \
  --expected-sha256 "<publisher-sha256>"
```

Run the same command for the testing CSV with `--output data/UNSW_NB15_testing-set.csv` and `--expected-rows 82332`. If the source does not publish a digest, the script prints the downloaded digest for your records; do not describe the file as checksum-verified in a report.

## Methodology

1. The official training and testing CSVs are loaded separately and checked for the expected schema, binary `label` values, and row counts.
2. The target is `label` (`0 = normal`, `1 = attack`). Predictors are 39 numeric traffic features plus `proto`, `state`, and `service`. `id` and `attack_cat` are not predictors.
3. Missing numeric values are median-imputed; missing categorical values are mode-imputed and one-hot encoded. These transformations are inside each scikit-learn `Pipeline`, so they are fitted only on the training fold and are persisted with the model.
4. Five-fold stratified cross-validation is used on the training split for an F1 estimate. Final metrics are computed once on the untouched official testing split. No random re-split of the publisher's data is performed and no outlier cap is fitted outside the pipeline.
5. Random Forest and XGBoost are reported as separate baselines. They are not called an ensemble because their probabilities are not combined.

Reported metrics include accuracy, precision, recall, F1, ROC-AUC, average precision, and false-alarm rate. Generated metrics, plots, and model files are local outputs and are not committed by default.

## Project structure

```text
.
├── src/
│   ├── ids_pipeline.py       # validation, EDA, training, evaluation, plots
│   └── download_dataset.py   # fail-closed official CSV downloader
├── data/                     # ignored local dataset files
├── models/                   # ignored persisted pipelines
├── results/                  # ignored metrics and reports
├── visualizations/           # ignored generated figures
├── requirements.txt
└── tests/
```

## Quick start

```bash
git clone https://github.com/seyakin/ml-intrusion-detection-system.git
cd ml-intrusion-detection-system
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# Obtain and validate both official CSVs as described above.
python src/ids_pipeline.py
```

To use a different validated location:

```bash
python src/ids_pipeline.py \
  --train-path /path/to/UNSW_NB15_training-set.csv \
  --test-path /path/to/UNSW_NB15_testing-set.csv
```

The saved `models/*_ids.joblib` files include preprocessing. For inference, pass a DataFrame containing the same 42 predictor columns to the loaded pipeline; do not manually encode categories or scale values.

## Reproducibility and limitations

- The random seed is fixed at 42 for model and cross-validation splits, but exact results can still vary across hardware and library versions.
- Dependency ranges are recorded in `requirements.txt`; use a lock file or capture `pip freeze` for a fully pinned experiment.
- UNSW-NB15 is a benchmark, not a guarantee of performance on production traffic. Thresholds, drift, class imbalance, and operational false positives require separate validation.
- This repository makes no claim of production deployment, customer adoption, or regulatory certification.
- Dataset permissions are separate from the application-code license. No application-code license is declared in this repository; add one only when the copyright holder has approved its terms.

## Reference

N. Moustafa and J. Slay, “UNSW-NB15: a comprehensive data set for network intrusion detection systems,” *MilCIS 2015*. The official dataset page lists the recommended citations and access conditions.
