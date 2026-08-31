# Provenance and methodology record

This file records what is in the repository, where inputs come from, and what the pipeline does. It is intended to make the project auditable without implying work or results that have not been verified.

## Source data

- Dataset: UNSW-NB15, published by the Australian Centre for Cyber Security at UNSW Canberra.
- Official split used by this project: 175,341 training records and 82,332 testing records.
- Source and citation guidance: <https://research.unsw.edu.au/projects/unsw-nb15-dataset>.
- The CSVs are not committed. They are large, subject to the publisher's access and use terms, and must be obtained by the person running the experiment.
- `src/download_dataset.py` validates the required schema and row count, optionally checks SHA-256, downloads to a temporary file, and replaces the destination only after validation. 

## Experimental procedure

1. Load the publisher's training and testing CSVs independently.
2. Validate the schema, labels, and official split sizes.
3. Use 39 numeric traffic features and the categorical features `proto`, `state`, and `service`. Exclude `id`, `attack_cat`, and `label` from predictors.
4. Fit median imputation, mode imputation, and one-hot encoding inside a scikit-learn `Pipeline`. This prevents fitting transformations on the test set or on a held-out cross-validation fold.
5. Fit Random Forest and XGBoost as separate baselines. Five-fold stratified cross-validation is performed on the training split for F1; final metrics are measured once on the untouched testing split.
6. Save the complete preprocessing-plus-classifier pipeline, metrics, reports, and plots as ignored local outputs.

The reported metrics are accuracy, precision, recall, F1, ROC-AUC, average precision, and false-alarm rate. No metric values are stored in the repository because they depend on the exact dataset files and dependency versions used by the experiment.

## Historical artifacts

Earlier public history included an archived zip and synthetic-data-derived outputs. This cleanup removes those artifacts from the current working tree and prevents their recreation, but it does not rewrite the existing public commit history. Do not present historical synthetic metrics as UNSW-NB15 benchmark results. If a clean public provenance is required, publish this reviewed tree as a new repository or use a separately approved history rewrite after preserving an archive and documenting it.
