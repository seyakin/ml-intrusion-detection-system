"""Train and evaluate intrusion-detection models on the official UNSW-NB15 split.

The repository deliberately does not manufacture a substitute dataset.  The
training and testing CSVs must be obtained from the dataset publisher and are
validated before they enter the pipeline.  All learned preprocessing is kept
inside each persisted model so that inference uses the same transformations as
training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBClassifier
except ImportError:  # Keep schema validation usable before dependencies are installed.
    XGBClassifier = None  # type: ignore[assignment,misc]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
VIZ_DIR = PROJECT_ROOT / "visualizations"
MODEL_DIR = PROJECT_ROOT / "models"

TRAIN_FILENAME = "UNSW_NB15_training-set.csv"
TEST_FILENAME = "UNSW_NB15_testing-set.csv"
EXPECTED_TRAIN_ROWS = 175_341
EXPECTED_TEST_ROWS = 82_332
RANDOM_STATE = 42

# The publisher's CSV contains an optional id column, 39 numeric predictors,
# three categorical predictors, attack_cat, and the binary label.
NUMERIC_FEATURES = [
    "dur",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "swin",
    "stcpb",
    "dtcpb",
    "dwin",
    "tcprtt",
    "synack",
    "ackdat",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "ct_srv_src",
    "ct_state_ttl",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports",
]
CATEGORICAL_FEATURES = ["proto", "state", "service"]
REQUIRED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# A few releases of the dataset use these historical spellings.  Normalising
# them makes the input contract explicit without silently changing values.
COLUMN_ALIASES = {
    "smeansz": "smean",
    "dmeansz": "dmean",
    "res_bdy_len": "response_body_len",
}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with canonical lower-case column names."""

    renamed = [str(column).strip().lower() for column in df.columns]
    if len(set(renamed)) != len(renamed):
        raise ValueError("The dataset contains duplicate column names.")
    result = df.copy()
    result.columns = renamed

    for old_name, new_name in COLUMN_ALIASES.items():
        if old_name in result.columns and new_name in result.columns:
            raise ValueError(
                f"Dataset contains both '{old_name}' and '{new_name}'; "
                "remove the duplicate spelling."
            )
        if old_name in result.columns:
            result = result.rename(columns={old_name: new_name})
    return result


def _validate_dataset(
    df: pd.DataFrame,
    *,
    source: str,
    expected_rows: int | None = None,
) -> pd.DataFrame:
    """Validate the publisher schema and return a canonicalised dataframe."""

    result = _normalise_columns(df)
    missing = [column for column in REQUIRED_FEATURES + ["label"] if column not in result]
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")
    if expected_rows is not None and len(result) != expected_rows:
        raise ValueError(
            f"{source} has {len(result):,} rows; expected {expected_rows:,} "
            "for the official UNSW-NB15 split."
        )

    labels = pd.to_numeric(result["label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError(f"{source} must contain a binary label column with values 0 and 1.")
    result["label"] = labels.astype("int8")

    for column in NUMERIC_FEATURES:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[REQUIRED_FEATURES].isna().all(axis=None):
        raise ValueError(f"{source} contains no usable feature values.")
    return result


def load_datasets(
    train_path: str | Path | None = None,
    test_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the official UNSW-NB15 training and testing CSVs."""

    train_file = Path(train_path) if train_path else DATA_DIR / TRAIN_FILENAME
    test_file = Path(test_path) if test_path else DATA_DIR / TEST_FILENAME
    missing = [str(path) for path in (train_file, test_file) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Official UNSW-NB15 CSVs were not found: "
            + ", ".join(missing)
            + ". Download them from the publisher and place them under data/."
        )

    train_df = _validate_dataset(
        pd.read_csv(train_file, low_memory=False),
        source=str(train_file),
        expected_rows=EXPECTED_TRAIN_ROWS,
    )
    test_df = _validate_dataset(
        pd.read_csv(test_file, low_memory=False),
        source=str(test_file),
        expected_rows=EXPECTED_TEST_ROWS,
    )
    print(f"[DATA] Train: {train_file} ({len(train_df):,} rows)")
    print(f"[DATA] Test : {test_file} ({len(test_df):,} rows)")
    return train_df, test_df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select predictors and the binary target without fitting transformations."""

    features = df[REQUIRED_FEATURES].copy()
    target = df["label"].astype("int8")
    return features, target


def build_preprocessor() -> ColumnTransformer:
    """Create leakage-safe preprocessing for the tree-based estimators."""

    numeric = Pipeline(
        steps=[("impute", SimpleImputer(strategy="median"))]
    )
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_models(random_state: int = RANDOM_STATE) -> dict[str, Pipeline]:
    """Build independently evaluated models with preprocessing attached."""

    if XGBClassifier is None:
        raise RuntimeError(
            "xgboost is required for model training; install dependencies with "
            "'python -m pip install -r requirements.txt'."
        )
    return {
        "random_forest": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=20,
                        min_samples_split=5,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=200,
                        max_depth=8,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        tree_method="hist",
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def run_eda(df: pd.DataFrame, output_dir: str | Path = VIZ_DIR) -> None:
    """Generate descriptive plots from the training split only."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    print("\n[EDA] Training-set shape:", df.shape)
    print("[EDA] Missing values:", int(df.isna().sum().sum()))
    print("[EDA] Class distribution:")
    print(df["label"].value_counts().sort_index().to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    class_counts = df["label"].value_counts().sort_index()
    axes[0].bar(["Normal (0)", "Attack (1)"], [class_counts.get(0, 0), class_counts.get(1, 0)])
    axes[0].set_title("UNSW-NB15 training labels")
    axes[0].set_ylabel("Records")
    if "attack_cat" in df.columns:
        categories = df.loc[df["label"] == 1, "attack_cat"].astype(str).value_counts()
        axes[1].barh(categories.index, categories.values)
        axes[1].set_title("Attack categories (training set)")
        axes[1].invert_yaxis()
    else:
        axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(output / "01_class_distribution.png", dpi=150)
    plt.close(fig)

    selected = ["dur", "sbytes", "dbytes", "sload", "dload", "spkts"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, feature in zip(axes.flat, selected):
        for label, name in ((0, "Normal"), (1, "Attack")):
            values = np.log1p(df.loc[df["label"] == label, feature].clip(lower=0).dropna())
            axis.hist(values, bins=50, alpha=0.5, label=name, density=True)
        axis.set_title(f"log1p({feature})")
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "02_feature_distributions.png", dpi=150)
    plt.close(fig)

    correlation = df[NUMERIC_FEATURES].corr(numeric_only=True)
    fig, axis = plt.subplots(figsize=(14, 10))
    image = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    axis.set_title("Numeric feature correlation (training set)")
    axis.set_xticks(range(len(NUMERIC_FEATURES)), NUMERIC_FEATURES, rotation=90, fontsize=7)
    axis.set_yticks(range(len(NUMERIC_FEATURES)), NUMERIC_FEATURES, fontsize=7)
    fig.colorbar(image, ax=axis, shrink=0.7)
    fig.tight_layout()
    fig.savefig(output / "03_correlation_heatmap.png", dpi=150)
    plt.close(fig)


def _metrics(
    y_true: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, _, _ = matrix.ravel()
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "false_alarm_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
    }


def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    cv_folds: int = 5,
    model_dir: str | Path = MODEL_DIR,
) -> dict[str, dict[str, Any]]:
    """Fit on the official training set and evaluate once on the test set."""

    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2")
    x_train, y_train = prepare_features(train_df)
    x_test, y_test = prepare_features(test_df)
    output = Path(model_dir)
    output.mkdir(parents=True, exist_ok=True)
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    results: dict[str, dict[str, Any]] = {}
    for name, model in build_models().items():
        print(f"\n[MODEL] {name}")
        cv_scores = cross_val_score(
            model,
            x_train,
            y_train,
            scoring="f1",
            cv=splitter,
            n_jobs=1,
        )
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]
        metrics = _metrics(y_test, predictions, probabilities)
        metrics["cv_f1_mean"] = float(cv_scores.mean())
        metrics["cv_f1_std"] = float(cv_scores.std())
        print(f"[MODEL] CV F1: {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}")
        print(f"[MODEL] Test F1: {metrics['f1']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
        artifact = output / f"{name}_ids.joblib"
        joblib.dump(model, artifact)
        results[name] = {
            "model": model,
            "metrics": metrics,
            "y_true": y_test,
            "predictions": predictions,
            "probabilities": probabilities,
            "classification_report": classification_report(
                y_test, predictions, target_names=["Normal", "Attack"], zero_division=0
            ),
            "artifact": artifact,
        }
        print(f"[MODEL] Saved pipeline: {artifact}")
    return results


def save_results(results: dict[str, dict[str, Any]], output_dir: str | Path = RESULTS_DIR) -> None:
    """Write metrics and reports generated from the official test split."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = [{"model": name, **data["metrics"]} for name, data in results.items()]
    pd.DataFrame(rows).to_csv(output / "model_metrics.csv", index=False)
    with (output / "classification_report.txt").open("w", encoding="utf-8") as report:
        report.write("UNSW-NB15 official test split\n")
        report.write("================================\n\n")
        for name, data in results.items():
            report.write(f"{name}\n{'-' * len(name)}\n")
            report.write(data["classification_report"])
            report.write("\n\n")


def plot_results(results: dict[str, dict[str, Any]], output_dir: str | Path = VIZ_DIR) -> None:
    """Create evaluation plots from predictions on the official test split."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5), squeeze=False)
    for axis, (name, data) in zip(axes.flat, results.items()):
        matrix = confusion_matrix(data["y_true"], data["predictions"], labels=[0, 1])
        axis.imshow(matrix, cmap="Blues")
        axis.set_title(f"{name}: confusion matrix")
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.set_xticks([0, 1], ["Normal", "Attack"])
        axis.set_yticks([0, 1], ["Normal", "Attack"])
        for (row, column), value in np.ndenumerate(matrix):
            axis.text(column, row, f"{value:,}", ha="center", va="center")
    figure.tight_layout()
    figure.savefig(output / "04_confusion_matrices.png", dpi=150)
    plt.close(figure)

    figure, (roc_axis, pr_axis) = plt.subplots(1, 2, figsize=(13, 5))
    for name, data in results.items():
        fpr, tpr, _ = roc_curve(data["y_true"], data["probabilities"])
        precision, recall, _ = precision_recall_curve(data["y_true"], data["probabilities"])
        roc_axis.plot(fpr, tpr, label=f"{name} (AUC={data['metrics']['roc_auc']:.3f})")
        pr_axis.plot(recall, precision, label=f"{name} (AP={data['metrics']['average_precision']:.3f})")
    roc_axis.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    roc_axis.set(title="ROC curve — official test split", xlabel="False-positive rate", ylabel="True-positive rate")
    pr_axis.set(title="Precision-recall — official test split", xlabel="Recall", ylabel="Precision")
    roc_axis.legend()
    pr_axis.legend()
    figure.tight_layout()
    figure.savefig(output / "05_evaluation_curves.png", dpi=150)
    plt.close(figure)

    labels = list(results)
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc", "false_alarm_rate"]
    values = np.array([[data["metrics"][metric] for metric in metric_names] for data in results.values()])
    figure, axis = plt.subplots(figsize=(12, 5))
    width = 0.8 / max(len(labels), 1)
    x = np.arange(len(metric_names))
    for index, (label, row) in enumerate(zip(labels, values)):
        axis.bar(x + index * width, row, width, label=label)
    axis.set_xticks(x + width * (len(labels) - 1) / 2, metric_names, rotation=25, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_title("Model metrics — official test split")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "06_metrics_comparison.png", dpi=150)
    plt.close(figure)

    for name, data in results.items():
        classifier = data["model"].named_steps["classifier"]
        preprocessor = data["model"].named_steps["preprocess"]
        feature_names = preprocessor.get_feature_names_out()
        importances = getattr(classifier, "feature_importances_", None)
        if importances is None:
            continue
        top = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(20).sort_values()
        figure, axis = plt.subplots(figsize=(10, 7))
        top.plot.barh(ax=axis)
        axis.set_title(f"Top transformed features — {name}")
        axis.set_xlabel("Feature importance")
        figure.tight_layout()
        figure.savefig(output / f"07_feature_importance_{name}.png", dpi=150)
        plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=None)
    parser.add_argument("--test-path", type=Path, default=None)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args(argv)

    try:
        train_df, test_df = load_datasets(args.train_path, args.test_path)
        run_eda(train_df)
        results = train_and_evaluate(train_df, test_df, cv_folds=args.cv_folds)
        save_results(results)
        plot_results(results)
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    print("\n[DONE] Results, plots, and model pipelines are available locally.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
