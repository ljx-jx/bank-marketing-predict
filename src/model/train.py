"""Model training script for bank marketing subscription prediction.

US-3: Train a classifier to predict subscribe (yes/no), validate AUC >= 0.70,
and persist the full pipeline (preprocessing + model) to disk.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Ensure src/ is on the path when run as a module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.data_loader import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    load_train_data,
)
from src.utils.preprocessing import build_preprocessing_pipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_TRAIN_PATH = Path(__file__).parent.parent.parent / "data" / "train.csv"
DEFAULT_MODEL_DIR = Path(__file__).parent.parent.parent / "models"
DEFAULT_MODEL_FILE = DEFAULT_MODEL_DIR / "model_pipeline.pkl"
DEFAULT_METRICS_FILE = DEFAULT_MODEL_DIR / "metrics.json"

# Validation config
VAL_SIZE = 0.2
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Model candidates
# ---------------------------------------------------------------------------
def _build_model_pipeline() -> Pipeline:
    """Build a full sklearn Pipeline: preprocessing → classifier.

    Uses RandomForest with class_weight='balanced' to handle the
    ~13% positive class ratio. The preprocessing stage handles
    numeric imputation+scaling and categorical imputation+one-hot encoding.
    """
    preprocessor = build_preprocessing_pipeline(NUMERIC_COLS, CATEGORICAL_COLS)

    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline


def _build_baseline_pipeline() -> Pipeline:
    """Build a Logistic Regression baseline for comparison."""
    preprocessor = build_preprocessing_pipeline(NUMERIC_COLS, CATEGORICAL_COLS)

    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(
    train_path: str,
    model_dir: str,
) -> Dict:
    """Train a classification model and save it to disk.

    Args:
        train_path: Path to train.csv.
        model_dir: Directory to save model_pipeline.pkl and metrics.json.

    Returns:
        Dict with keys: model, auc, accuracy, precision, recall, f1,
                        model_file, metrics_file.
    """
    logger.info("Loading training data from %s", train_path)
    X, y = load_train_data(train_path)
    logger.info("Loaded %d samples, %d features", len(X), X.shape[1])
    logger.info("Positive class ratio: %.2f%%", y.mean() * 100)

    # Split into train/validation since test.csv has no labels
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    logger.info("Train: %d, Validation: %d (stratified)", len(X_train), len(X_val))

    # Train primary model (RandomForest)
    pipeline = _build_model_pipeline()
    logger.info("Training RandomForest pipeline...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    metrics = _evaluate(pipeline, X_val, y_val)
    logger.info(
        "RF   AUC=%.4f  Accuracy=%.4f  Precision=%.4f  Recall=%.4f  F1=%.4f",
        metrics["auc"],
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    )

    # Check AUC threshold
    if metrics["auc"] < 0.70:
        logger.warning(
            "AUC below 0.70 threshold, trying LogisticRegression baseline..."
        )
        lr_pipeline = _build_baseline_pipeline()
        lr_pipeline.fit(X_train, y_train)
        lr_metrics = _evaluate(lr_pipeline, X_val, y_val)
        logger.info(
            "LR   AUC=%.4f  Accuracy=%.4f  Precision=%.4f  Recall=%.4f  F1=%.4f",
            lr_metrics["auc"],
            lr_metrics["accuracy"],
            lr_metrics["precision"],
            lr_metrics["recall"],
            lr_metrics["f1"],
        )

        if lr_metrics["auc"] > metrics["auc"]:
            pipeline = lr_pipeline
            metrics = lr_metrics
            logger.info("Switching to LogisticRegression (higher AUC)")

    if metrics["auc"] < 0.70:
        logger.error(
            "AUC=%.4f still below 0.70 threshold. Model may not meet quality bar.",
            metrics["auc"],
        )

    # Save model and metrics
    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)

    model_file = model_dir_path / "model_pipeline.pkl"
    metrics_file = model_dir_path / "metrics.json"

    joblib.dump(pipeline, model_file)
    logger.info("Model saved to %s", model_file)

    metrics_out = {
        "timestamp": datetime.now().isoformat(),
        "model_type": type(pipeline.named_steps["classifier"]).__name__,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "positive_ratio": float(y.mean()),
        **{k: float(v) for k, v in metrics.items()},
    }
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False)
    logger.info("Metrics saved to %s", metrics_file)

    return {
        "model": pipeline,
        "model_file": str(model_file),
        "metrics_file": str(metrics_file),
        **metrics,
    }


def _evaluate(
    pipeline: Pipeline, X_val: pd.DataFrame, y_val: pd.Series
) -> Dict[str, float]:
    """Compute classification metrics on validation data.

    Args:
        pipeline: Fitted sklearn Pipeline.
        X_val: Validation feature DataFrame.
        y_val: Validation target Series (0/1).

    Returns:
        Dict with auc, accuracy, precision, recall, f1.
    """
    y_pred = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val)[:, 1]

    return {
        "auc": roc_auc_score(y_val, y_proba),
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "f1": f1_score(y_val, y_pred, zero_division=0),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI entry point: python -m src.model.train"""
    train_path = str(DEFAULT_TRAIN_PATH)
    model_dir = str(DEFAULT_MODEL_DIR)

    if not Path(train_path).exists():
        logger.error("Training data not found: %s", train_path)
        sys.exit(1)

    result = train_model(train_path, model_dir)
    logger.info("Training complete. AUC=%.4f", result["auc"])

    if result["auc"] >= 0.70:
        logger.info("SUCCESS: AUC meets the 0.70 threshold.")
    else:
        logger.warning("WARNING: AUC below 0.70 threshold.")


if __name__ == "__main__":
    main()
