"""Prediction logic for the bank marketing subscription model.

US-4: Load a trained pipeline and predict on a single customer's features.
"""

import sys
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

DEFAULT_MODEL_PATH = (
    Path(__file__).parent.parent.parent / "models" / "model_pipeline.pkl"
)


def load_model(model_path: str | None = None):
    """Load a trained sklearn Pipeline from disk.

    Args:
        model_path: Path to the saved model file. Defaults to models/model_pipeline.pkl.

    Returns:
        The loaded sklearn Pipeline.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}. Please run `python -m src.model.train` first."
        )
    return joblib.load(path)


def predict(
    model_path: str | None,
    features: Dict[str, object],
) -> Dict[str, object]:
    """Predict subscription probability for a single customer.

    Args:
        model_path: Path to the saved model file. If None, uses default.
        features: Dict mapping feature name to value. Must include all 20
                  expected feature columns.

    Returns:
        Dict with:
            - 'subscribe': bool — True if predicted to subscribe.
            - 'probability': float — predicted probability (0.0 to 1.0).
            - 'label': str — 'yes' or 'no' for display.

    Raises:
        FileNotFoundError: If model file is missing.
        ValueError: If features are missing or invalid.
    """
    pipeline = load_model(model_path)

    # Convert single sample to a DataFrame with correct column order
    expected_cols = (
        pipeline.feature_names_in_
        if hasattr(pipeline, "feature_names_in_")
        else list(features.keys())
    )

    try:
        input_df = pd.DataFrame([features])
    except Exception as exc:
        raise ValueError(f"Invalid feature values: {exc}") from exc

    # Ensure all expected columns are present (fill missing with 0 / empty string)
    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0 if col in _numeric_cols() else ""

    # Reorder to match model expectations
    input_df = input_df[expected_cols]

    # Predict
    proba = pipeline.predict_proba(input_df)[0, 1]
    pred_label = "yes" if proba >= 0.5 else "no"

    return {
        "subscribe": bool(proba >= 0.5),
        "probability": round(float(proba), 4),
        "label": pred_label,
    }


def _numeric_cols():
    """Return list of numeric feature columns (avoid circular import)."""
    return [
        "age",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "emp_var_rate",
        "cons_price_index",
        "cons_conf_index",
        "lending_rate3m",
        "nr_employed",
    ]
