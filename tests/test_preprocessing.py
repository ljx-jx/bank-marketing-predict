"""Tests for preprocessing module."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.preprocessing import (
    build_preprocessing_pipeline,
    get_categorical_columns,
    get_feature_names,
    get_numeric_columns,
)


def _make_mixed_df() -> pd.DataFrame:
    """Return a small mixed-type DataFrame for testing."""
    return pd.DataFrame(
        {
            "age": [30, 45, 52],
            "salary": [50000.0, 60000.0, 75000.0],
            "job": ["admin", "manager", "worker"],
            "marital": ["single", "married", "divorced"],
            "subscribed": ["yes", "no", "no"],
        }
    )


class TestColumnDetection:
    """Tests for get_numeric_columns and get_categorical_columns."""

    def test_numeric_returns_expected(self):
        """Given a mixed DataFrame, returns numeric columns."""
        df = _make_mixed_df()
        cols = get_numeric_columns(df)
        assert set(cols) == {"age", "salary"}

    def test_numeric_empty_for_all_strings(self):
        """Given only string columns, returns empty list."""
        df = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
        assert get_numeric_columns(df) == []

    def test_categorical_returns_expected(self):
        """Given a mixed DataFrame, returns object/string columns."""
        df = _make_mixed_df()
        cols = get_categorical_columns(df)
        assert set(cols) == {"job", "marital", "subscribed"}

    def test_categorical_empty_for_all_numeric(self):
        """Given only numeric columns, returns empty list."""
        df = pd.DataFrame({"x": [1, 2], "y": [3.0, 4.0]})
        assert get_categorical_columns(df) == []


class TestPreprocessingPipeline:
    """Tests for build_preprocessing_pipeline and related functions."""

    def test_pipeline_fit_transform_no_error(self, sample_df):
        """Given sample data, fit_transform completes without error."""
        X = sample_df.drop(columns=["id", "subscribe"])
        y = (sample_df["subscribe"] == "yes").astype(int)

        num_cols = get_numeric_columns(X)
        cat_cols = get_categorical_columns(X)

        preprocessor = build_preprocessing_pipeline(num_cols, cat_cols)
        X_transformed = preprocessor.fit_transform(X, y)

        assert X_transformed is not None
        assert isinstance(X_transformed, np.ndarray)
        assert X_transformed.shape[0] == len(X)

    def test_pipeline_output_is_numeric(self, sample_df):
        """Given sample data, transformed output is all numeric (no NaN)."""
        X = sample_df.drop(columns=["id", "subscribe"])
        y = (sample_df["subscribe"] == "yes").astype(int)

        num_cols = get_numeric_columns(X)
        cat_cols = get_categorical_columns(X)

        preprocessor = build_preprocessing_pipeline(num_cols, cat_cols)
        X_transformed = preprocessor.fit_transform(X, y)

        assert np.isfinite(X_transformed).all()
        assert not np.isnan(X_transformed).any()

    def test_pipeline_handles_unknown_category(self, sample_df):
        """Given an unseen category at transform time, no error is raised."""
        X = sample_df.drop(columns=["id", "subscribe"])
        y = (sample_df["subscribe"] == "yes").astype(int)

        num_cols = get_numeric_columns(X)
        cat_cols = get_categorical_columns(X)

        preprocessor = build_preprocessing_pipeline(num_cols, cat_cols)
        preprocessor.fit(X, y)

        # Create new row with unseen category
        new_row = X.iloc[[0]].copy()
        new_row["job"] = "astronaut"  # unseen category
        new_row["marital"] = "unknown"  # unseen in sample

        X_t = preprocessor.transform(new_row)
        assert np.isfinite(X_t).all()

    def test_get_feature_names_returns_correct_count(self, sample_df):
        """Given a fitted pipeline, get_feature_names returns expected count."""
        X = sample_df.drop(columns=["id", "subscribe"])
        y = (sample_df["subscribe"] == "yes").astype(int)

        num_cols = get_numeric_columns(X)
        cat_cols = get_categorical_columns(X)

        preprocessor = build_preprocessing_pipeline(num_cols, cat_cols)
        preprocessor.fit(X, y)

        X_t = preprocessor.transform(X)
        names = get_feature_names(preprocessor)

        assert len(names) == X_t.shape[1]

    def test_pipeline_with_missing_values(self):
        """Given data with NaN values, pipeline handles them gracefully."""
        df = pd.DataFrame(
            {
                "age": [30.0, np.nan, 52.0, 38.0],
                "job": ["admin", "manager", None, "admin"],
                "target": [0, 1, 0, 1],
            }
        )
        X = df[["age", "job"]]
        y = df["target"]

        preprocessor = build_preprocessing_pipeline(["age"], ["job"])
        X_t = preprocessor.fit_transform(X, y)

        # 1 numeric + 3 one-hot columns (admin, manager, missing → 3 unique values)
        assert X_t.shape == (4, 4)
        assert np.isfinite(X_t).all()
