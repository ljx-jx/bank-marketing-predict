"""Tests for data_loader module."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loader import (
    FEATURE_COLS,
    ID_COL,
    TARGET_COL,
    load_bank_data,
    load_csv,
)


class TestLoadCsv:
    """Tests for load_csv function."""

    def test_loads_dataframe(self, sample_csv_path):
        """Given a valid CSV, load_csv returns a DataFrame."""
        df = load_csv(sample_csv_path)
        assert isinstance(df, pd.DataFrame)

    def test_preserves_data(self, sample_csv_path):
        """Given a CSV with known values, load_csv preserves them."""
        df = load_csv(sample_csv_path)
        assert len(df) == 6
        assert "age" in df.columns
        assert "subscribe" in df.columns

    def test_raises_on_missing_file(self):
        """Given a non-existent path, load_csv raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_csv("nonexistent/path.csv")


class TestLoadBankData:
    """Tests for load_bank_data with real data."""

    def test_returns_correct_shapes(self, sample_csv_path):
        """Given train/test CSVs, returns four objects with correct shapes."""
        X_train, y_train, X_test, y_test = load_bank_data(
            sample_csv_path, sample_csv_path
        )

        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_test, pd.Series)

    def test_id_and_target_removed(self, sample_csv_path):
        """Given CSVs, id and subscribe columns are removed from features."""
        X_train, _, _, _ = load_bank_data(sample_csv_path, sample_csv_path)

        assert ID_COL not in X_train.columns
        assert TARGET_COL not in X_train.columns

    def test_target_is_binary(self, sample_csv_path):
        """Given CSVs, y values are binary (0/1)."""
        _, y_train, _, y_test = load_bank_data(sample_csv_path, sample_csv_path)

        assert y_train.isin([0, 1]).all()
        assert y_test.isin([0, 1]).all()
        assert y_train.dtype == int
        assert y_test.dtype == int

    def test_feature_count(self, sample_csv_path):
        """Given CSVs, X has exactly the expected 20 feature columns."""
        X_train, _, _, _ = load_bank_data(sample_csv_path, sample_csv_path)

        assert list(X_train.columns) == FEATURE_COLS

    def test_loads_real_data(self, real_train_path, real_test_path):
        """Given real data files, load_bank_data works end-to-end."""
        if real_train_path is None or real_test_path is None:
            pytest.skip("Real data files not available")

        X_train, y_train, X_test, y_test = load_bank_data(
            real_train_path, real_test_path
        )

        assert len(X_train) == 22500
        assert len(X_test) == 7500
        assert list(X_train.columns) == FEATURE_COLS
        assert y_train.value_counts().sum() == 22500
        # Class imbalance: yes ~13%
        yes_pct = y_train.mean()
        assert 0.05 < yes_pct < 0.25
        # Real test.csv has no subscribe column → y_test is None
        assert y_test is None
