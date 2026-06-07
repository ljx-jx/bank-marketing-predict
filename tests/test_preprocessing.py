"""Tests for preprocessing module."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.preprocessing import get_categorical_columns, get_numeric_columns


def _make_df():
    """Create a sample DataFrame with numeric and categorical columns."""
    return pd.DataFrame(
        {
            "age": [30, 45, 52],
            "salary": [50000.0, 60000.0, 75000.0],
            "job": ["admin", "manager", "worker"],
            "marital": ["single", "married", "divorced"],
            "subscribed": ["yes", "no", "no"],
        }
    )


def test_get_numeric_columns_returns_expected():
    """Given a mixed DataFrame, get_numeric_columns returns int64/float64 cols."""
    df = _make_df()
    numeric_cols = get_numeric_columns(df)
    assert set(numeric_cols) == {"age", "salary"}


def test_get_numeric_columns_empty_for_all_categorical():
    """Given a DataFrame with only object cols, returns empty list."""
    df = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
    numeric_cols = get_numeric_columns(df)
    assert numeric_cols == []


def test_get_categorical_columns_returns_expected():
    """Given a mixed DataFrame, get_categorical_columns returns object cols."""
    df = _make_df()
    cat_cols = get_categorical_columns(df)
    assert set(cat_cols) == {"job", "marital", "subscribed"}


def test_get_categorical_columns_empty_for_all_numeric():
    """Given a DataFrame with only numeric cols, returns empty list."""
    df = pd.DataFrame({"x": [1, 2], "y": [3.0, 4.0]})
    cat_cols = get_categorical_columns(df)
    assert cat_cols == []
