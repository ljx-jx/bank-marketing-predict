"""Tests for data_loader module."""

import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loader import load_csv


def test_load_csv_returns_dataframe():
    """Given a valid CSV file, load_csv should return a DataFrame."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        f.write("a,b\n1,2\n3,4\n")
        tmp_path = f.name

    try:
        df = load_csv(tmp_path)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 2
    finally:
        Path(tmp_path).unlink()


def test_load_csv_preserves_data():
    """Given a CSV with known values, load_csv preserves them."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        f.write("name,score\nAlice,85\nBob,92\n")
        tmp_path = f.name

    try:
        df = load_csv(tmp_path)
        assert df.iloc[0]["name"] == "Alice"
        assert df.iloc[0]["score"] == 85
        assert df.iloc[1]["name"] == "Bob"
    finally:
        Path(tmp_path).unlink()
