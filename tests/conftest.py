"""Pytest fixtures shared across test modules."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_df():
    """Return a small DataFrame mimicking the bank marketing dataset."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "age": [30, 45, 52, 38, 60, 27],
            "job": [
                "admin.",
                "services",
                "blue-collar",
                "admin.",
                "management",
                "student",
            ],
            "marital": [
                "married",
                "single",
                "divorced",
                "married",
                "married",
                "single",
            ],
            "education": [
                "university.degree",
                "high.school",
                "basic.9y",
                "university.degree",
                "professional.course",
                "high.school",
            ],
            "default": ["no", "no", "no", "unknown", "no", "no"],
            "housing": ["yes", "no", "yes", "yes", "no", "yes"],
            "loan": ["no", "yes", "no", "no", "no", "no"],
            "contact": [
                "cellular",
                "telephone",
                "cellular",
                "cellular",
                "cellular",
                "telephone",
            ],
            "month": ["may", "jun", "jul", "aug", "may", "nov"],
            "day_of_week": ["mon", "tue", "wed", "thu", "fri", "mon"],
            "duration": [120, 300, 150, 450, 90, 200],
            "campaign": [1, 2, 1, 3, 1, 2],
            "pdays": [999, 500, 999, 200, 999, 100],
            "previous": [0, 1, 0, 2, 0, 3],
            "poutcome": [
                "nonexistent",
                "failure",
                "nonexistent",
                "success",
                "nonexistent",
                "success",
            ],
            "emp_var_rate": [-1.8, 1.1, -1.8, 1.4, -1.8, 1.4],
            "cons_price_index": [92.2, 93.5, 91.2, 94.1, 90.8, 95.0],
            "cons_conf_index": [-42.0, -36.0, -44.0, -35.5, -46.0, -35.0],
            "lending_rate3m": [3.5, 4.0, 2.8, 4.5, 2.5, 5.0],
            "nr_employed": [5000.0, 5100.0, 4950.0, 5200.0, 4900.0, 5250.0],
            "subscribe": ["no", "yes", "no", "yes", "no", "yes"],
        }
    )


@pytest.fixture
def sample_csv_path(sample_df):
    """Write the sample DataFrame to a temp CSV and return its path."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8", newline=""
    ) as f:
        sample_df.to_csv(f, index=False)
        tmp_path = Path(f.name)

    yield str(tmp_path)
    tmp_path.unlink(missing_ok=True)


@pytest.fixture
def real_train_path():
    """Return the real train.csv path if available, else None."""
    path = Path(__file__).parent.parent / "data" / "train.csv"
    return str(path) if path.exists() else None


@pytest.fixture
def real_test_path():
    """Return the real test.csv path if available, else None."""
    path = Path(__file__).parent.parent / "data" / "test.csv"
    return str(path) if path.exists() else None
