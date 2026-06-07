"""Data loading utilities for the bank marketing dataset."""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

# Columns that are not features
ID_COL = "id"
TARGET_COL = "subscribe"

# Expected input features (both numeric and categorical)
FEATURE_COLS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp_var_rate",
    "cons_price_index",
    "cons_conf_index",
    "lending_rate3m",
    "nr_employed",
]

# Numeric feature columns
NUMERIC_COLS = [
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

# Categorical feature columns
CATEGORICAL_COLS = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]


def load_csv(filepath: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with the loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    return pd.read_csv(path)


def load_train_data(
    train_path: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load labeled training data and split into features and target.

    Args:
        train_path: Path to train.csv (must have 'subscribe' column).

    Returns:
        Tuple of (X_train, y_train). y values are binary: 1 for 'yes', 0 for 'no'.
    """
    train_df = load_csv(train_path)

    if TARGET_COL not in train_df.columns:
        raise ValueError(f"Training data must contain '{TARGET_COL}' column.")

    X_train = train_df.drop(columns=[ID_COL, TARGET_COL])
    y_train = (train_df[TARGET_COL] == "yes").astype(int)

    _validate_features(X_train)
    return X_train, y_train


def load_test_data(test_path: str) -> pd.DataFrame:
    """Load unlabeled test data (features only, no target).

    Args:
        test_path: Path to test.csv (may or may not have 'subscribe' column).

    Returns:
        Feature DataFrame X_test (id and subscribe columns dropped if present).
    """
    test_df = load_csv(test_path)
    drop_cols = [ID_COL]
    if TARGET_COL in test_df.columns:
        drop_cols.append(TARGET_COL)
    X_test = test_df.drop(columns=drop_cols, errors="ignore")
    _validate_features(X_test)
    return X_test


def load_bank_data(
    train_path: str, test_path: str
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, Optional[pd.Series]]:
    """Load bank marketing train/test data and split into features and targets.

    The 'id' column is dropped from features as it is a row identifier.
    If test.csv does not contain the 'subscribe' column, y_test will be None.

    Args:
        train_path: Path to train.csv.
        test_path: Path to test.csv.

    Returns:
        Tuple of (X_train, y_train, X_test, y_test).
        y values are binary: 1 for 'yes', 0 for 'no'.
        y_test is None if test data has no target column.
    """
    X_train, y_train = load_train_data(train_path)

    test_df = load_csv(test_path)
    if TARGET_COL in test_df.columns:
        X_test, y_test = load_train_data(
            test_path
        )  # test_path has labels, reuse load_train_data
        return X_train, y_train, X_test, y_test

    X_test = load_test_data(test_path)
    return X_train, y_train, X_test, None


def _validate_features(df: pd.DataFrame) -> None:
    """Ensure all expected feature columns are present.

    Args:
        df: Feature DataFrame to validate.

    Raises:
        ValueError: If any expected column is missing.
    """
    missing = set(FEATURE_COLS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing feature columns: {sorted(missing)}. "
            f"Expected {len(FEATURE_COLS)} features, got {len(df.columns)}."
        )
