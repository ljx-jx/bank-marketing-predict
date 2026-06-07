"""Preprocessing utilities for bank marketing data."""


def get_numeric_columns(df):
    """Return list of numeric column names."""
    return df.select_dtypes(include=["int64", "float64"]).columns.tolist()


def get_categorical_columns(df):
    """Return list of categorical column names."""
    return df.select_dtypes(include=["object", "category"]).columns.tolist()
