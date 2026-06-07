"""Data loader utility."""


def load_csv(filepath: str):
    """Load CSV file into a pandas DataFrame."""
    import pandas as pd

    return pd.read_csv(filepath)
