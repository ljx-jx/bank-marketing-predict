"""Tests for Streamlit pages."""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_analysis_page_imports():
    """Verify the data analysis page module is importable."""
    mod = importlib.import_module("src.pages.01_data_analysis")
    assert mod is not None


def test_prediction_page_imports():
    """Verify the prediction page module is importable."""
    mod = importlib.import_module("src.pages.02_prediction")
    assert mod is not None
