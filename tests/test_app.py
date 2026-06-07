"""Tests for the Streamlit application entry point."""

import sys
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_src_package_imports():
    """Verify all core packages are importable."""
    import src
    import src.pages
    import src.model
    import src.utils

    assert src is not None
    assert src.pages is not None
    assert src.model is not None
    assert src.utils is not None


def test_app_module_has_main():
    """Verify app module exposes a main() function."""
    from src.app import main

    assert callable(main)


def test_data_loader_import():
    """Verify data_loader module is importable and has load_csv."""
    from src.utils.data_loader import load_csv

    assert callable(load_csv)


def test_preprocessing_import():
    """Verify preprocessing module is importable."""
    from src.utils.preprocessing import get_numeric_columns, get_categorical_columns

    assert callable(get_numeric_columns)
    assert callable(get_categorical_columns)


def test_train_module_import():
    """Verify train module exposes train_model function."""
    from src.model.train import train_model

    assert callable(train_model)


def test_predict_module_import():
    """Verify predict module exposes predict function."""
    from src.model.predict import predict

    assert callable(predict)
