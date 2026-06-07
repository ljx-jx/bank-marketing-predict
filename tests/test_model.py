"""Tests for model training and prediction modules."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.predict import predict
from src.model.train import train_model


def test_train_model_raises_not_implemented():
    """Given the placeholder train_model, calling it raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="US-3"):
        train_model("data/train.csv", "models/")


def test_predict_raises_not_implemented():
    """Given the placeholder predict, calling it raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="US-4"):
        predict("models/model.pkl", {"age": 30})
