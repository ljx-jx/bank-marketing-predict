"""Integration tests for the Streamlit application using AppTest."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_app_runs_without_error():
    """Given the main app script, running it with AppTest completes without error."""
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        pytest.skip("streamlit.testing.v1 not available")

    at = AppTest.from_file(str(Path(__file__).parent.parent / "src" / "app.py"))
    at.run()

    assert not at.exception
    assert len(at.title) >= 1
    assert "银行营销" in at.title[0].value


def test_app_has_three_metrics():
    """Given the app running, three KPI metric cards are shown."""
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        pytest.skip("streamlit.testing.v1 not available")

    at = AppTest.from_file(str(Path(__file__).parent.parent / "src" / "app.py"))
    at.run()

    assert not at.exception
    assert len(at.metric) == 3
