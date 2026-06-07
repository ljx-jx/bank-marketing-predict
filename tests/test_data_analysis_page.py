"""Tests for the data analysis page (US-2)."""

import importlib
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Pages with numeric prefixes must be imported via importlib
PAGE_MODULE = "src.pages.01_data_analysis"


def _make_sample_csv(tmp_dir: str) -> str:
    """Create a minimal train.csv for testing and return its path."""
    df = pd.DataFrame(
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
    csv_path = Path(tmp_dir) / "train.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


class TestDataAnalysisPage:
    """Integration tests using Streamlit AppTest."""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch, tmp_path):
        """Create temp data and set BANK_TRAIN_PATH env var."""
        self.csv_path = _make_sample_csv(str(tmp_path))
        monkeypatch.setenv("BANK_TRAIN_PATH", self.csv_path)
        self.page_mod = importlib.import_module(PAGE_MODULE)
        self.page_mod._load_data.clear()

    def test_page_renders_without_error(self):
        """AC1: Page loads and renders without exception."""
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            pytest.skip("streamlit.testing.v1 not available")

        at = AppTest.from_file(
            str(Path(__file__).parent.parent / "src" / "pages" / "01_data_analysis.py")
        )
        at.run()

        assert not at.exception

    def test_page_shows_title(self):
        """Given the page running, the title is displayed."""
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            pytest.skip("streamlit.testing.v1 not available")

        at = AppTest.from_file(
            str(Path(__file__).parent.parent / "src" / "pages" / "01_data_analysis.py")
        )
        at.run()

        assert not at.exception
        assert len(at.title) >= 1
        assert "数据分析" in at.title[0].value

    def test_kpi_metrics_displayed(self):
        """AC2: Four KPI metric cards are shown."""
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            pytest.skip("streamlit.testing.v1 not available")

        at = AppTest.from_file(
            str(Path(__file__).parent.parent / "src" / "pages" / "01_data_analysis.py")
        )
        at.run()

        assert not at.exception
        assert len(at.metric) >= 4

    def test_categorical_selectbox_present(self):
        """AC4: Categorical feature selectbox is rendered."""
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            pytest.skip("streamlit.testing.v1 not available")

        at = AppTest.from_file(
            str(Path(__file__).parent.parent / "src" / "pages" / "01_data_analysis.py")
        )
        at.run()

        assert not at.exception
        # Should have at least 2 selectboxes (cat + num)
        assert len(at.selectbox) >= 2

        # One of them should have the categorical label
        cat_labels = [sb.label for sb in at.selectbox]
        assert any("分类特征" in lbl for lbl in cat_labels)

    def test_numeric_selectbox_present(self):
        """AC5: Numeric feature selectbox is rendered."""
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            pytest.skip("streamlit.testing.v1 not available")

        at = AppTest.from_file(
            str(Path(__file__).parent.parent / "src" / "pages" / "01_data_analysis.py")
        )
        at.run()

        assert not at.exception
        num_labels = [sb.label for sb in at.selectbox]
        assert any("数值特征" in lbl for lbl in num_labels)

    def test_bivariate_radio_present(self):
        """AC7: Chart type radio buttons are shown."""
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            pytest.skip("streamlit.testing.v1 not available")

        at = AppTest.from_file(
            str(Path(__file__).parent.parent / "src" / "pages" / "01_data_analysis.py")
        )
        at.run()

        assert not at.exception
        assert len(at.radio) >= 1

        radio_labels = [r.label for r in at.radio]
        assert any("图表类型" in lbl for lbl in radio_labels)


class TestDataAnalysisHelpers:
    """Unit tests for data analysis helper functions."""

    def test_load_data_returns_dataframe(self):
        """Given a valid CSV, _load_data returns a DataFrame."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = _make_sample_csv(tmp_dir)
            page_mod = importlib.import_module(PAGE_MODULE)
            page_mod._load_data.clear()

            df = page_mod._load_data(csv_path)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 6
            assert "subscribe" in df.columns

    def test_get_data_path_returns_valid(self):
        """_get_data_path returns a string ending with train.csv."""
        page_mod = importlib.import_module(PAGE_MODULE)
        result = page_mod._get_data_path()
        assert isinstance(result, str)
        assert result.endswith("train.csv")
