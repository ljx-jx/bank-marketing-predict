"""Data Analysis Page — interactive exploration of bank marketing data.

US-2: Provides KPI overview, feature distributions, correlation analysis,
and bivariate exploration to help analysts understand customer patterns.
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.utils.data_loader import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    ID_COL,
    NUMERIC_COLS,
    TARGET_COL,
    load_csv,
)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def _load_data(data_path: str) -> pd.DataFrame:
    """Load and cache the training data."""
    return load_csv(data_path)


def _get_data_path() -> str:
    """Resolve the train.csv path relative to the project root."""
    candidates = [
        Path(__file__).parent.parent.parent / "data" / "train.csv",
        Path("data") / "train.csv",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return str(candidates[0])


# ---------------------------------------------------------------------------
# Page render
# ---------------------------------------------------------------------------
def main() -> None:
    """Render the data analysis dashboard."""
    st.title("📊 数据分析")
    st.markdown("交互式探索银行营销数据，了解客户特征分布与认购率趋势。")

    data_path = _get_data_path()
    try:
        df = _load_data(data_path)
    except FileNotFoundError:
        st.error(f"数据文件未找到：`{data_path}`。请确保 `data/train.csv` 存在。")
        st.stop()

    # Feature columns only (exclude id and target)
    feature_df = df.drop(columns=[ID_COL, TARGET_COL])

    # ---- AC2: KPI Overview ----
    _render_kpi_section(df)

    st.markdown("---")

    # ---- AC3: Descriptive Statistics ----
    _render_statistics_section(feature_df)

    st.markdown("---")

    # ---- AC4: Categorical Feature Analysis ----
    _render_categorical_section(df)

    st.markdown("---")

    # ---- AC5: Numeric Feature Distribution ----
    _render_numeric_section(df)

    st.markdown("---")

    # ---- AC6: Correlation Heatmap ----
    _render_correlation_section(feature_df)

    st.markdown("---")

    # ---- AC7: Bivariate Exploration ----
    _render_bivariate_section(df)


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------
def _render_kpi_section(df: pd.DataFrame) -> None:
    """AC2: Render KPI overview cards."""
    st.subheader("📈 关键指标")

    total = len(df)
    yes_count = (df[TARGET_COL] == "yes").sum()
    no_count = total - yes_count
    yes_pct = yes_count / total * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 客户总数", f"{total:,}")
    col2.metric("✅ 认购客户", f"{yes_count:,}", f"{yes_pct:.1f}%")
    col3.metric("❌ 未认购客户", f"{no_count:,}", f"{100 - yes_pct:.1f}%")
    col4.metric("📊 特征数量", str(len(FEATURE_COLS)))


def _render_statistics_section(feature_df: pd.DataFrame) -> None:
    """AC3: Render descriptive statistics for numeric features."""
    st.subheader("📋 数值特征描述性统计")

    with st.expander("点击展开描述性统计表"):
        numeric_df = feature_df[NUMERIC_COLS]
        stats = numeric_df.describe().T
        stats["median"] = numeric_df.median()
        stats["skew"] = numeric_df.skew()
        st.dataframe(
            stats.style.format("{:.2f}"),
            use_container_width=True,
            height=400,
        )


def _render_categorical_section(df: pd.DataFrame) -> None:
    """AC4: Render subscription rate by categorical feature."""
    st.subheader("📊 分类特征认购率分析")

    cat_feature = st.selectbox(
        "选择分类特征",
        CATEGORICAL_COLS,
        key="cat_select",
        help="选择一个分类特征查看各取值下的认购率对比",
    )

    # Compute subscription rate per category
    grouped = (
        df.groupby(cat_feature)[TARGET_COL]
        .apply(lambda x: (x == "yes").mean() * 100)
        .reset_index()
    )
    grouped.columns = [cat_feature, "认购率 (%)"]

    # Sort by subscription rate for better readability
    grouped = grouped.sort_values("认购率 (%)", ascending=False)

    fig = px.bar(
        grouped,
        x=cat_feature,
        y="认购率 (%)",
        title=f"{cat_feature} 各取值下的认购率",
        text_auto=".1f",
        color="认购率 (%)",
        color_continuous_scale="Blues",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


def _render_numeric_section(df: pd.DataFrame) -> None:
    """AC5: Render distribution histogram for a numeric feature."""
    st.subheader("📉 数值特征分布")

    num_feature = st.selectbox(
        "选择数值特征",
        NUMERIC_COLS,
        key="num_select",
        help="选择一个数值特征查看其分布（按认购状态分层）",
    )

    # Filter outliers for better visualization (using IQR)
    q1 = df[num_feature].quantile(0.01)
    q99 = df[num_feature].quantile(0.99)
    filtered = df[(df[num_feature] >= q1) & (df[num_feature] <= q99)]

    fig = px.histogram(
        filtered,
        x=num_feature,
        color=TARGET_COL,
        barmode="overlay",
        histnorm="percent",
        title=f"{num_feature} 分布（按认购状态分层，剔除 1% 极端值）",
        marginal="box",
        opacity=0.7,
        color_discrete_map={"yes": "#2E86AB", "no": "#A23B72"},
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_correlation_section(feature_df: pd.DataFrame) -> None:
    """AC6: Render correlation heatmap for numeric features."""
    st.subheader("🔥 数值特征相关性热力图")

    numeric_df = feature_df[NUMERIC_COLS]
    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        title="数值特征 Pearson 相关系数",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)


def _render_bivariate_section(df: pd.DataFrame) -> None:
    """AC7: Render bivariate exploration (scatter / box plot)."""
    st.subheader("🔍 双变量探索")

    plot_type = st.radio(
        "选择图表类型",
        ["散点图", "箱线图"],
        horizontal=True,
        key="bivar_type",
    )

    all_cols = NUMERIC_COLS + CATEGORICAL_COLS

    col1, col2 = st.columns(2)
    with col1:
        x_feature = st.selectbox(
            "X 轴特征",
            all_cols,
            index=0,
            key="bivar_x",
        )
    with col2:
        y_feature = st.selectbox(
            "Y 轴特征",
            all_cols,
            index=min(1, len(all_cols) - 1),
            key="bivar_y",
        )

    if plot_type == "散点图":
        # For scatter, both should ideally be numeric
        if x_feature in NUMERIC_COLS and y_feature in NUMERIC_COLS:
            sample = df.sample(min(3000, len(df)), random_state=42)
            fig = px.scatter(
                sample,
                x=x_feature,
                y=y_feature,
                color=TARGET_COL,
                title=f"{x_feature} vs {y_feature}（抽样 3000 条）",
                opacity=0.6,
                color_discrete_map={"yes": "#2E86AB", "no": "#A23B72"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("散点图需要两个数值特征。请选择数值列，或使用箱线图。")

    elif plot_type == "箱线图":
        # Box plot: x = categorical, y = numeric
        if x_feature in CATEGORICAL_COLS and y_feature in NUMERIC_COLS:
            fig = px.box(
                df,
                x=x_feature,
                y=y_feature,
                color=TARGET_COL,
                title=f"{y_feature} 按 {x_feature} 分布",
                color_discrete_map={"yes": "#2E86AB", "no": "#A23B72"},
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("箱线图建议 X 轴为分类特征，Y 轴为数值特征。")


# Support direct execution and Streamlit page auto-discovery
if __name__ == "__main__":
    main()
