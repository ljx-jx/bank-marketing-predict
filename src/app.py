"""Bank Marketing Prediction App - Main Entry.

Streamlit multi-page application for data analysis and subscription prediction.
"""

import streamlit as st


def main() -> None:
    """Configure and launch the Streamlit app."""
    st.set_page_config(
        page_title="银行营销数据分析与预测系统",
        page_icon=":bank:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🏦 银行营销数据分析与预测系统")
    st.markdown("---")

    st.markdown(
        """
        ### 欢迎使用银行营销数据分析与预测系统

        本系统提供两大核心功能：

        | 功能 | 说明 |
        |------|------|
        | 📊 **数据分析** | 交互式探索银行营销数据，了解客户特征分布、认购率趋势与关键影响因素 |
        | 🔮 **在线预测** | 基于离线训练的机器学习模型，输入客户特征即可预测认购意向 |

        **请从左侧导航栏选择功能页面。**
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📁 训练样本", "22,501 条", "train.csv")
    with col2:
        st.metric("🧪 测试样本", "7,501 条", "test.csv")
    with col3:
        st.metric("🎯 预测目标", "subscribe", "yes / no")

    st.markdown("---")
    st.caption(
        "Tech Stack: Python 3.11 · Streamlit · scikit-learn · Docker | Port: 8004"
    )


if __name__ == "__main__":
    main()
