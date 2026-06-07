"""Online Prediction Page — point-and-click customer subscription prediction.

US-4: Provides an input form covering all 20 model features, loads the
trained pipeline, and returns a real-time prediction with probability.
"""

from pathlib import Path

import streamlit as st

from src.model.predict import load_model, predict

# ---------------------------------------------------------------------------
# Default values (medians / modes from the training dataset)
# ---------------------------------------------------------------------------
NUMERIC_DEFAULTS = {
    "age": 38,
    "duration": 353,
    "campaign": 1,
    "pdays": 964,
    "previous": 0,
    "emp_var_rate": 1.1,
    "cons_price_index": 93.54,
    "cons_conf_index": -40.60,
    "lending_rate3m": 3.92,
    "nr_employed": 5134.0,
}

NUMERIC_RANGES = {
    "age": (16, 101, 1),
    "duration": (0, 5200, 10),
    "campaign": (0, 60, 1),
    "pdays": (0, 1050, 1),
    "previous": (0, 7, 1),
    "emp_var_rate": (-3.5, 1.5, 0.1),
    "cons_price_index": (87.0, 100.0, 0.01),
    "cons_conf_index": (-54.0, -25.0, 0.1),
    "lending_rate3m": (0.5, 5.5, 0.01),
    "nr_employed": (4700.0, 5500.0, 1.0),
}

CATEGORICAL_OPTIONS = {
    "job": [
        "admin.",
        "blue-collar",
        "technician",
        "services",
        "management",
        "retired",
        "entrepreneur",
        "self-employed",
        "housemaid",
        "unemployed",
        "student",
        "unknown",
    ],
    "marital": ["married", "single", "divorced", "unknown"],
    "education": [
        "university.degree",
        "high.school",
        "basic.9y",
        "professional.course",
        "unknown",
        "basic.4y",
        "basic.6y",
        "illiterate",
    ],
    "default": ["no", "unknown", "yes"],
    "housing": ["yes", "no", "unknown"],
    "loan": ["no", "yes", "unknown"],
    "contact": ["cellular", "telephone"],
    "month": ["may", "jun", "jul", "aug", "oct", "nov", "dec", "mar", "apr", "sep"],
    "day_of_week": ["mon", "tue", "wed", "thu", "fri"],
    "poutcome": ["nonexistent", "failure", "success"],
}

CATEGORICAL_DEFAULTS = {
    "job": "admin.",
    "marital": "married",
    "education": "university.degree",
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "month": "may",
    "day_of_week": "thu",
    "poutcome": "nonexistent",
}

# Feature labels for display
FEATURE_LABELS = {
    "age": "年龄",
    "job": "职业",
    "marital": "婚姻状况",
    "education": "教育程度",
    "default": "信用违约",
    "housing": "住房贷款",
    "loan": "个人贷款",
    "contact": "联系方式",
    "month": "联系月份",
    "day_of_week": "联系星期",
    "duration": "通话时长 (秒)",
    "campaign": "营销次数",
    "pdays": "上次联系距今天数",
    "previous": "之前联系次数",
    "poutcome": "上次营销结果",
    "emp_var_rate": "就业变动率",
    "cons_price_index": "消费者价格指数",
    "cons_conf_index": "消费者信心指数",
    "lending_rate3m": "3个月贷款利率",
    "nr_employed": "就业人数",
}


# ---------------------------------------------------------------------------
# Page render
# ---------------------------------------------------------------------------
def main() -> None:
    """Render the online prediction page."""
    st.title("🔮 在线预测")
    st.markdown("输入客户特征，点击预测按钮即可获得认购意向预测结果。")

    # --- Check model availability ---
    model_path = _resolve_model_path()
    if not Path(model_path).exists():
        st.warning(
            "⚠️ 模型文件未找到，请先训练模型。\n\n"
            "在终端中运行以下命令：\n"
            "```bash\npython -m src.model.train\n```"
        )
        st.stop()

    # Load model once (with Streamlit caching)
    try:
        pipeline = _load_model_cached(model_path)
    except Exception as exc:
        st.error(f"模型加载失败：{exc}")
        st.stop()

    if pipeline is None:
        st.error("模型加载失败，请重新训练。")
        st.stop()

    # --- Input form ---
    st.subheader("📝 客户特征输入")
    st.caption("请填写以下客户特征信息，默认为训练数据中位数/众数值。")

    with st.form("prediction_form"):
        features = {}

        # ---- Personal Info ----
        st.markdown("#### 👤 个人信息")
        col1, col2, col3 = st.columns(3)
        with col1:
            features["age"] = st.number_input(
                f"{FEATURE_LABELS['age']}",
                min_value=NUMERIC_RANGES["age"][0],
                max_value=NUMERIC_RANGES["age"][1],
                value=NUMERIC_DEFAULTS["age"],
                step=NUMERIC_RANGES["age"][2],
            )
        with col2:
            features["job"] = st.selectbox(
                FEATURE_LABELS["job"],
                CATEGORICAL_OPTIONS["job"],
                index=CATEGORICAL_OPTIONS["job"].index(CATEGORICAL_DEFAULTS["job"]),
            )
        with col3:
            features["marital"] = st.selectbox(
                FEATURE_LABELS["marital"],
                CATEGORICAL_OPTIONS["marital"],
                index=CATEGORICAL_OPTIONS["marital"].index(
                    CATEGORICAL_DEFAULTS["marital"]
                ),
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            features["education"] = st.selectbox(
                FEATURE_LABELS["education"],
                CATEGORICAL_OPTIONS["education"],
                index=CATEGORICAL_OPTIONS["education"].index(
                    CATEGORICAL_DEFAULTS["education"]
                ),
            )
        with col2:
            features["default"] = st.selectbox(
                FEATURE_LABELS["default"],
                CATEGORICAL_OPTIONS["default"],
                index=CATEGORICAL_OPTIONS["default"].index(
                    CATEGORICAL_DEFAULTS["default"]
                ),
            )
        with col3:
            features["housing"] = st.selectbox(
                FEATURE_LABELS["housing"],
                CATEGORICAL_OPTIONS["housing"],
                index=CATEGORICAL_OPTIONS["housing"].index(
                    CATEGORICAL_DEFAULTS["housing"]
                ),
            )

        # ---- Contact Info ----
        st.markdown("#### 📞 联系信息")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            features["contact"] = st.selectbox(
                FEATURE_LABELS["contact"],
                CATEGORICAL_OPTIONS["contact"],
                index=CATEGORICAL_OPTIONS["contact"].index(
                    CATEGORICAL_DEFAULTS["contact"]
                ),
            )
        with col2:
            features["month"] = st.selectbox(
                FEATURE_LABELS["month"],
                CATEGORICAL_OPTIONS["month"],
                index=CATEGORICAL_OPTIONS["month"].index(CATEGORICAL_DEFAULTS["month"]),
            )
        with col3:
            features["day_of_week"] = st.selectbox(
                FEATURE_LABELS["day_of_week"],
                CATEGORICAL_OPTIONS["day_of_week"],
                index=CATEGORICAL_OPTIONS["day_of_week"].index(
                    CATEGORICAL_DEFAULTS["day_of_week"]
                ),
            )
        with col4:
            features["loan"] = st.selectbox(
                FEATURE_LABELS["loan"],
                CATEGORICAL_OPTIONS["loan"],
                index=CATEGORICAL_OPTIONS["loan"].index(CATEGORICAL_DEFAULTS["loan"]),
            )

        # ---- Campaign Info ----
        st.markdown("#### 📊 营销信息")
        col1, col2, col3 = st.columns(3)
        with col1:
            features["duration"] = st.slider(
                FEATURE_LABELS["duration"],
                min_value=NUMERIC_RANGES["duration"][0],
                max_value=NUMERIC_RANGES["duration"][1],
                value=NUMERIC_DEFAULTS["duration"],
                step=NUMERIC_RANGES["duration"][2],
            )
        with col2:
            features["campaign"] = st.slider(
                FEATURE_LABELS["campaign"],
                min_value=NUMERIC_RANGES["campaign"][0],
                max_value=NUMERIC_RANGES["campaign"][1],
                value=NUMERIC_DEFAULTS["campaign"],
                step=NUMERIC_RANGES["campaign"][2],
            )
        with col3:
            features["pdays"] = st.slider(
                FEATURE_LABELS["pdays"],
                min_value=NUMERIC_RANGES["pdays"][0],
                max_value=NUMERIC_RANGES["pdays"][1],
                value=NUMERIC_DEFAULTS["pdays"],
                step=NUMERIC_RANGES["pdays"][2],
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            features["previous"] = st.slider(
                FEATURE_LABELS["previous"],
                min_value=NUMERIC_RANGES["previous"][0],
                max_value=NUMERIC_RANGES["previous"][1],
                value=NUMERIC_DEFAULTS["previous"],
                step=NUMERIC_RANGES["previous"][2],
            )
        with col2:
            features["poutcome"] = st.selectbox(
                FEATURE_LABELS["poutcome"],
                CATEGORICAL_OPTIONS["poutcome"],
                index=CATEGORICAL_OPTIONS["poutcome"].index(
                    CATEGORICAL_DEFAULTS["poutcome"]
                ),
            )

        # ---- Economic Indicators ----
        st.markdown("#### 💰 经济指标")
        col1, col2, col3 = st.columns(3)
        with col1:
            features["emp_var_rate"] = st.slider(
                FEATURE_LABELS["emp_var_rate"],
                min_value=NUMERIC_RANGES["emp_var_rate"][0],
                max_value=NUMERIC_RANGES["emp_var_rate"][1],
                value=NUMERIC_DEFAULTS["emp_var_rate"],
                step=NUMERIC_RANGES["emp_var_rate"][2],
            )
        with col2:
            features["cons_price_index"] = st.slider(
                FEATURE_LABELS["cons_price_index"],
                min_value=NUMERIC_RANGES["cons_price_index"][0],
                max_value=NUMERIC_RANGES["cons_price_index"][1],
                value=NUMERIC_DEFAULTS["cons_price_index"],
                step=NUMERIC_RANGES["cons_price_index"][2],
            )
        with col3:
            features["cons_conf_index"] = st.slider(
                FEATURE_LABELS["cons_conf_index"],
                min_value=NUMERIC_RANGES["cons_conf_index"][0],
                max_value=NUMERIC_RANGES["cons_conf_index"][1],
                value=NUMERIC_DEFAULTS["cons_conf_index"],
                step=NUMERIC_RANGES["cons_conf_index"][2],
            )

        col1, col2 = st.columns(2)
        with col1:
            features["lending_rate3m"] = st.slider(
                FEATURE_LABELS["lending_rate3m"],
                min_value=NUMERIC_RANGES["lending_rate3m"][0],
                max_value=NUMERIC_RANGES["lending_rate3m"][1],
                value=NUMERIC_DEFAULTS["lending_rate3m"],
                step=NUMERIC_RANGES["lending_rate3m"][2],
            )
        with col2:
            features["nr_employed"] = st.slider(
                FEATURE_LABELS["nr_employed"],
                min_value=NUMERIC_RANGES["nr_employed"][0],
                max_value=NUMERIC_RANGES["nr_employed"][1],
                value=NUMERIC_DEFAULTS["nr_employed"],
                step=NUMERIC_RANGES["nr_employed"][2],
            )

        st.markdown("---")
        submitted = st.form_submit_button(
            "🔮 预测认购意向",
            type="primary",
            use_container_width=True,
        )

    # --- Prediction result ---
    if submitted:
        _show_prediction(features, model_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_model_path() -> str:
    """Find the trained model file.

    Checks BANK_MODEL_PATH env var first (for CI/testing).
    """
    import os

    env_path = os.environ.get("BANK_MODEL_PATH")
    if env_path:
        return env_path

    candidates = [
        Path(__file__).parent.parent.parent / "models" / "model_pipeline.pkl",
        Path("models") / "model_pipeline.pkl",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return str(candidates[0])


@st.cache_resource
def _load_model_cached(model_path: str):
    """Load the model with Streamlit's resource caching."""
    return load_model(model_path)


def _show_prediction(features: dict, model_path: str) -> None:
    """Display the prediction result in a styled card."""
    try:
        result = predict(model_path, features)
    except Exception as exc:
        st.error(f"预测失败：{exc}")
        return

    proba = result["probability"]
    is_yes = result["subscribe"]

    st.markdown("---")
    st.subheader("📊 预测结果")

    col1, col2 = st.columns([1, 2])

    with col1:
        if is_yes:
            st.markdown(
                """
                <div style="padding:20px;border-radius:10px;background-color:#d4edda;
                            text-align:center;font-size:24px;">
                    ✅ <b>会认购</b>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="padding:20px;border-radius:10px;background-color:#f8d7da;
                            text-align:center;font-size:24px;">
                    ❌ <b>不会认购</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col2:
        st.metric(
            label="认购概率",
            value=f"{proba:.1%}",
            delta=f"{proba - 0.5:.1%} vs 50%阈值"
            if proba >= 0.5
            else f"{proba - 0.5:.1%} vs 50%阈值",
            delta_color="normal",
        )
        st.progress(proba, text=f"{proba:.2%}")

    # Interpretation hints
    st.info(
        f"💡 **解读**：该客户的预测认购概率为 **{proba:.1%}**。"
        + (
            "高于 50% 阈值，建议优先跟进。"
            if is_yes
            else "低于 50% 阈值，认购可能性较低。"
        )
    )


if __name__ == "__main__":
    main()
