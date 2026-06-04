"""批量解释面板：复现 project/streamlit_app 六标签页功能。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "data"


def load_json(name: str):
    with (OUTPUT_DIR / name).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / name)


def load_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def _load_batch_assets():
    metrics = load_json("model_metrics.json")
    global_importance = pd.DataFrame(load_json("global_importance.json"))
    group_importance = pd.DataFrame(load_json("group_importance.json"))
    response_curves = load_json("response_curves.json")
    local_explanations = load_json("local_explanations.json")
    routing = load_csv("case_routing.csv")
    public_mapping = load_optional_csv(OUTPUT_DIR / "public_feature_mapping.csv")
    if public_mapping is None:
        public_mapping = load_optional_csv(OUTPUT_DIR / "nsfc_public_enriched_mapping.csv")
    public_sample = load_optional_csv(DATA_DIR / "openalex_reference_sample.csv")
    return (
        metrics,
        global_importance,
        group_importance,
        response_curves,
        local_explanations,
        routing,
        public_mapping,
        public_sample,
    )


def render_batch_explain_panel() -> None:
    st.markdown("### 模型解释与协同分流面板")
    st.caption(
        "本面板将 EBM 解释结果转化为专家可审计的决策证据，对应理论中的「解释层」与「协同层」。"
        " 与「新申请端到端分析」互补：此处展示批量实验与典型个案，后者支持单条原文输入。"
    )

    try:
        assets = _load_batch_assets()
    except FileNotFoundError as exc:
        st.error(f"缺少解释输出文件：{exc}。请先运行 src/train_models.py 与 src/explain_and_export.py。")
        return

    (
        metrics,
        global_importance,
        group_importance,
        response_curves,
        local_explanations,
        routing,
        public_mapping,
        public_sample,
    ) = assets

    model_rows = pd.DataFrame(metrics["models"])
    ebm_metrics = model_rows[model_rows["model"].str.contains("Explainable Boosting Machine")].iloc[0]

    metric_cols = st.columns(4)
    metric_cols[0].metric("样本数", metrics["data_rows"])
    metric_cols[1].metric("资助率", f"{metrics['positive_rate']:.2%}")
    metric_cols[2].metric("EBM AUC", f"{ebm_metrics['auc']:.4f}")
    metric_cols[3].metric("EBM Brier", f"{ebm_metrics['brier_score']:.4f}")

    tab_compare, tab_global, tab_case, tab_response, tab_routing, tab_public = st.tabs(
        ["模型对比", "全局解释", "个案证据", "响应曲线", "协同分流", "公开数据依据"]
    )

    with tab_compare:
        st.subheader("三类模型对比")
        st.dataframe(model_rows, width="stretch", hide_index=True)
        long_metrics = model_rows.melt(
            id_vars="model",
            value_vars=["accuracy", "auc", "brier_score", "log_loss"],
            var_name="metric",
            value_name="value",
        )
        st.plotly_chart(
            px.bar(
                long_metrics,
                x="metric",
                y="value",
                color="model",
                barmode="group",
                title="模型性能对比",
            ),
            width="stretch",
        )
        st.markdown(
            """
            **理论解读**
            - **Logistic Regression**：透明线性基线，说明传统可解释模型的能力边界。
            - **EBM**：主模型，兼顾非线性、局部加减分解释与响应曲线，支撑信任校准。
            - **Random Forest**：黑盒对照，说明「预测性能 ≠ 可审计的信任机制」。
            """
        )

    with tab_global:
        left, right = st.columns([1.4, 1])
        with left:
            top_terms = global_importance.head(15).copy()
            top_terms["label"] = top_terms["feature_labels_zh"].apply(
                lambda x: " & ".join(x) if isinstance(x, list) else str(x)
            )
            fig = px.bar(
                top_terms.sort_values("importance"),
                x="importance",
                y="label",
                orientation="h",
                labels={"importance": "重要性", "label": "特征/交互项"},
                title="全局特征重要性 Top 15",
            )
            st.plotly_chart(fig, width="stretch")
        with right:
            fig_group = px.pie(
                group_importance,
                names="group",
                values="importance",
                hole=0.45,
                title="特征组贡献度",
            )
            st.plotly_chart(fig_group, width="stretch")
            st.dataframe(group_importance, width="stretch", hide_index=True)
        st.markdown(
            "全局解释支撑专家对 AI **宏观判断依据** 的理解：项目质量组通常贡献最高，"
            "有助于缓解「模型是否过度依赖申请人背景」的担忧。"
        )

    with tab_case:
        options = [item["application_id"] for item in local_explanations]
        selected_id = st.selectbox("选择典型项目", options)
        case = next(item for item in local_explanations if item["application_id"] == selected_id)
        st.metric("AI 建议", case["ai_recommendation"], f"预测概率 {case['predicted_funding_probability']}")
        pos = pd.DataFrame(case["top_positive"])
        neg = pd.DataFrame(case["top_negative"])
        col_pos, col_neg = st.columns(2)
        with col_pos:
            st.subheader("主要加分证据")
            if not pos.empty:
                st.plotly_chart(
                    px.bar(pos.sort_values("contribution"), x="contribution", y="term", orientation="h"),
                    width="stretch",
                )
        with col_neg:
            st.subheader("主要扣分证据")
            if not neg.empty:
                st.plotly_chart(
                    px.bar(neg.sort_values("contribution"), x="contribution", y="term", orientation="h"),
                    width="stretch",
                )
        st.markdown(
            "个案解释对应 **可解释性** 维度：专家可据此检查加分/扣分是否与领域常识一致，"
            "从而决定采纳、修正或否决 AI 建议。"
        )

    with tab_response:
        feature = st.selectbox(
            "选择变量",
            list(response_curves.keys()),
            format_func=lambda key: response_curves[key]["feature_label_zh"],
        )
        curve = response_curves[feature]
        curve_df = pd.DataFrame(curve["points"])
        if curve["kind"] == "numeric":
            fig_curve = px.line(
                curve_df,
                x="x",
                y="funding_probability",
                labels={"x": curve["feature_label_zh"], "funding_probability": "预测资助概率"},
                title=f"{curve['feature_label_zh']} 响应曲线",
            )
        else:
            fig_curve = px.bar(
                curve_df,
                x="x",
                y="funding_probability",
                labels={"x": curve["feature_label_zh"], "funding_probability": "预测资助概率"},
                title=f"{curve['feature_label_zh']} 类别响应",
            )
        st.plotly_chart(fig_curve, width="stretch")
        st.caption("响应曲线帮助专家判断变量—概率关系是否符合学科常识，是形成适当信任的关键证据。")

    with tab_routing:
        counts = routing["routing"].value_counts().rename_axis("routing").reset_index(name="count")
        st.plotly_chart(px.bar(counts, x="routing", y="count", title="协同决策分流统计"), width="stretch")
        st.dataframe(
            routing.sort_values(["routing", "confidence"], ascending=[True, False]).head(200),
            width="stretch",
            hide_index=True,
        )
        st.markdown(
            "分流结果体现 **预测—解释—校准—复核** 闭环：高置信低风险可走快速复核；"
            "低置信、高风险或偏差审计信号突出者须专家重点审查。"
        )

    with tab_public:
        if public_mapping is None:
            st.info("尚未生成公开数据映射。请先运行 src/download_public_reference.py。")
        else:
            st.subheader("真实公开字段到仿真特征的映射")
            st.dataframe(public_mapping.head(30), width="stretch", hide_index=True)
        if public_sample is not None:
            st.subheader("OpenAlex 真实元数据样例")
            st.dataframe(public_sample.head(30), width="stretch", hide_index=True)
        st.caption("公开数据依据支撑 **证据可追溯性**：说明特征设计并非凭空设定。")
