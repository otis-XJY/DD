"""统一样式与理论框架展示（融合 human_ai_trust 视觉设计与 project 理论叙述）。"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Noto Sans SC', 'Microsoft YaHei', 'Segoe UI', sans-serif;
        }
        .block-container { padding-top: 1.2rem; max-width: 1200px; }
        .trust-hero {
            background: linear-gradient(135deg, #123c7c 0%, #2c7be5 55%, #66c2ff 100%);
            color: #fff;
            padding: 2.2rem 2rem 1.8rem;
            border-radius: 18px;
            margin-bottom: 1.5rem;
            box-shadow: 0 12px 40px rgba(18, 60, 124, 0.25);
        }
        .trust-hero .eyebrow {
            text-transform: uppercase;
            letter-spacing: 2px;
            opacity: 0.85;
            font-size: 0.85rem;
            margin: 0 0 0.4rem;
        }
        .trust-hero h1 {
            font-size: 1.85rem;
            margin: 0 0 0.75rem;
            font-weight: 700;
            line-height: 1.35;
        }
        .trust-hero .subtitle {
            font-size: 1.05rem;
            line-height: 1.75;
            opacity: 0.95;
            max-width: 900px;
            margin: 0;
        }
        .trust-metric-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 14px;
            margin-top: 1.4rem;
        }
        .trust-metric-card {
            background: rgba(255,255,255,0.92);
            color: #123c7c;
            border-radius: 14px;
            padding: 14px 16px;
            box-shadow: 0 8px 24px rgba(40, 64, 96, 0.12);
        }
        .trust-metric-card span {
            display: block;
            font-size: 0.8rem;
            opacity: 0.85;
            margin-bottom: 6px;
        }
        .trust-metric-card strong {
            font-size: 1.5rem;
            font-weight: 700;
        }
        .trust-panel {
            background: rgba(255,255,255,0.95);
            border-radius: 16px;
            padding: 1.25rem 1.4rem;
            margin-bottom: 1rem;
            border: 1px solid #dbe7f5;
            box-shadow: 0 8px 24px rgba(18, 60, 124, 0.06);
        }
        .trust-panel h3 {
            color: #123c7c;
            margin-top: 0;
            font-size: 1.1rem;
        }
        .trust-panel p, .trust-panel li {
            color: #4a6078;
            line-height: 1.8;
        }
        .theory-loop {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 0.5rem 0 0;
        }
        .theory-loop span {
            background: #e8f2ff;
            color: #123c7c;
            padding: 6px 14px;
            border-radius: 999px;
            font-size: 0.9rem;
            font-weight: 500;
        }
        .dim-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
        }
        .dim-card {
            background: #f5f9ff;
            border-left: 4px solid #2c7be5;
            padding: 12px 14px;
            border-radius: 0 12px 12px 0;
        }
        .dim-card b { color: #123c7c; }
        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f4f8ff 0%, #eef3f8 100%);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _load_metrics_summary() -> dict | None:
    path = OUTPUT_DIR / "model_metrics.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def render_hero() -> None:
    metrics = _load_metrics_summary()
    samples = metrics["data_rows"] if metrics else "—"
    rate = f"{metrics['positive_rate']:.1%}" if metrics else "—"
    ebm_auc = "—"
    ebm_brier = "—"
    if metrics:
        for row in metrics.get("models", []):
            if "Explainable Boosting Machine" in row.get("model", ""):
                ebm_auc = f"{row['auc']:.4f}"
                ebm_brier = f"{row['brier_score']:.4f}"
                break

    st.markdown(
        f"""
        <div class="trust-hero">
            <p class="eyebrow">Human-AI Research Trust · Unified Panel</p>
            <h1>人机协同科研决策信任机制解释面板</h1>
            <p class="subtitle">
                本系统面向科研项目资助评审场景，采用「AI 参谋、专家决策」定位：
                不替代专家审批，而是通过可解释模型输出资助概率、证据链与信任校准信号，
                支撑专家在采纳、修正、复核或否决时形成<strong>适当信任</strong>。
            </p>
            <div class="trust-metric-row">
                <div class="trust-metric-card"><span>研究样本量</span><strong>{samples}</strong></div>
                <div class="trust-metric-card"><span>资助率</span><strong>{rate}</strong></div>
                <div class="trust-metric-card"><span>EBM AUC</span><strong>{ebm_auc}</strong></div>
                <div class="trust-metric-card"><span>EBM Brier</span><strong>{ebm_brier}</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_research_design() -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="trust-panel">
                <h3>研究设计</h3>
                <ul>
                    <li><b>研究对象</b>：科研评审与科研资助决策中的人机协同判断过程。</li>
                    <li><b>研究方法</b>：Explainable Boosting Machine（EBM）玻璃盒模型 + 半仿真数据 + OpenAlex/NSFC 公开元数据映射。</li>
                    <li><b>研究目标</b>：分析可解释性、证据可追溯性、任务风险等如何影响信任校准，避免过度信任与信任不足。</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="trust-panel">
                <h3>四层研究框架</h3>
                <ol>
                    <li><b>输入层</b>：半仿真申请特征 + 公开样本元数据映射。</li>
                    <li><b>模型层</b>：EBM 主模型，LR/RF 作为透明基线与黑盒对照。</li>
                    <li><b>解释层</b>：全局重要性、局部加减分、响应曲线。</li>
                    <li><b>协同层</b>：五维度信任评估 + 协同分流 + 专家复核闭环。</li>
                </ol>
                <div class="theory-loop">
                    <span>预测</span><span>解释</span><span>校准</span><span>复核</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_trust_dimensions_theory() -> None:
    st.markdown(
        """
        <div class="trust-panel">
            <h3>五维度信任评估（理论依据）</h3>
            <p>新申请分析模块（步骤4）将模型输出映射为专家可理解的信任信号，对应高责任科研决策中的信任校准需求：</p>
            <div class="dim-grid">
                <div class="dim-card"><b>可解释性</b><br/>EBM 局部解释是否清晰、预测置信度是否足以支撑核验。</div>
                <div class="dim-card"><b>证据可追溯性</b><br/>原文能否映射至 NSFC/OpenAlex 等公开参照字段。</div>
                <div class="dim-card"><b>任务风险</b><br/>风险识别与缓释能力（与资助概率影响区分）。</div>
                <div class="dim-card"><b>研究者素养</b><br/>论文、项目、h-index 等履历信号与原文一致性。</div>
                <div class="dim-card"><b>组织规范</b><br/>单位层次、伦理合规等制度性信任因素。</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_routing_theory() -> None:
    st.markdown(
        """
        <div class="trust-panel">
            <h3>协同分流规则（适当依赖 AI）</h3>
            <p>分流依据预测置信度 <code>abs(p-0.5)×2</code>、项目风险与偏差审计信号（单位层次、海外经历、年龄等在局部解释中的占比），
            将项目分配至不同强度的人工复核流程，体现「AI 帮助专家分配注意力」：</p>
            <table style="width:100%; border-collapse:collapse; background:#fff;">
                <tr style="background:#f5f9ff;">
                    <th style="padding:10px; text-align:left;">分流类别</th>
                    <th style="padding:10px; text-align:left;">含义</th>
                    <th style="padding:10px; text-align:left;">人机协同方式</th>
                </tr>
                <tr><td style="padding:10px; border-bottom:1px solid #e4ebf5;"><b>快速复核</b></td>
                    <td style="padding:10px; border-bottom:1px solid #e4ebf5;">高置信、低风险、偏差信号低</td>
                    <td style="padding:10px; border-bottom:1px solid #e4ebf5;">可提高效率，仍需抽检</td></tr>
                <tr><td style="padding:10px; border-bottom:1px solid #e4ebf5;"><b>常规人工复核</b></td>
                    <td style="padding:10px; border-bottom:1px solid #e4ebf5;">中等置信或一般风险</td>
                    <td style="padding:10px; border-bottom:1px solid #e4ebf5;">标准专家审查流程</td></tr>
                <tr><td style="padding:10px;"><b>专家重点审查</b></td>
                    <td style="padding:10px;">低置信、高风险或偏差审计突出</td>
                    <td style="padding:10px;">必须深度人工论证</td></tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_theory_overview_page() -> None:
    """研究概览页：human_ai 式研究设计 + project 理论分流说明。"""
    render_research_design()
    render_trust_dimensions_theory()
    render_routing_theory()
    st.info(
        "数据策略：半仿真数据承载模型训练与解释演示；OpenAlex/NSFC 公开元数据用于特征映射与外部效度参照。"
        " 重新训练后刷新页面即可读取 outputs/ 下最新结果。"
    )
