"""
人机协同科研决策信任机制 — 统一可视化入口。

融合：
- human_ai_trust_project：研究设计、四层框架、五维度信任理论、视觉样式
- project：模型对比、全局/个案解释、响应曲线、协同分流、公开数据六标签页
- unified 既有功能：新申请端到端六步分析（new_application_analyzer）
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
for p in (str(APP_DIR), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from batch_explain_panel import render_batch_explain_panel
from new_application_analyzer import render_new_application_workflow
from ui_styles import (
    inject_custom_css,
    render_hero,
    render_theory_overview_page,
)

st.set_page_config(
    page_title="人机协同科研决策信任机制",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()

with st.sidebar:
    st.markdown("### 功能导航")
    module = st.radio(
        "选择模块",
        ["研究概览与理论", "模型解释面板", "新申请端到端分析"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown(
        """
        **理论闭环**  
        预测 → 解释 → 校准 → 复核

        **数据策略**  
        半仿真训练 + 公开元数据映射
        """
    )

render_hero()

if module == "研究概览与理论":
    render_theory_overview_page()
elif module == "模型解释面板":
    render_batch_explain_panel()
else:
    render_new_application_workflow()
