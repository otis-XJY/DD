import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (
    FEATURE_COLUMNS,
    FEATURE_LABELS_ZH,
    SENSITIVE_OR_BIAS_AUDIT_FEATURES,
)

OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / name)


@st.cache_resource
def load_ebm_model():
    return joblib.load(MODEL_DIR / "ebm_model.joblib")


def input_excerpt(text: str, max_len: int = 160) -> str:
    cleaned = " ".join(text.strip().split())
    return cleaned if len(cleaned) <= max_len else cleaned[:max_len] + "…"


def parse_input_signals(text: str) -> dict:
    lower = text.lower()
    return {
        "has_ai": any(k in lower for k in ["人工智能", "大模型", "深度学习", "机器学习"]),
        "has_innovation": "创新" in text,
        "has_risk": "风险" in text,
        "has_risk_mitigation": any(k in text for k in ["缓释", "应对", "防范", "控制", "监测", "预案"]),
        "has_collab": any(k in text for k in ["合作", "联合", "跨单位"]),
        "has_ethics": any(k in text for k in ["伦理", "合规"]),
        "has_data": any(k in text for k in ["数据", "公开", "openalex", "nsfc", "国自然"]),
        "has_top_unit": any(k in lower for k in ["985", "双一流", "顶尖", "中科院"]),
        "innovation_count": text.count("创新"),
        "keyword_sample": "、".join([w for w in ["人工智能", "创新", "跨学科", "伦理", "风险", "合作"] if w in text][:4])
        or "未识别到典型关键词",
    }


def find_input_quotes(text: str, keywords: list[str], max_quotes: int = 2, window: int = 50) -> list[str]:
    if not text or not text.strip():
        return ["（未提供可引用的原文）"]
    quotes: list[str] = []
    for part in text.replace("。", "。\n").replace("；", "；\n").split("\n"):
        part = part.strip()
        if part and any(kw in part for kw in keywords):
            snippet = part if len(part) <= 140 else part[:140] + "…"
            if snippet not in quotes:
                quotes.append(snippet)
        if len(quotes) >= max_quotes:
            return quotes
    for kw in keywords:
        idx = text.find(kw)
        if idx >= 0:
            start, end = max(0, idx - window), min(len(text), idx + len(kw) + window)
            snippet = ("…" if start > 0 else "") + text[start:end].strip() + ("…" if end < len(text) else "")
            if snippet not in quotes:
                quotes.append(snippet)
        if len(quotes) >= max_quotes:
            break
    return quotes or ["（原文中未检测到与该维度直接相关的表述）"]


FEATURE_QUOTE_KEYWORDS = {
    "novelty_score": ["创新", "新型", "首创", "突破"],
    "frontier_score": ["人工智能", "大模型", "前沿", "深度学习", "机器学习"],
    "methodology_score": ["方法", "模型", "算法", "技术路线"],
    "feasibility_score": ["可行", "实施", "验证", "阶段"],
    "project_risk": ["风险", "缓释", "应对", "不确定性", "挑战"],
    "papers_5y": ["论文", "发表", "篇"],
    "interdisciplinarity": ["跨学科", "交叉", "多学科"],
    "collaboration_strength": ["合作", "联合", "协作"],
    "institution_tier": ["985", "双一流", "中科院", "依托", "大学"],
    "prior_grants": ["国自然", "nsfc", "基金", "项目"],
    "data_availability": ["数据", "公开", "共享"],
    "expected_impact": ["影响", "引用", "成果"],
}


def quotes_for_feature(feature: str, text: str) -> str:
    kws = FEATURE_QUOTE_KEYWORDS.get(feature, [FEATURE_LABELS_ZH.get(feature, feature)[:2]])
    qs = find_input_quotes(text, kws, max_quotes=1)
    return qs[0]


def route_case(prob: float, project_risk: float, fairness_share: float) -> tuple[str, str]:
    confidence = abs(prob - 0.5) * 2
    fairness_flag = fairness_share >= 0.25
    if confidence >= 0.65 and project_risk <= 60 and not fairness_flag:
        return "fast_track", "高置信且解释风险较低，可进入快速复核"
    if confidence >= 0.35 and project_risk <= 80 and not (fairness_flag and confidence < 0.75):
        return "regular_review", "进入常规人工复核"
    return "expert_review", "低置信、高风险或存在偏差审计信号，建议专家重点审查"


ROUTING_LABELS = {"fast_track": "快速复核", "regular_review": "常规人工复核", "expert_review": "专家重点审查"}


def ebm_local_terms(model, row_df: pd.DataFrame) -> list[dict]:
    local_exp = model.explain_local(row_df[FEATURE_COLUMNS])
    data = local_exp.data(0)
    rows = []
    for name, score, value in zip(data.get("names", []), data.get("scores", []), data.get("values", [])):
        rows.append({
            "term": str(name),
            "value": "" if value is None else str(value),
            "contribution": round(float(score), 4),
            "abs_contribution": round(abs(float(score)), 4),
        })
    rows.sort(key=lambda item: item["abs_contribution"], reverse=True)
    return rows


def fairness_share_from_terms(terms: list[dict]) -> float:
    total = sum(item["abs_contribution"] for item in terms) or 1.0
    sensitive = sum(
        item["abs_contribution"]
        for item in terms
        if any(f in item["term"] for f in SENSITIVE_OR_BIAS_AUDIT_FEATURES)
    )
    return sensitive / total


def describe_model_impact(contribution: float) -> str:
    if contribution > 0:
        return "提高资助概率"
    if contribution < 0:
        return "降低资助概率"
    return "几乎无影响"


def link_feature_to_input(feature: str, text: str, value, signals: dict) -> str:
    label = FEATURE_LABELS_ZH.get(feature, feature)
    if feature == "novelty_score":
        return f"原文「创新」等表述 {signals['innovation_count']} 处 → {label}={value:.1f}"
    if feature == "frontier_score" and signals["has_ai"]:
        return f"原文提及 AI/大模型等 → {label}={value:.1f}"
    if feature == "project_risk":
        extra = "；原文含缓释/应对（信任加分，见步骤4）" if signals.get("has_risk_mitigation") else ""
        return f"原文风险相关表述 → {label}={value:.1f}（越高=客观风险越大）{extra}"
    if feature == "institution_tier":
        return f"原文单位线索 → {label}={value}"
    if feature == "interdisciplinarity":
        return f"原文跨学科/交叉表述 → {label}={value:.1f}"
    if feature == "papers_5y":
        return f"原文论文/发表表述 → {label}={int(value)}"
    return f"步骤1 从原文推断 → {label}={value}"


def render_input_result_bridge(title: str, input_text: str, signals: dict, judgment: str, explanation: str):
    st.markdown(f"#### {title}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**您输入的原文（摘要）**")
        st.markdown(f"> {input_excerpt(input_text, 220)}")
        st.caption(f"识别关键词：{signals['keyword_sample']}")
    with c2:
        st.markdown("**当前系统判断**")
        st.markdown(judgment)
    st.markdown("**输入原文 → 当前判断 关联说明**")
    st.info(explanation)


def build_step3_evidence_cards(input_text: str, signals: dict, row: pd.Series, prob: float | None) -> list[dict]:
    cards = []
    quotes_unit = find_input_quotes(input_text, ["985", "双一流", "中科院", "依托", "大学", "团队", "人"])
    cards.append({
        "标题": "机构与团队条件",
        "原文引述": quotes_unit,
        "输入线索": f"依托单位：{row['institution_tier']}；团队规模={int(row['team_size'])}",
        "公开映射": "NSFC unit → institution_tier；participant_count → team_size",
        "对判断的支撑": (
            f"原文中关于单位与团队的描述（见引述）映射为 {row['institution_tier']}、团队 {int(row['team_size'])} 人。"
            f"{'与 OpenAlex/NSFC 中顶尖单位合作网络参照一致。' if signals['has_top_unit'] else '单位层级为一般参照，需结合项目质量综合判断。'}"
        ),
    })
    quotes_innov = find_input_quotes(input_text, ["创新", "风险", "缓释", "应对", "防范", "挑战"])
    risk = float(row["project_risk"])
    cards.append({
        "标题": "创新性与客观风险",
        "原文引述": quotes_innov,
        "输入线索": f"创新表述 {signals['innovation_count']} 次；风险披露={'有' if signals['has_risk'] else '无'}；缓释措施={'有' if signals.get('has_risk_mitigation') else '无'}",
        "公开映射": "keywords → novelty_score；funding/execution → project_risk",
        "对判断的支撑": (
            f"novelty_score={row['novelty_score']:.1f}，project_risk={risk:.1f}。"
            f"{'原文已披露风险并说明缓释（见引述）——信任评估加分；若 risk 仍偏高，AI 对资助概率仍可能保守。' if signals.get('has_risk_mitigation') else '请对照引述核对风险表述是否充分。'}"
        ),
    })
    if prob is not None:
        quotes_impact = find_input_quotes(input_text, ["引用", "影响", "成果", "论文", "国自然", "项目"])
        cards.append({
            "标题": "与步骤2 AI 概率的交叉验证",
            "原文引述": quotes_impact,
            "输入线索": f"步骤2 资助概率 {prob:.1%}",
            "公开映射": "OpenAlex cited_by_count/fwci → expected_impact；awards → prior_grants",
            "对判断的支撑": (
                f"expected_impact={row['expected_impact']:.1f}，prior_grants={int(row['prior_grants'])}。"
                f"原文引述中的产出/项目描述与 AI 概率 {prob:.1%} {'方向一致' if prob >= 0.5 else '存在张力，建议专家复核'}。"
            ),
        })
    quotes_discipline = find_input_quotes(input_text, ["人工智能", "信息", "医学", "工程", "学科", "领域"])
    cards.append({
        "标题": "学科与公开数据可追溯",
        "原文引述": quotes_discipline,
        "输入线索": f"学科={row['discipline']}；关键词={signals['keyword_sample']}",
        "公开映射": "OpenAlex topic → discipline, frontier_score；NSFC application_code → discipline",
        "对判断的支撑": (
            f"原文学科/领域表述（见引述）映射为 {row['discipline']}。"
            f"{'可与 OpenAlex/NSFC 公开主题对照，证据可追溯性强。' if signals['has_ai'] or signals['has_data'] else '建议补充公开文献或同类项目编号以增强可追溯性。'}"
        ),
    })
    return cards


def build_trust_dimension_details(input_text, signals, row, prob, confidence, local_terms):
    details = []
    top_terms = (local_terms or [])[:3]
    top_names = "、".join(t["term"] for t in top_terms) if top_terms else "（请先完成步骤2）"

    details.append({
        "维度": "可解释性",
        "评分": 5 if confidence >= 0.5 else (4 if confidence >= 0.25 else 3),
        "原文引述": find_input_quotes(input_text, ["创新", "方法", "模型", "团队", "数据"]),
        "映射特征/证据": f"步骤2 主要解释项：{top_names}；prob={prob:.1%}，置信度={confidence:.1%}",
        "评分理由": f"EBM 已给出局部解释；置信度 {confidence:.1%}。请对照下方原文与步骤2 表格逐项核验。",
        "评审提示": "核对原文关键句是否与解释项一一对应。",
    })
    details.append({
        "维度": "证据可追溯性",
        "评分": 5 if (signals["has_data"] or signals["has_ai"]) else 3,
        "原文引述": find_input_quotes(input_text, ["数据", "公开", "nsfc", "引用", "创新", "人工智能"]),
        "映射特征/证据": f"discipline={row['discipline']}，data_availability={row['data_availability']:.1f}",
        "评分理由": "原文含可映射公开数据的线索则可追溯性强，否则建议补充 DOI/项目编号。",
        "评审提示": "确认引述是否能在 NSFC/OpenAlex 找到同类参照。",
    })
    risk = float(row["project_risk"])
    if risk < 40:
        sr = 5
    elif risk < 65:
        sr = 4 if signals.get("has_risk_mitigation") else 3
    elif risk < 80:
        sr = 3 if signals.get("has_risk_mitigation") else 2
    else:
        sr = 2 if signals.get("has_risk_mitigation") else 1
    details.append({
        "维度": "任务风险",
        "评分": sr,
        "原文引述": find_input_quotes(input_text, ["风险", "缓释", "应对", "防范", "挑战"]),
        "映射特征/证据": f"project_risk={risk:.1f}",
        "评分理由": "评的是风险识别与管控能力；与步骤2 对资助概率的影响是不同问题。",
        "评审提示": "披露风险+缓释=良好表现；客观 risk 高仍可能使 AI 保守。",
    })
    papers, grants, hidx = int(row["papers_5y"]), int(row["prior_grants"]), int(row["h_index"])
    details.append({
        "维度": "研究者素养",
        "评分": min(5, max(2, (papers // 8) + (grants // 2) + (hidx // 5) + 1)),
        "原文引述": find_input_quotes(input_text, ["论文", "负责人", "团队", "项目", "h-index", "成果"]),
        "映射特征/证据": f"papers_5y={papers}，prior_grants={grants}，h_index={hidx}",
        "评分理由": "对照原文引述核实负责人履历与团队能力描述。",
        "评审提示": "数值来自关键词统计+规则，建议人工核实。",
    })
    tier = str(row["institution_tier"])
    details.append({
        "维度": "组织规范",
        "评分": min(5, max(2, 3 + (1 if "双一流" in tier else 0) + (1 if signals["has_ethics"] else 0))),
        "原文引述": find_input_quotes(input_text, ["伦理", "合规", "985", "依托", "规范"]),
        "映射特征/证据": f"institution_tier={tier}，ethics={row['ethics_compliance']:.0f}",
        "评分理由": "原文是否描述伦理审查与机构规范。",
        "评审提示": "高责任场景下组织规范是信任校准重要一环。",
    })
    return details


def render_new_application_workflow() -> None:
    st.subheader("人机协同科研决策信任机制 — 新申请端到端分析")
    st.caption("原文输入 → 22 特征 → EBM 预测 → 公开映射 → 五维度信任评估 → 协同分流 → 结构化报告")
    # Session state
    for key, default in [("new_app", None), ("input_text", ""), ("input_signals", {}), ("analysis_results", {})]:
        if key not in st.session_state:
            st.session_state[key] = default

    st.sidebar.header("分析流程导航")
    if st.session_state.get("input_text"):
        st.sidebar.markdown("**当前申请原文摘要**")
        st.sidebar.caption(input_excerpt(st.session_state.input_text, 100))
    page = st.sidebar.radio(
        "选择步骤",
        ["首页概览", "步骤1: 数据准备", "步骤2: 模型预测", "步骤3: 证据映射", "步骤4: 信任评估", "步骤5: 协同分流", "步骤6: 生成报告"],
    )

    # ========== HOME ==========
    if page == "首页概览":
        st.markdown("### 新申请端到端分析")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """
                **流程说明**  
                原文输入 → 22 维特征提取 → EBM 预测与局部解释 → 公开数据证据映射 →  
                五维度信任评估 → 协同分流 → 可下载结构化报告。
                """
            )
        with c2:
            st.markdown(
                """
                **理论对齐**  
                对应「预测—解释—校准—复核」闭环；步骤4 映射可解释性、证据可追溯性、  
                任务风险、研究者素养、组织规范五维信任信号。
                """
            )
        st.info("请从步骤1 粘贴申请 URL 或正文，再依次完成各步骤。步骤2/3/4 均会引用**您输入的原文片段**。")

    # ========== STEP 1 ==========
    elif page == "步骤1: 数据准备":
        st.header("步骤 1: 新申请数据准备")
        with st.form("input_form"):
            text_input = st.text_area("项目 URL 或正文（摘要、关键词、团队、经费等）", height=200)
            submitted = st.form_submit_button("提取 22 个特征")
        if submitted and text_input.strip():
            text_lower = text_input.lower()
            words = text_input.split()
            num_numbers = len([w for w in words if w.replace(".", "").isdigit() or "万" in w])
            if any(k in text_lower for k in ["985", "双一流", "顶尖"]):
                institution_tier = "双一流/国家级平台"
            elif any(k in text_lower for k in ["211", "重点", "中科院"]):
                institution_tier = "省部级重点单位"
            else:
                institution_tier = "普通高校/机构"
            overseas = "长期经历" if any(k in text_lower for k in ["长期", "postdoc"]) else (
                "短期交流" if any(k in text_lower for k in ["海外", "交流"]) else "无"
            )
            age_group = "资深" if any(k in text_lower for k in ["资深", "院士"]) else (
                "青年" if any(k in text_lower for k in ["青年", "博士"]) else "中青年"
            )
            if any(k in text_lower for k in ["人工智能", "机器学习", "信息"]):
                discipline = "信息科学"
            elif "医学" in text_input or "生命" in text_input:
                discipline = "生命科学"
            elif "跨学科" in text_input or "交叉" in text_input:
                discipline = "交叉学科"
            else:
                discipline = "信息科学"
            risk_base = 42 + (12 if institution_tier == "普通高校/机构" else 0)
            if "风险" in text_input:
                risk_base += -10 if any(k in text_input for k in ["缓释", "应对", "防范", "控制", "监测"]) else 6
            has_ai_frontier = any(k in text_lower for k in ["人工智能", "大模型", "前沿"])
            extracted = {
                "papers_5y": min(50, max(3, text_input.count("论文") * 3 + 5)),
                "citations_5y": min(500, max(10, text_input.count("引用") * 20 + 100)),
                "h_index": min(30, max(3, 5 + text_input.count("h-index"))),
                "research_years": min(20, max(2, text_input.count("年") // 2 + 4)),
                "prior_grants": min(10, max(0, text_input.count("国自然") + text_input.count("NSFC"))),
                "institution_tier": institution_tier,
                "overseas_experience": overseas,
                "age_group": age_group,
                "novelty_score": min(95, max(40, 50 + text_input.count("创新") * 8)),
                "feasibility_score": min(95, max(50, 85 - num_numbers * 2)),
                "frontier_score": min(95, max(40, 60 + (15 if has_ai_frontier else 0))),
                "methodology_score": min(95, max(50, 70 + text_input.count("方法") * 3)),
                "expected_impact": min(95, max(40, 55 + text_input.count("影响") * 4)),
                "budget_reasonableness": 85 if "万" in text_input else 65,
                "project_risk": max(20, min(90, risk_base)),
                "data_availability": 80 if "数据" in text_input else 65,
                "ethics_compliance": 90 if "伦理" in text_input else 75,
                "team_size": min(12, max(2, text_input.count("团队") * 2 + 3)),
                "interdisciplinarity": min(95, max(20, 40 + text_input.count("跨学科") * 10)),
                "collaboration_strength": min(95, max(35, 55 + text_input.count("合作") * 8)),
                "platform_support": 75 if "平台" in text_input or "实验室" in text_input else 60,
                "discipline": discipline,
            }
            df = pd.DataFrame([extracted])
            df["application_id"] = text_input.split("\n")[0][:20].replace("http", "url_")
            st.session_state.new_app = df
            st.session_state.input_text = text_input
            st.session_state.input_signals = parse_input_signals(text_input)
            st.success("特征提取完成")
            st.dataframe(df[FEATURE_COLUMNS], width="stretch")
            st.subheader("原文 → 特征映射（节选）")
            sig = st.session_state.input_signals
            for feat in ["novelty_score", "frontier_score", "project_risk", "institution_tier", "discipline"]:
                st.info(link_feature_to_input(feat, text_input, df[feat].iloc[0], sig))
        elif submitted:
            st.warning("请输入正文")

    # ========== STEP 2 ==========
    elif page == "步骤2: 模型预测":
        st.header("步骤 2: EBM 模型预测与局部解释")
        if st.session_state.new_app is None:
            st.warning("请先完成步骤 1")
        else:
            model = load_ebm_model()
            df = st.session_state.new_app
            input_text = st.session_state.get("input_text", "")
            signals = st.session_state.get("input_signals", parse_input_signals(input_text))

            with st.expander("📄 查看您提交的完整原文", expanded=False):
                st.text_area("完整原文", input_text, height=180, disabled=True)

            prob = float(model.predict_proba(df[FEATURE_COLUMNS])[:, 1][0])
            confidence = abs(prob - 0.5) * 2
            terms = ebm_local_terms(model, df)

            cols = st.columns(4)
            cols[0].metric("AI 预测概率", f"{prob:.2%}")
            cols[1].metric("置信度", f"{confidence:.2%}")
            cols[2].metric("解释项数量", len(terms))
            cols[3].metric("判别", "倾向资助" if prob >= 0.5 else "倾向不资助")

            verdict = (
                f"**倾向建议资助**（{prob:.1%}）" if prob >= 0.6 else
                f"**边界案例**（{prob:.1%}）" if prob >= 0.4 else
                f"**倾向不建议**（{prob:.1%}）"
            )
            render_input_result_bridge(
                "步骤2 总览：原文与 AI 预测",
                input_text,
                signals,
                verdict + f"\n\n置信度 {confidence:.1%}",
                "下表每一行均包含：**原文引述**、从原文映射的特征、EBM 对资助概率的影响。"
                "请重点对照「原文引述」列，确认 AI 是否准确理解了您的表述。",
            )

            contrib_rows = []
            for item in terms[:12]:
                matched = [f for f in FEATURE_COLUMNS if f in item["term"]]
                feat = matched[0] if matched else ""
                quote = quotes_for_feature(feat, input_text) if feat else find_input_quotes(input_text, [item["term"][:4]], 1)[0]
                val = df[feat].iloc[0] if feat in df.columns else item["value"]
                contrib_rows.append({
                    "EBM 解释项": item["term"],
                    "特征取值": item["value"],
                    "对资助概率的影响": describe_model_impact(item["contribution"]),
                    "影响强度": item["contribution"],
                    "原文引述": quote,
                    "原文→特征映射": link_feature_to_input(feat, input_text, val, signals) if feat else "多特征交互项",
                })

            st.subheader("局部解释表（含原文引述）")
            st.caption("右=提高资助概率，左=降低资助概率。project_risk 若显示「提高」表示该特征当前取值相对有利于概率（与特征名含义请结合「原文→特征映射」理解）。")
            local_df = pd.DataFrame(contrib_rows)
            st.dataframe(local_df, width="stretch", hide_index=True)

            plot_df = pd.DataFrame(terms[:8]).assign(label=[t["term"][:26] for t in terms[:8]])
            st.plotly_chart(
                px.bar(plot_df, x="contribution", y="label", orientation="h",
                       title="Top 8 特征对资助概率的影响（右=提高，左=降低）",
                       color="contribution", color_continuous_scale="RdBu"),
                width="stretch",
            )

            st.subheader("逐步解读：原文 → 特征 → 对资助概率的影响")
            for i, item in enumerate(terms[:6]):
                matched = [f for f in FEATURE_COLUMNS if f in item["term"]]
                feat = matched[0] if matched else None
                quote = quotes_for_feature(feat, input_text) if feat else find_input_quotes(input_text, [], 1)[0]
                impact = describe_model_impact(item["contribution"])
                with st.expander(f"{i+1}. {item['term']}（{impact}，强度 {item['contribution']:+.3f}）", expanded=i < 2):
                    st.markdown(f"**原文引述**：「{quote}」")
                    if feat:
                        st.markdown(f"**映射**：{link_feature_to_input(feat, input_text, df[feat].iloc[0], signals)}")
                    st.markdown(
                        f"**对当前预测的含义**：该解释项使资助概率**{impact}**。"
                        + (" 若与您的直觉不符，请检查原文是否被步骤1 正确理解。" if abs(item["contribution"]) > 0.3 else "")
                    )

            top_pos = [t for t in terms if t["contribution"] > 0][:2]
            top_neg = [t for t in terms if t["contribution"] < 0][:2]
            if top_pos:
                st.success("主要提高资助概率的因素：" + "；".join(f"{t['term']}({t['contribution']:+.3f})" for t in top_pos))
            if top_neg:
                st.warning(
                    "主要降低资助概率的因素：" + "；".join(f"{t['term']}({t['contribution']:+.3f})" for t in top_neg)
                    + "。请对照上表「原文引述」核实。"
                )

            st.session_state.analysis_results.update({
                "prob": prob, "confidence": confidence, "local_terms": terms, "local_contrib": local_df,
            })

    # ========== STEP 3 ==========
    elif page == "步骤3: 证据映射":
        st.header("步骤 3: 公开数据映射与证据可追溯性")
        if st.session_state.new_app is None:
            st.warning("请先完成步骤 1")
        else:
            df = st.session_state.new_app
            row = df.iloc[0]
            input_text = st.session_state.get("input_text", "")
            signals = st.session_state.get("input_signals", parse_input_signals(input_text))
            prob = st.session_state.analysis_results.get("prob")
            mapping_df = load_csv("nsfc_public_enriched_mapping.csv")

            with st.expander("📄 查看您提交的完整原文", expanded=False):
                st.text_area("完整原文", input_text, height=180, disabled=True)

            evidence_cards = build_step3_evidence_cards(input_text, signals, row, prob)
            render_input_result_bridge(
                "步骤3 总览：原文与公开证据",
                input_text,
                signals,
                f"证据可追溯性：{'强' if signals['has_data'] or signals['has_ai'] else '中'}",
                "每张证据卡片均包含**原文引述**，说明公开数据规则如何支撑步骤2 的 AI 判断。",
            )

            st.subheader("针对本申请的证据卡片（含原文引述）")
            for card in evidence_cards:
                with st.expander(card["标题"], expanded=True):
                    st.markdown("**① 原文依据（您输入的具体内容）**")
                    for q in card["原文引述"]:
                        st.markdown(f"> 「{q}」")
                    st.markdown(f"**② 从原文提取的线索**：{card['输入线索']}")
                    st.markdown(f"**③ 公开映射规则**：{card['公开映射']}")
                    st.markdown("**④ 对当前判断的支撑**")
                    st.info(card["对判断的支撑"])

            st.subheader("完整映射规则表（前 8 条）")
            st.dataframe(mapping_df.head(8), width="stretch", hide_index=True)
            st.session_state.analysis_results["evidence_cards"] = evidence_cards

    # ========== STEP 4 ==========
    elif page == "步骤4: 信任评估":
        st.header("步骤 4: 五维度信任评估")
        if st.session_state.new_app is None:
            st.warning("请先完成步骤1-3")
        else:
            row = st.session_state.new_app.iloc[0]
            input_text = st.session_state.get("input_text", "")
            signals = st.session_state.get("input_signals", parse_input_signals(input_text))
            prob = st.session_state.analysis_results.get("prob", 0.5)
            confidence = st.session_state.analysis_results.get("confidence", 0.0)
            terms = st.session_state.analysis_results.get("local_terms", [])
            trust_details = build_trust_dimension_details(input_text, signals, row, prob, confidence, terms)
            scores = {d["维度"]: d["评分"] for d in trust_details}
            avg = sum(scores.values()) / len(scores)
            render_input_result_bridge(
                "步骤4 总览", input_text, signals,
                f"综合信任 {avg:.1f}/5", "每维度含原文引述与评分理由。",
            )
            st.session_state.analysis_results["trust_scores"] = scores
            st.session_state.analysis_results["trust_details"] = trust_details
            fig = px.line_polar(pd.DataFrame({"维度": list(scores), "评分": list(scores.values())}),
                                r="评分", theta="维度", line_close=True, title="信任维度雷达图")
            st.plotly_chart(fig, width="stretch")
            for d in trust_details:
                st.markdown(f"### {d['维度']}：{d['评分']}/5")
                for q in d["原文引述"]:
                    st.markdown(f"> 「{q}」")
                st.info(d["评分理由"])
                st.divider()

    # ========== STEP 5 ==========
    elif page == "步骤5: 协同分流":
        st.header("步骤 5: 协同分流决策")
        if st.session_state.new_app is None:
            st.warning("请先完成前序步骤")
        else:
            input_text = st.session_state.get("input_text", "")
            signals = st.session_state.get("input_signals", parse_input_signals(input_text))
            prob = st.session_state.analysis_results.get("prob", 0.5)
            confidence = st.session_state.analysis_results.get("confidence", 0.0)
            risk = float(st.session_state.new_app["project_risk"].iloc[0])
            terms = st.session_state.analysis_results.get("local_terms", [])
            fs = fairness_share_from_terms(terms) if terms else 0.0
            routing, reason = route_case(prob, risk, fs)
            quotes_risk = find_input_quotes(input_text, ["风险", "缓释", "合作", "创新"])
            render_input_result_bridge(
                "步骤5 总览", input_text, signals,
                f"**{ROUTING_LABELS[routing]}** — {reason}",
                "分流信号均回溯至下方原文引述与步骤2 解释。",
            )
            st.dataframe(pd.DataFrame([
                {"信号": "置信度", "值": f"{confidence:.1%}", "原文关联": f"由原文映射特征经 EBM 得 prob={prob:.1%}"},
                {"信号": "项目风险", "值": f"{risk:.1f}", "原文关联": quotes_risk[0]},
                {"信号": "偏差审计占比", "值": f"{fs:.1%}", "原文关联": f"单位={st.session_state.new_app['institution_tier'].iloc[0]}"},
            ]), width="stretch", hide_index=True)
            st.session_state.analysis_results["routing"] = routing
            st.session_state.analysis_results["routing_reason"] = reason

    # ========== STEP 6 ==========
    elif page == "步骤6: 生成报告":
        st.header("步骤 6: 结构化分析报告")
        if st.session_state.new_app is None:
            st.warning("请先完成前序步骤")
        else:
            ar = st.session_state.analysis_results
            trust_md = ""
            for d in ar.get("trust_details", []):
                trust_md += f"\n### {d['维度']} ({d['评分']}/5)\n"
                for q in d.get("原文引述", []):
                    trust_md += f"- 原文：「{q}」\n"
                trust_md += f"- {d['评分理由']}\n"
            report = f"""# 新国自然申请信任机制分析报告

    **项目**：{st.session_state.new_app['application_id'].iloc[0]}
    **原文摘要**：{input_excerpt(st.session_state.get('input_text',''), 300)}

    ## AI 预测
    - 概率：{ar.get('prob',0):.2%}
    - 分流：{ROUTING_LABELS.get(ar.get('routing','regular_review'),'常规复核')}

    ## 信任维度
    {trust_md}

    ## 专家决策
    - [ ] 采纳  - [ ] 修正  - [ ] 复核  - [ ] 否决
    """
            st.markdown(report)
            st.download_button("下载报告", report, file_name="trust_report.md")

    st.caption("数据来源：outputs/ 与 models/ | 公开元数据仅作外部效度参照")


if __name__ == "__main__":
    import streamlit as st
    st.set_page_config(page_title="新国自然申请分析系统", layout="wide")
    render_new_application_workflow()
