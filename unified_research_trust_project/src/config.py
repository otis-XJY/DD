from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

RANDOM_SEED = 42

TARGET = "funded"
TRUE_PROB = "true_funding_probability"

FEATURE_GROUPS = {
    "applicant_profile": [
        "papers_5y",
        "citations_5y",
        "h_index",
        "research_years",
        "prior_grants",
        "institution_tier",
        "overseas_experience",
        "age_group",
    ],
    "project_quality": [
        "novelty_score",
        "feasibility_score",
        "frontier_score",
        "methodology_score",
        "expected_impact",
        "budget_reasonableness",
        "project_risk",
        "data_availability",
        "ethics_compliance",
    ],
    "team_condition": [
        "team_size",
        "interdisciplinarity",
        "collaboration_strength",
        "platform_support",
    ],
    "domain_context": [
        "discipline",
    ],
}

SENSITIVE_OR_BIAS_AUDIT_FEATURES = {
    "institution_tier",
    "overseas_experience",
    "age_group",
}

NUMERIC_FEATURES = [
    "papers_5y",
    "citations_5y",
    "h_index",
    "research_years",
    "prior_grants",
    "novelty_score",
    "feasibility_score",
    "frontier_score",
    "methodology_score",
    "expected_impact",
    "budget_reasonableness",
    "project_risk",
    "data_availability",
    "ethics_compliance",
    "team_size",
    "interdisciplinarity",
    "collaboration_strength",
    "platform_support",
]

CATEGORICAL_FEATURES = [
    "institution_tier",
    "overseas_experience",
    "age_group",
    "discipline",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FEATURE_LABELS_ZH = {
    "papers_5y": "近五年论文数",
    "citations_5y": "近五年引用数",
    "h_index": "H-index",
    "research_years": "研究年限",
    "prior_grants": "既往项目数",
    "institution_tier": "单位层次",
    "overseas_experience": "海外经历",
    "age_group": "年龄阶段",
    "novelty_score": "创新性评分",
    "feasibility_score": "可行性评分",
    "frontier_score": "前沿度评分",
    "methodology_score": "方法成熟度",
    "expected_impact": "预期影响力",
    "budget_reasonableness": "经费合理性",
    "project_risk": "项目风险",
    "data_availability": "数据可得性",
    "ethics_compliance": "伦理合规性",
    "team_size": "团队规模",
    "interdisciplinarity": "跨学科程度",
    "collaboration_strength": "合作网络强度",
    "platform_support": "平台支撑水平",
    "discipline": "学科类别",
}

NSFC_SUBJECT_CODES = [
    "A01", "C01", "C05", "C06", "E01", "F01", "G01", "H01",
]

INSTITUTION_TIERS = {
    "C9": "顶尖",
    "985": "一流",
    "211": "重点",
    "双一流/国家级平台": "一流",
    "省部级重点单位": "重点",
    "普通高校/机构": "普通",
}
