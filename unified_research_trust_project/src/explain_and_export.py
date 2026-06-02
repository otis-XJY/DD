import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import (
    DATA_DIR,
    FEATURE_COLUMNS,
    FEATURE_GROUPS,
    FEATURE_LABELS_ZH,
    MODEL_DIR,
    NUMERIC_FEATURES,
    OUTPUT_DIR,
    SENSITIVE_OR_BIAS_AUDIT_FEATURES,
    TARGET,
    TRUE_PROB,
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def feature_to_group() -> dict[str, str]:
    mapping = {}
    for group, features in FEATURE_GROUPS.items():
        for feature in features:
            mapping[feature] = group
    return mapping


def term_features(term_name: str) -> list[str]:
    hits = []
    for feature in FEATURE_COLUMNS:
        if feature in term_name:
            hits.append(feature)
    return hits


def export_global_importance(model) -> list[dict]:
    names = list(getattr(model, "term_names_", []))
    if hasattr(model, "term_importances"):
        importances = model.term_importances()
    else:
        importances = np.zeros(len(names))

    records = []
    for name, importance in zip(names, importances):
        features = term_features(str(name))
        records.append(
            {
                "term": str(name),
                "features": features,
                "feature_labels_zh": [FEATURE_LABELS_ZH.get(f, f) for f in features],
                "importance": round(float(importance), 6),
            }
        )
    records.sort(key=lambda row: row["importance"], reverse=True)
    return records


def export_group_importance(global_records: list[dict]) -> list[dict]:
    mapping = feature_to_group()
    totals = {group: 0.0 for group in FEATURE_GROUPS}
    totals["cross_group_interaction"] = 0.0

    for record in global_records:
        groups = sorted({mapping.get(feature, "unknown") for feature in record["features"]})
        groups = [group for group in groups if group != "unknown"]
        if not groups:
            continue
        if len(groups) > 1:
            totals["cross_group_interaction"] += record["importance"]
        share = record["importance"] / len(groups)
        for group in groups:
            totals[group] += share

    total = sum(totals.values()) or 1.0
    return [
        {
            "group": group,
            "importance": round(float(value), 6),
            "share": round(float(value / total), 4),
        }
        for group, value in sorted(totals.items(), key=lambda item: item[1], reverse=True)
        if value > 0
    ]


def make_reference_case(df: pd.DataFrame) -> dict:
    row = {}
    for col in FEATURE_COLUMNS:
        if col in NUMERIC_FEATURES:
            row[col] = float(df[col].median())
        else:
            row[col] = str(df[col].mode(dropna=True).iloc[0])
    return row


def export_response_curves(model, df: pd.DataFrame, features: list[str]) -> dict:
    reference = make_reference_case(df)
    payload = {}
    for feature in features:
        if feature in NUMERIC_FEATURES:
            low, high = df[feature].quantile([0.05, 0.95])
            grid = np.linspace(low, high, 60)
            probe = pd.DataFrame([reference] * len(grid))
            probe[feature] = grid
            prob = model.predict_proba(probe[FEATURE_COLUMNS])[:, 1]
            payload[feature] = {
                "feature_label_zh": FEATURE_LABELS_ZH.get(feature, feature),
                "kind": "numeric",
                "points": [
                    {"x": round(float(x), 4), "funding_probability": round(float(p), 5)}
                    for x, p in zip(grid, prob)
                ],
            }
        else:
            categories = sorted(df[feature].dropna().astype(str).unique().tolist())
            probe = pd.DataFrame([reference] * len(categories))
            probe[feature] = categories
            prob = model.predict_proba(probe[FEATURE_COLUMNS])[:, 1]
            payload[feature] = {
                "feature_label_zh": FEATURE_LABELS_ZH.get(feature, feature),
                "kind": "categorical",
                "points": [
                    {"x": str(x), "funding_probability": round(float(p), 5)}
                    for x, p in zip(categories, prob)
                ],
            }
    return payload


def local_records(model, x: pd.DataFrame, y: pd.Series | None = None) -> list[dict]:
    local_exp = model.explain_local(x, y)
    records = []
    for i in range(len(x)):
        data = local_exp.data(i)
        names = data.get("names", [])
        scores = data.get("scores", [])
        values = data.get("values", [""] * len(names))
        rows = []
        for name, score, value in zip(names, scores, values):
            rows.append(
                {
                    "term": str(name),
                    "value": "" if value is None else str(value),
                    "contribution": round(float(score), 6),
                    "abs_contribution": round(abs(float(score)), 6),
                }
            )
        rows.sort(key=lambda item: item["abs_contribution"], reverse=True)
        records.append({"ranked_terms": rows})
    return records


def choose_case_indices(df: pd.DataFrame, probs: np.ndarray) -> list[int]:
    candidates = {
        "highest_probability": int(np.argmax(probs)),
        "lowest_probability": int(np.argmin(probs)),
        "borderline_case": int(np.argmin(np.abs(probs - 0.5))),
        "high_risk_case": int(df["project_risk"].idxmax()),
    }
    seen = []
    for idx in candidates.values():
        if idx not in seen:
            seen.append(idx)
    return seen


def route_case(prob: float, project_risk: float, fairness_share: float) -> tuple[str, str]:
    confidence = abs(prob - 0.5) * 2
    fairness_flag = fairness_share >= 0.25
    if confidence >= 0.65 and project_risk <= 60 and not fairness_flag:
        return "fast_track", "高置信且解释风险较低，可进入快速复核"
    if confidence >= 0.35 and project_risk <= 80 and not (fairness_flag and confidence < 0.75):
        return "regular_review", "进入常规人工复核"
    return "expert_review", "低置信、高风险或存在偏差审计信号，建议专家重点审查"


def export_case_routing(model, df: pd.DataFrame) -> pd.DataFrame:
    x = df[FEATURE_COLUMNS]
    probs = model.predict_proba(x)[:, 1]
    local = local_records(model, x, df[TARGET] if TARGET in df else None)

    rows = []
    for i, row in df.reset_index(drop=True).iterrows():
        terms = local[i]["ranked_terms"]
        total_abs = sum(item["abs_contribution"] for item in terms) or 1.0
        sensitive_abs = sum(
            item["abs_contribution"]
            for item in terms
            if any(feature in item["term"] for feature in SENSITIVE_OR_BIAS_AUDIT_FEATURES)
        )
        fairness_share = sensitive_abs / total_abs
        route, reason = route_case(float(probs[i]), float(row["project_risk"]), fairness_share)
        rows.append(
            {
                "application_id": row["application_id"],
                "discipline": row["discipline"],
                "predicted_funding_probability": round(float(probs[i]), 5),
                "ai_recommendation": "建议资助" if probs[i] >= 0.5 else "建议不资助",
                "confidence": round(float(abs(probs[i] - 0.5) * 2), 5),
                "project_risk": round(float(row["project_risk"]), 2),
                "fairness_audit_share": round(float(fairness_share), 5),
                "routing": route,
                "routing_reason": reason,
                "true_label": int(row[TARGET]),
                "true_funding_probability": round(float(row[TRUE_PROB]), 5) if TRUE_PROB in row else None,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_DIR / "applications.csv")
    parser.add_argument("--model", type=Path, default=MODEL_DIR / "ebm_model.joblib")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)
    model = joblib.load(args.model)

    global_importance = export_global_importance(model)
    group_importance = export_group_importance(global_importance)

    key_features = [
        "novelty_score",
        "feasibility_score",
        "budget_reasonableness",
        "project_risk",
        "research_years",
        "interdisciplinarity",
        "institution_tier",
    ]
    response_curves = export_response_curves(model, df, key_features)

    probs = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    case_indices = choose_case_indices(df, probs)
    case_x = df.iloc[case_indices][FEATURE_COLUMNS]
    case_y = df.iloc[case_indices][TARGET]
    local = local_records(model, case_x, case_y)
    local_payload = []
    for idx, explanation in zip(case_indices, local):
        row = df.iloc[idx]
        local_payload.append(
            {
                "application_id": row["application_id"],
                "discipline": row["discipline"],
                "predicted_funding_probability": round(float(probs[idx]), 5),
                "ai_recommendation": "建议资助" if probs[idx] >= 0.5 else "建议不资助",
                "true_label": int(row[TARGET]),
                "top_positive": [t for t in explanation["ranked_terms"] if t["contribution"] > 0][:8],
                "top_negative": [t for t in explanation["ranked_terms"] if t["contribution"] < 0][:8],
                "all_terms": explanation["ranked_terms"][:20],
            }
        )

    routing = export_case_routing(model, df)
    routing.to_csv(OUTPUT_DIR / "case_routing.csv", index=False, encoding="utf-8-sig")

    write_json(OUTPUT_DIR / "global_importance.json", global_importance)
    write_json(OUTPUT_DIR / "group_importance.json", group_importance)
    write_json(OUTPUT_DIR / "response_curves.json", response_curves)
    write_json(OUTPUT_DIR / "local_explanations.json", local_payload)

    summary = {
        "global_top_5": global_importance[:5],
        "group_importance": group_importance,
        "routing_counts": routing["routing"].value_counts().to_dict(),
    }
    write_json(OUTPUT_DIR / "explanation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
