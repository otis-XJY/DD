import json
from pathlib import Path

import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / 'data' / 'research_trust_data.csv'
OUTPUT_DIR = BASE_DIR / 'outputs'
STATIC_GEN_DIR = BASE_DIR / 'src' / 'static' / 'generated'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATIC_GEN_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    feature_cols = [
        'prior_output', 'academic_reputation', 'team_diversity', 'proposal_innovation',
        'budget_rationality', 'evidence_traceability', 'explainability', 'task_risk',
        'ai_literacy', 'org_norm_clarity', 'human_review_score', 'ai_recommendation', 'trust_score'
    ]
    X = df[feature_cols]
    y = df['funding_decision']
    return df, X, y, feature_cols


def train_ebm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    model = ExplainableBoostingClassifier(random_state=42, interactions=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        'accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
        'roc_auc': round(float(roc_auc_score(y_test, y_prob)), 4),
        'train_size': int(len(X_train)),
        'test_size': int(len(X_test)),
    }
    return model, metrics


def serialize_global_explanations(model):
    global_exp = model.explain_global()
    internal = global_exp.data()
    scores = []
    for name, score in zip(internal['names'], internal['scores']):
        if score is None:
            continue
        scores.append({'feature': name, 'importance': round(float(score), 4)})
    scores = sorted(scores, key=lambda x: x['importance'], reverse=True)

    shape_functions = []
    for idx, name in enumerate(internal['names']):
        detail = global_exp.data(idx)
        if not detail:
            continue
        names = detail.get('names', [])
        values = detail.get('scores', [])
        cleaned_points = []
        for x, y in zip(list(names), list(values)):
            if x is None or y is None:
                continue
            try:
                x_val = float(x)
            except (TypeError, ValueError):
                x_val = str(x)
            cleaned_points.append({'x': x_val, 'y': round(float(y), 4)})
        if cleaned_points:
            shape_functions.append({'feature': name, 'points': cleaned_points[:30]})
    return scores, shape_functions


def build_case_explanations(model, df, X):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = model.predict(X)
    enriched = df.copy()
    enriched['predicted_probability'] = probabilities
    enriched['predicted_label'] = predictions
    top_cases = enriched.sort_values('predicted_probability', ascending=False).head(6)
    low_cases = enriched.sort_values('predicted_probability', ascending=True).head(6)
    return {
        'high_confidence_cases': top_cases[[
            'project_id', 'discipline', 'predicted_probability', 'trust_score',
            'evidence_traceability', 'explainability', 'task_risk', 'funding_decision'
        ]].round(4).to_dict(orient='records'),
        'low_confidence_cases': low_cases[[
            'project_id', 'discipline', 'predicted_probability', 'trust_score',
            'evidence_traceability', 'explainability', 'task_risk', 'funding_decision'
        ]].round(4).to_dict(orient='records'),
    }


def export_outputs(df, feature_scores, shape_functions, metrics, cases):
    pd.DataFrame(feature_scores).to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False, encoding='utf-8-sig')

    shape_rows = []
    for item in shape_functions:
        for point in item['points']:
            shape_rows.append({'feature': item['feature'], 'x': point['x'], 'y': point['y']})
    pd.DataFrame(shape_rows).to_csv(OUTPUT_DIR / 'shape_functions.csv', index=False, encoding='utf-8-sig')

    summary = {
        'project_theme': '人机协同科研信任机制',
        'model': 'ExplainableBoostingClassifier (EBM)',
        'metrics': metrics,
        'top_features': feature_scores[:8],
        'shape_functions': shape_functions[:5],
        'case_analysis': cases,
        'dataset_overview': {
            'samples': int(len(df)),
            'funded_ratio': round(float(df['funding_decision'].mean()), 4),
            'avg_trust_score': round(float(df['trust_score'].mean()), 4),
            'avg_explainability': round(float(df['explainability'].mean()), 4),
            'avg_task_risk': round(float(df['task_risk'].mean()), 4),
        }
    }
    (OUTPUT_DIR / 'model_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    df, X, y, feature_cols = load_data()
    model, metrics = train_ebm(X, y)
    feature_scores, shape_functions = serialize_global_explanations(model)
    cases = build_case_explanations(model, df, X)
    export_outputs(df, feature_scores, shape_functions, metrics, cases)
    print('模型训练完成')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
