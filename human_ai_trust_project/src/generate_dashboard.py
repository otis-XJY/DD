import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / 'data' / 'research_trust_data.csv'
OUTPUT_DIR = BASE_DIR / 'outputs'
TEMPLATE_DIR = BASE_DIR / 'src' / 'templates'
STATIC_GEN_DIR = BASE_DIR / 'src' / 'static' / 'generated'
WEB_DIR = BASE_DIR / 'web'
WEB_DIR.mkdir(parents=True, exist_ok=True)
STATIC_GEN_DIR.mkdir(parents=True, exist_ok=True)


def save_html(fig, name):
    path = STATIC_GEN_DIR / name
    fig.write_html(str(path), include_plotlyjs='cdn', full_html=False)
    return f'static/generated/{name}'


def build_figures(df, summary):
    feature_df = pd.DataFrame(summary['top_features'])
    fig1 = px.bar(
        feature_df.head(8),
        x='importance', y='feature', orientation='h',
        title='全局特征重要性',
        color='importance', color_continuous_scale='Blues'
    )
    fig1.update_layout(template='plotly_white', yaxis={'categoryorder': 'total ascending'})

    fig2 = px.scatter(
        df,
        x='explainability', y='trust_score', size='adoption_probability', color='discipline',
        hover_name='project_id', title='可解释性与信任得分关系'
    )
    fig2.update_layout(template='plotly_white')

    risk_df = df.copy()
    risk_df['funding_decision_label'] = risk_df['funding_decision'].map({0: '未资助', 1: '资助'})
    fig3 = px.box(
        risk_df,
        x='funding_decision_label', y='task_risk', color='funding_decision_label',
        title='任务风险与资助决策分布',
        labels={'funding_decision_label': '资助决策', 'task_risk': '任务风险'}
    )
    fig3.update_layout(template='plotly_white', showlegend=False)

    avg_df = df.groupby('discipline', as_index=False)[['trust_score', 'explainability', 'evidence_traceability']].mean()
    radar = go.Figure()
    for _, row in avg_df.head(6).iterrows():
        radar.add_trace(go.Scatterpolar(
            r=[float(row['trust_score']), float(row['explainability']), float(row['evidence_traceability'])],
            theta=['信任得分', '可解释性', '证据可追溯性'],
            fill='toself',
            name=row['discipline']
        ))
    radar.update_layout(
        template='plotly_white',
        title='学科维度信任画像雷达图',
        polar=dict(radialaxis=dict(visible=True, range=[0, 1]))
    )

    shape_functions = summary['shape_functions']
    line_fig = go.Figure()
    for item in shape_functions[:3]:
        x_vals = [float(p['x']) for p in item['points'] if isinstance(p['x'], (int, float)) or str(p['x']).replace('.', '', 1).isdigit()]
        y_vals = [float(p['y']) for p in item['points'][:len(x_vals)]]
        line_fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines+markers', name=item['feature']))
    line_fig.update_layout(template='plotly_white', title='关键特征响应曲线', xaxis_title='特征值', yaxis_title='对模型输出的贡献')

    return {
        'feature_importance': save_html(fig1, 'feature_importance.html'),
        'trust_scatter': save_html(fig2, 'trust_scatter.html'),
        'risk_box': save_html(fig3, 'risk_box.html'),
        'discipline_radar': save_html(radar, 'discipline_radar.html'),
        'shape_curve': save_html(line_fig, 'shape_curve.html'),
    }


def render_page(summary, figures):
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    template = env.get_template('index.html.j2')
    html = template.render(summary=summary, figures=figures)
    output_path = WEB_DIR / 'index.html'
    output_path.write_text(html, encoding='utf-8')
    return output_path


if __name__ == '__main__':
    df = pd.read_csv(DATA_PATH)
    summary = json.loads((OUTPUT_DIR / 'model_summary.json').read_text(encoding='utf-8'))
    figures = build_figures(df, summary)
    output = render_page(summary, figures)
    print(f'网页已生成: {output}')
