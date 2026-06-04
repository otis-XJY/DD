# DD — 人机协同科研决策信任机制

本仓库包含三个相关子项目，**请以 [`unified_research_trust_project`](unified_research_trust_project/) 作为唯一主交付入口**进行安装、复现与答辩演示。

| 目录 | 说明 |
|------|------|
| **[unified_research_trust_project](unified_research_trust_project/)** | **核心**：统一 Streamlit 应用 + 完整 ML 流水线 + 文档 |
| [project](project/) | 参考：早期可解释模型与六标签页原型 |
| [human_ai_trust_project](human_ai_trust_project/) | 参考：信任理论框架与 Flask 静态展示 |

## 快速开始

```bash
cd unified_research_trust_project
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/run_pipeline.py          # 或见子项目 README 完整步骤
streamlit run app/streamlit_app.py
```

详细说明、实验结果与 FAQ 见 **[unified_research_trust_project/README.md](unified_research_trust_project/README.md)**。
