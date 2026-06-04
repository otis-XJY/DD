# 人机协同科研决策信任机制（统一版）

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

面向「智能决策理论与方法」课程与「人机协同情境下的科研决策信任机制」研究的**可运行统一原型**。本项目以 `unified_research_trust_project` 为交付核心，融合同仓库中 `project`（可解释模型与协同分流流水线）与 `human_ai_trust_project`（信任理论框架与展示形态）的设计，形成「理论说明 + 批量解释面板 + 新申请端到端分析」的一体化系统。

> **定位**：不是替代专家的自动审批系统，而是 **AI 参谋、专家决策** 的可解释决策支持原型——说明 AI 如何建议、为何如此建议，以及专家如何据此校准信任与复核强度。

---

## 核心能力

| 模块 | 说明 |
|------|------|
| 研究概览与理论 | 四层人机协同框架、五维度信任机制、数据与模型策略说明 |
| 模型解释面板 | 三模型对比、全局/组别解释、个案证据、响应曲线、协同分流、公开数据映射 |
| 新申请端到端分析 | 文本/结构化输入 → 特征推断 → EBM 预测 → 局部解释 → 分流建议 → 证据归档 |
| 离线流水线 | 半仿真数据生成、OpenAlex 元数据参照、训练、解释导出、完整性校验 |

**理论闭环**：预测 → 解释 → 校准 → 复核

**数据策略**：半仿真数据监督训练 + OpenAlex / NSFC 等公开元数据支撑特征现实性

---

## 项目结构

```text
unified_research_trust_project/
├── README.md                 # 本文件：交接、环境、复现、FAQ
├── LICENSE                   # MIT
├── requirements.txt          # pip 依赖（推荐）
├── environment.yml           # conda 可选环境
├── Makefile                  # 常用命令快捷入口（Linux/macOS）
│
├── app/                      # Streamlit 统一可视化
│   ├── streamlit_app.py      # 主入口（三模块导航）
│   ├── batch_explain_panel.py
│   ├── new_application_analyzer.py
│   └── ui_styles.py
│
├── src/                      # 数据处理与建模
│   ├── config.py             # 路径、特征、中文标签、分组
│   ├── generate_data.py      # 半仿真科研项目申请数据
│   ├── download_public_reference.py
│   ├── train_models.py       # LR / EBM / RF 三模型训练
│   ├── explain_and_export.py # 全局/局部解释与分流导出
│   ├── train_model_ebm_trust.py  # 信任维度仿真数据 EBM（补充实验）
│   ├── run_pipeline.py       # 一键：生成 → 训练 → 解释
│   └── validate_project.py   # 产物完整性校验
│
├── data/
│   ├── applications.csv      # 主训练集（半仿真）
│   ├── research_trust_data.csv
│   ├── adult_income_openml.csv
│   ├── openalex_reference_raw.json
│   └── openalex_reference_sample.csv
│
├── models/                   # 训练后生成（默认不纳入版本库）
│   ├── ebm_model.joblib
│   ├── logistic_regression_baseline.joblib
│   └── random_forest_baseline.joblib
│
├── outputs/                  # 指标、解释、分流、映射等 JSON/CSV
│
└── docs/
    ├── 技术路线与复现实验说明.md
    ├── 核心模型与算法报告.md
    └── open_source_links.md
```

### 与同仓库其他目录的关系

| 目录 | 角色 |
|------|------|
| **`unified_research_trust_project`** | **主交付**：统一 UI + 完整流水线 + 文档 |
| `project` | 参考：早期 Streamlit 六标签页与 `validate_project` 设计 |
| `human_ai_trust_project` | 参考：信任理论叙述、Flask 静态页、公开数据演示 |

---

## 环境要求

- **Python** 3.11（推荐；3.10+ 一般可用）
- 约 **2 GB** 磁盘空间（含依赖与 `outputs`）
- 可选：访问 OpenAlex API 时需网络（`download_public_reference.py`）

### 方式一：venv + pip（推荐，Linux/macOS）

```bash
cd unified_research_trust_project
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 方式二：conda

```bash
cd unified_research_trust_project
conda env create -f environment.yml
conda activate unified_research_trust
```

### 方式三：Makefile（需已安装 `make`）

```bash
cd unified_research_trust_project
make install    # 创建 .venv 并安装依赖
make pipeline   # 完整离线流水线
make app        # 启动 Streamlit
```

---

## 快速开始

### 1. 完整复现（推荐首次运行）

在项目根目录执行：

```bash
# 生成 1500 条半仿真申请数据
python src/generate_data.py --n-samples 1500

# 下载 OpenAlex 参照元数据（需网络，可跳过见下方「快速模式」）
python src/download_public_reference.py --per-page 80

# 训练三模型并写入 models/
python src/train_models.py

# 导出解释与协同分流
python src/explain_and_export.py

# 校验关键产物
python src/validate_project.py
```

或使用一键入口：

```bash
python src/run_pipeline.py
# 不含 OpenAlex 下载；完整复现请按上表逐步执行
```

**快速模式**（仅重训模型与解释，不拉取 OpenAlex）：

```bash
python src/generate_data.py --n-samples 1500
python src/train_models.py
python src/explain_and_export.py
```

### 2. 启动可视化应用

```bash
streamlit run app/streamlit_app.py
```

浏览器打开终端提示的本地地址（通常为 `http://localhost:8501`）。

**侧边栏模块**：

1. **研究概览与理论** — 框架、信任维度、方法论说明  
2. **模型解释面板** — 模型对比、全局/个案解释、响应曲线、分流、公开映射  
3. **新申请端到端分析** — 粘贴申请书摘要或填写特征，获得预测与解释链  

### 3. 补充实验：信任维度 EBM

```bash
python src/train_model_ebm_trust.py
```

基于 `data/research_trust_data.csv` 输出 `feature_importance.csv`、`shape_functions.csv` 等，用于五维度信任机制的对照展示。

---

## 主要脚本说明

| 脚本 | 作用 |
|------|------|
| `src/config.py` | 特征列、特征组、敏感/偏差审计字段、中文名映射 |
| `src/generate_data.py` | 规则 + 非线性交互 + 噪声生成 `applications.csv` |
| `src/download_public_reference.py` | OpenAlex Works API → 真实字段与仿真特征映射 |
| `src/train_models.py` | Logistic Regression、EBM、Random Forest |
| `src/explain_and_export.py` | 全局/组别重要性、响应曲线、局部解释、`case_routing.csv` |
| `src/run_pipeline.py` | 生成数据 → 训练 → 解释（不含 OpenAlex） |
| `src/validate_project.py` | 检查必需文件、指标、分流类别与映射行数 |

---

## 当前实验结果（默认 1500 样本）

资助率约 **25.53%**。

| 模型 | Accuracy | AUC | Brier | Log Loss |
|------|---------:|----:|------:|---------:|
| Logistic Regression（透明线性基线） | 0.7067 | 0.7997 | 0.1810 | 0.5346 |
| **Explainable Boosting Machine（主模型）** | **0.7813** | 0.7610 | **0.1554** | **0.4755** |
| Random Forest（黑盒对照） | 0.7653 | 0.7820 | 0.1593 | 0.4884 |

**解读要点**：

- EBM 在 Accuracy、Brier、Log Loss 上更适合概率校准与专家复核场景。  
- 全局解释侧重 **项目质量**（创新性、风险、方法成熟度、跨学科、可行性等）。  
- **协同分流**（`outputs/case_routing.csv`）示例：`fast_track` / `regular_review` / `expert_review`。

详细论述见 [`docs/核心模型与算法报告.md`](docs/核心模型与算法报告.md)。

---

## 输出物说明

| 文件 | 含义 |
|------|------|
| `outputs/model_metrics.json` | 三模型分类与校准指标 |
| `outputs/global_importance.json` | EBM 全局特征重要性 |
| `outputs/group_importance.json` | 申请人/项目/团队/领域组贡献 |
| `outputs/response_curves.json` | 关键变量响应曲线 |
| `outputs/local_explanations.json` | 典型项目局部加减分证据 |
| `outputs/case_routing.csv` | 置信度、风险、偏差审计 → 复核强度 |
| `outputs/public_feature_mapping.csv` | OpenAlex 字段 → 仿真特征映射 |
| `outputs/nsfc_*` | NSFC 与 OpenAlex  enriched 映射（若已生成） |

---

## 常见问题

**为何不用真实基金评审数据训练？**  
公开库多仅有已资助项目，缺少未资助样本、专家评分与最终排序，无法构成可靠监督标签。本项目用半仿真数据训练，用 OpenAlex 等证明特征设计有现实依据。

**为何以 EBM 为主模型？**  
EBM 兼顾非线性与玻璃盒解释（全局项、局部贡献、形状函数），契合「信任校准」而非单纯追求黑盒精度。

**校验失败、提示缺少 `models/*.joblib`？**  
先运行 `python src/train_models.py`，再执行 `python src/validate_project.py`。

**Streamlit 无法加载新申请分析？**  
确认 `models/ebm_model.joblib` 存在；若不存在，按「快速开始」完成训练流程。

---

## 文档与引用

- [`docs/技术路线与复现实验说明.md`](docs/技术路线与复现实验说明.md) — 技术路线与复现步骤  
- [`docs/核心模型与算法报告.md`](docs/核心模型与算法报告.md) — 结题报告 / PPT 素材  
- [`docs/open_source_links.md`](docs/open_source_links.md) — InterpretML、LIME 等开源资源  

**主要依赖**：[InterpretML](https://github.com/interpretml/interpret)、scikit-learn、Streamlit、Plotly。

---

## 后续扩展方向

1. 接入更多公开资助库（NSF、NIH、CORDIS）并完善字段映射  
2. 公平性/偏差审计：对单位层次、海外经历、年龄等做系统分析  
3. 专家反馈闭环：在面板中记录采纳/修正/否决并沉淀为样本  
4. 与 `human_ai_trust_project` 的 Flask 静态页打通，或导出答辩用 PDF/HTML  

---

## 许可证

本项目采用 [MIT License](LICENSE)，课程作业与学术研究可自由复用；使用 OpenAlex 等第三方数据时请遵守其服务条款与引用规范。
