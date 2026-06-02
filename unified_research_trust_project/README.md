# 人机协同科研决策信任机制：综合模型、算法与信任分析平台（合并项目）

> 本项目由两个原始项目合并而成：
> 1. 科研决策信任机制模型与算法（核心模型、数据生成、多模型训练、解释导出、协同分流、OpenAlex映射）
> 2. 人机协同科研信任机制项目（信任维度特征、EBM玻璃盒分析、证据可追溯性、任务风险、研究者素养、组织规范等综合分析）
>
> **合并原则**：保留所有可能的方式方法、数据集、模型、解释维度和分析能力，不阉割任何功能。网页/可视化前端部分暂不考虑，重点保留后端分析能力。

本项目是“智能决策理论与方法”课程小组作业中“核心模型和算法”部分的可运行原型。项目目标不是构建一个替代专家的自动审批系统，而是构建一个面向科研项目评审场景的可解释 AI 决策支持系统，用于展示 AI 如何给出资助建议、为什么给出该建议，以及专家如何基于解释证据进行信任校准。

## 一、项目核心思路

系统采用“AI 参谋，专家决策”的协同定位：

1. AI 根据科研项目申请特征输出资助概率和建议结果。
2. 可解释模型输出全局重要性、局部加减分证据和关键变量响应曲线。
3. 信任校准模块结合预测置信度、项目风险和偏差审计信号，将项目分流为快速复核、常规人工复核和专家重点审查。
4. 专家基于可视化证据进行采纳、修正、复核或否决。

项目采用“半仿真数据为主，公开真实元数据为辅”的数据策略。半仿真数据用于监督学习和解释机制展示；OpenAlex 真实科研元数据用于证明仿真特征设计具有现实依据。

## 二、项目结构

```text
project/
  README.md
  requirements.txt
  environment.yml

  app/
    streamlit_app.py

  data/
    applications.csv
    openalex_reference_raw.json
    openalex_reference_sample.csv

  docs/
    技术路线与复现实验说明.md
    核心模型与算法报告.md
    figures/
      可视化网站1.jpg
      可视化网站2.jpg
      可视化网站3.jpg

  models/
    ebm_model.joblib
    logistic_regression_baseline.joblib
    random_forest_baseline.joblib

  outputs/
    model_metrics.json
    public_data_summary.json
    public_feature_mapping.csv
    global_importance.json
    group_importance.json
    response_curves.json
    local_explanations.json
    case_routing.csv
    explanation_summary.json
    test_set.csv

  src/
    config.py
    generate_data.py
    download_public_reference.py
    train_models.py
    explain_and_export.py
    validate_project.py
    run_pipeline.py
```

## 三、主要文件说明

### 根目录

- `README.md`：项目交接说明，包含项目结构、环境配置、运行流程和输出解释。
- `requirements.txt`：Python 依赖列表。
- `environment.yml`：可选的 conda 环境配置文件。

### `src/`

- `config.py`：统一配置文件，定义目录、随机种子、目标变量、特征列表、特征组和中文特征名。
- `generate_data.py`：生成半仿真科研项目申请数据。数据标签由显式规则、非线性交互和随机噪声共同生成。
- `download_public_reference.py`：从 OpenAlex Works API 下载少量真实科研元数据，并生成真实字段到仿真特征的映射表。
- `train_models.py`：训练三类模型：Logistic Regression、Explainable Boosting Machine 和 Random Forest。
- `explain_and_export.py`：基于 EBM 导出全局解释、组别解释、响应曲线、局部解释和协同分流结果。
- `validate_project.py`：项目完整性校验脚本，检查关键文件、模型指标、分流类别和公开数据映射是否齐全。
- `run_pipeline.py`：简化入口脚本，仅用于快速生成仿真数据、训练和解释导出。完整复现建议按“六、复现流程”逐步运行。

### `app/`

- `streamlit_app.py`：可视化面板，展示模型对比、全局解释、个案证据、响应曲线、协同分流和公开数据依据。

### `data/`

- `applications.csv`：半仿真科研项目申请数据，包含项目特征、真实资助概率和二分类标签。
- `openalex_reference_raw.json`：OpenAlex API 返回的原始真实科研元数据。
- `openalex_reference_sample.csv`：清洗后的 OpenAlex 样例字段，便于展示真实数据依据。

### `models/`

- `ebm_model.joblib`：主模型，Explainable Boosting Machine。
- `logistic_regression_baseline.joblib`：透明线性基线模型。
- `random_forest_baseline.joblib`：黑盒对照模型。

### `outputs/`

- `model_metrics.json`：三类模型的 Accuracy、AUC、Brier Score 和 Log Loss。
- `public_data_summary.json`：OpenAlex 公开数据下载摘要。
- `public_feature_mapping.csv`：公开真实字段到仿真特征的映射关系。
- `global_importance.json`：EBM 全局特征重要性。
- `group_importance.json`：申请人基础、项目质量、团队条件等特征组贡献度。
- `response_curves.json`：关键变量的响应曲线。
- `local_explanations.json`：典型项目的局部加分和扣分证据。
- `case_routing.csv`：每个项目的预测概率、置信度、风险信号和协同分流结果。
- `explanation_summary.json`：解释结果摘要。
- `test_set.csv`：测试集样本。

### `docs/`

- `技术路线与复现实验说明.md`：较简洁的技术路线和复现实验说明。
- `核心模型与算法报告.md`：完整报告，可直接供结题材料、报告撰写和 PPT 讲稿参考。
- `figures/`：可视化面板截图。

## 四、环境准备

推荐在项目根目录创建独立 conda 环境，避免污染已有 Python 环境。

PowerShell：

```powershell
conda create -p .\.conda python=3.11 -y
conda activate .\.conda
$env:PYTHONNOUSERSITE="1"
pip install -r requirements.txt
```

也可以使用 `environment.yml`：

```powershell
conda env create -f environment.yml
conda activate trust_xai
$env:PYTHONNOUSERSITE="1"
```

说明：如果本机启用了用户级 Python 包，建议运行前设置 `$env:PYTHONNOUSERSITE="1"`，避免读取其他环境中的旧包。

## 五、完整复现流程

在项目根目录依次运行：

```powershell
conda activate .\.conda
$env:PYTHONNOUSERSITE="1"

python .\src\generate_data.py --n-samples 1500
python .\src\download_public_reference.py --per-page 80
python .\src\train_models.py
python .\src\explain_and_export.py
python .\src\validate_project.py
```

如果只想快速重新生成仿真模型和解释结果，可以运行：

```powershell
python .\src\generate_data.py --n-samples 1500
python .\src\train_models.py
python .\src\explain_and_export.py
```

## 六、启动可视化面板

```powershell
streamlit run .\app\streamlit_app.py
```

启动后，终端会显示本地访问地址。面板包含以下页面：

- `模型对比`：Logistic Regression、EBM 和 Random Forest 的指标表与柱状图。
- `全局解释`：全局特征重要性和特征组贡献度。
- `个案证据`：单个项目的主要加分证据和扣分证据。
- `响应曲线`：关键变量变化对预测资助概率的影响。
- `协同分流`：快速复核、常规人工复核和专家重点审查的分流结果。
- `公开数据依据`：OpenAlex 真实字段样例和仿真特征映射。

## 七、当前实验结果

当前默认配置下，半仿真数据为 1500 条，资助率约为 25.53%。

三类模型结果如下：

| 模型 | Accuracy | AUC | Brier Score | Log Loss |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7067 | 0.7997 | 0.1810 | 0.5346 |
| Explainable Boosting Machine | 0.7813 | 0.7610 | 0.1554 | 0.4755 |
| Random Forest | 0.7653 | 0.7820 | 0.1593 | 0.4884 |

解释口径：

- Logistic Regression 是透明线性基线，AUC 较高，但默认阈值下的 Accuracy 和概率校准不如 EBM。
- Random Forest 是黑盒对照，预测能力较强，但无法直接提供面向专家复核的透明证据链。
- EBM 是主模型，Accuracy、Brier Score 和 Log Loss 表现较好，并能输出全局解释、局部解释和响应曲线，更适合支撑信任校准。

EBM 全局解释结果显示，重要因素主要集中在创新性评分、项目风险、方法成熟度、跨学科程度和可行性评分。组别贡献中，项目质量约占 53.5%，团队条件约占 19.4%，申请人基础约占 16.5%。这说明模型主要依据项目本身质量，而不是过度依赖背景变量。

协同分流结果如下：

| 分流类别 | 数量 | 含义 |
|---|---:|---|
| fast_track | 403 | 高置信、低风险、偏差审计信号较低，适合快速复核 |
| regular_review | 612 | 常规人工复核 |
| expert_review | 485 | 低置信、高风险或存在偏差审计信号，建议专家重点审查 |

## 八、公开数据依据

`download_public_reference.py` 通过 OpenAlex Works API 下载真实科研元数据。该数据不作为主训练集，因为它缺少未获资助申请、专家评分和最终评审决策；它用于证明仿真特征池有现实数据结构依据。

映射示例：

```text
OpenAlex cited_by_count / fwci -> citations_5y / expected_impact
OpenAlex authorships / institution_count / country_count -> team_size / collaboration_strength
OpenAlex primary_topic / topic_field / topic_domain -> discipline / frontier_score / interdisciplinarity
OpenAlex referenced_works_count / related_works_count -> methodology_score / feasibility_score
OpenAlex awards / funders -> funded / prior_grants 的现实参照
```

## 九、常见问题

### 1. 为什么不用真实基金评审数据训练？

公开数据库通常只有已资助项目，缺少未资助项目、专家评分、评审意见和最终排序。直接使用公开数据训练二分类模型会缺少负样本和真实标签，因此本项目采用半仿真数据训练，用公开真实元数据支撑特征设计。

### 2. 为什么不用大模型 API 生成数据？

本项目需要可复现和可解释的数据生成机制。规则生成可以明确说明标签由哪些因素产生，并能控制边界样本、高风险样本和偏差审计信号。大模型生成文本会引入不可控噪声，不利于复现和课堂答辩。

### 3. 为什么选择 EBM 作为主模型？

EBM 能表达非线性关系，同时保留可解释结构。它可以输出每个特征的贡献函数、局部加减分证据和响应曲线，适合解释“AI 为什么这样建议”。这比单纯追求黑盒模型精度更符合“信任机制研究”的课程主题。

### 4. 如何确认项目完整？

运行：

```powershell
python .\src\validate_project.py
```

如果输出 `Project validation passed.`，说明关键文件、模型结果、解释结果、公开数据映射和分流结果均已生成。

## 十、后续同学可继续扩展的方向

1. 扩展真实数据参考：进一步下载 NSF、NIH 或 CORDIS 的公开项目数据，补充字段映射。
2. 增加公平性分析：对单位层次、海外经历、年龄阶段等变量进行更系统的偏差审计。
3. 增加交互解释：重点展示创新性与可行性、前沿度与跨学科程度等交互项。
4. 增加人工反馈闭环：让专家在面板中标注采纳、修正或拒绝 AI 建议，并保存为后续训练样本。
5. 制作结题 PPT：可直接使用 `docs/核心模型与算法报告.md` 中的结构和结论。

## 十一、合并后的综合分析能力（不阉割）

本统一项目保留并整合了以下**所有分析方式与维度**，确保对科研项目的信任机制分析全面、深入：

### 1. 数据层面（保留所有数据集与生成方式）
- 半仿真科研项目申请数据（generate_data.py）：规则+非线性交互+噪声生成，支持1500+样本，含创新性、风险、团队、方法成熟度等特征
- 信任机制专用特征数据集（research_trust_data.csv）：显式包含 evidence_traceability（证据可追溯性）、explainability（可解释性）、task_risk（任务风险）、ai_literacy（研究者素养）、org_norm_clarity（组织规范清晰度）、trust_score 等维度
- 公开元数据参考（OpenAlex）：真实 cited_by_count、authorships、topic 等字段到仿真特征的映射表
- 成人收入公开数据集（adult_income_openml.csv）：用于通用 AI 辅助决策信任演示

### 2. 模型层面（保留所有模型与基线）
- Logistic Regression：透明线性基线
- Explainable Boosting Machine (EBM)：主模型，支持玻璃盒全局/局部解释、响应曲线
- Random Forest：黑盒对照模型
- 额外 EBM 信任专用训练（train_model_ebm_trust.py）：针对信任特征的独立训练与案例解释

### 3. 解释与分析维度（保留所有解释输出）
- 全局特征重要性 + 特征组贡献度（项目质量、团队条件、申请人基础等）
- 局部加减分证据（每个项目的具体支持/反对理由）
- 响应曲线（关键变量如创新性、风险对资助概率的影响）
- 协同分流路由（fast_track / regular_review / expert_review）：基于预测置信度、项目风险、偏差审计信号的多维度分流
- 案例高/低置信分析（含 trust_score、evidence_traceability 等）
- 模型指标对比（Accuracy, AUC, Brier Score, Log Loss）
- 公开数据映射与真实依据验证
- 数据集概览（资助率、平均信任分数、可解释性、任务风险等）

### 4. 信任机制综合分析框架
- **可解释性**：EBM 形状函数、局部证据
- **证据可追溯性**：局部解释 + OpenAlex 字段映射
- **任务风险**：risk 特征 + 分流中的高风险标记
- **研究者素养**：ai_literacy 特征
- **组织规范**：org_norm_clarity 特征
- **高责任场景**：科研评审、资助决策、建议采纳

所有脚本、模型文件、输出 JSON/CSV 均保留，可通过 run_pipeline.py 或单独脚本复现完整分析链路。
