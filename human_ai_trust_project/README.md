# 人机协同科研信任机制项目

这是一个围绕“人机良好科研信任机制”主题构建的课程项目。根据你的新要求，项目已经调整为：**优先使用现有 GitHub 开源仓库中的研究图、公开数据集，以及自行整理的数据统计表**，而不是以自动生成图为主。

核心工作可以概括为：

1. 关注 AI 深度参与科研决策后，研究者如何形成“适当信任”。
2. 从可解释性、证据可追溯性、任务风险、研究者素养和组织规范等维度分析信任机制。
3. 采用 InterpretML 的 EBM 玻璃盒模型作为技术支撑。
4. 通过全局解释、局部解释、响应曲线与证据归档构建人机协同科研信任框架。
5. 将研究落脚于科研评审、科研资助和建议采纳等高责任场景。

## 当前项目结构

```text
human_ai_trust_project/
├── data/
│   ├── adult_uci_raw.data
│   └── adult_income_openml.csv
├── outputs/
│   ├── feature_importance.csv
│   ├── shape_functions.csv
│   └── model_summary.json
├── src/
│   ├── app.py
│   ├── train_model.py
│   ├── generate_dashboard.py
│   ├── templates/
│   │   └── index.html.j2
│   └── static/
│       ├── ebm_global.png
│       ├── ebm_local.png
│       ├── dashboard.png
│       └── css/style.css
├── web/
│   └── index.html
└── README.md
```

## 数据来源说明

当前项目中的数据主要包括：

### 1. 公开数据集
- `adult_uci_raw.data`
- `adult_income_openml.csv`

说明：该数据集来自公开成人收入分类任务数据，可作为“AI 辅助决策”建模演示数据。虽然它不直接等于你们的人机科研信任问卷数据，但适合用于展示可解释模型、特征解释和信任校准逻辑。

### 2. 研究图来源
以下图片已从相关公开仓库下载并纳入项目：
- `src/static/ebm_global.png`
- `src/static/ebm_local.png`
- `src/static/dashboard.png`

这些图来自 InterpretML 官方仓库，适合展示 EBM 的解释能力。

## 模型说明

项目仍保留 Python 模型训练脚本：

- `src/train_model.py`

作用：
- 在公开数据上训练 EBM 模型；
- 输出特征重要性、形状函数和模型摘要；
- 为网页中的统计表提供结果支持。

## 网页展示方式

网页不再以“自动生成交互图”为主体，而改为：

- 现有研究图展示
- 公开数据统计表展示
- 关键特征结果表展示
- 项目结构和来源说明展示

这种形式更贴近课程作业、研究汇报和答辩材料的呈现习惯。

## 如何运行

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
python src/train_model.py
```

### 3. 生成网页

```bash
python src/generate_dashboard.py
```

### 4. 启动网页

```bash
python src/app.py
```

浏览器访问：

```text
http://127.0.0.1:8050
```

## 相关开源仓库

1. **InterpretML**  
   https://github.com/interpretml/interpret

2. **GAM Changer**  
   https://github.com/interpretml/gam-changer

3. **DimVis**  
   https://github.com/parisa-salmanian/DimVis

4. **LIME**  
   https://github.com/marcotcr/lime

## 说明

当前版本更强调“基于公开数据和现有研究图完成课程项目展示”。如果你后续希望，我还可以继续把这个项目再改成：

- 更像正式论文附录展示页的版本；
- 更像课程答辩网站的版本；
- 或进一步替换为与你们研究主题更贴近的公开问卷/实验数据集版本。
