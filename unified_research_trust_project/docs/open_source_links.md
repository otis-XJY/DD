# 相关开源代码与资源链接

本项目统一版在 `project` 与 `human_ai_trust_project` 基础上整合实现，下列资源为理论与实现的重要参考。

## 1. InterpretML

- GitHub: https://github.com/interpretml/interpret
- 用途：Explainable Boosting Machine（EBM）训练、全局/局部解释、形状函数
- 本项目：`src/train_models.py`、`src/explain_and_export.py` 主模型与解释导出

## 2. GAM Changer

- GitHub: https://github.com/interpretml/gam-changer
- 用途：GAM/EBM 交互式编辑与人机协同修正
- 扩展方向：专家反馈反哺模型、信任闭环

## 3. DimVis

- GitHub: https://github.com/parisa-salmanian/DimVis
- 用途：高维投影与 EBM 可视化分析

## 4. LIME

- GitHub: https://github.com/marcotcr/lime
- 用途：黑盒模型局部解释，可与 EBM 对照

## 5. OpenAlex

- 文档: https://docs.openalex.org/
- 用途：`src/download_public_reference.py` 拉取 Works 元数据，支撑 `public_feature_mapping.csv`

## 6. Adult Income 公开数据（演示/对照）

- UCI: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
- OpenML: https://www.openml.org/search?type=data&sort=runs&id=1590&status=active
- 本地路径：`data/adult_income_openml.csv`（来自 `human_ai_trust_project` 数据策略）

## 7. 同仓库参考项目

| 目录 | 借鉴内容 |
|------|----------|
| `../project` | 六标签页解释面板、协同分流、`validate_project` |
| `../human_ai_trust_project` | 信任理论叙述、静态研究图展示思路 |
