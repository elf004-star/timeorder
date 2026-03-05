# TimeOrder

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 井眼轨迹状态识别机器学习系统 - 面向油气钻井场景

## 项目简介

本项目是一个用于 **井眼轨迹状态识别** 的机器学习系统。基于井斜角、狗腿度等测井数据，结合梯度提升类集成模型（GBM / LightGBM / XGBoost / CatBoost）和专家规则修正，实现对井眼轨迹四种状态的自动识别：

| 状态码 | 状态名称 | 英文名称 |
|:------:|:--------:|:--------:|
| 0 | 直井段 | Vertical |
| 1 | 造斜段 | Build-up |
| 2 | 稳斜段 | Hold |
| 3 | 降斜段 | Drop-off |

核心输出为每个深度点的 `status` / `status_predict`，以及根据预测结果自动标注的关键点（造斜开始、稳斜开始、降斜开始）。

## 环境要求

- **Python**: 3.10 及以上

## 快速安装

```bash
# 方式一：使用 requirements.txt（推荐）
pip install -r requirements.txt

# 方式二：以包形式安装
pip install .

# 可选：安装 ML 扩展依赖（超参数调优、Jupyter）
pip install .[ml]
```

## 数据文件说明

| 文件 | 描述 |
|------|------|
| `data/data.csv` | 原始按深度采样的定向数据（包含井斜、方位角等），用于计算狗腿度 |
| `data001.csv` | 建模数据示例，包含 `序号`、`转换后JH`、`JX`、`Dogleg Severity`、`status` 等字段 |

> 详细字段及特征工程说明请参考：`Methodology.md`、`feature_engineering_documentation.md`

## 主要脚本

### 数据预处理

| 脚本 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `calculate_dogleg.py` | 计算狗腿度 | `data/data.csv` | `data/data_with_dogleg.csv` |
| `add_status_column.py` | 生成状态标签 | `data/data_with_dogleg.csv` | `data_with_status.csv` |
| `explore_data.py` | 数据探索与可视化 | `data001.csv` | 统计图表 (PNG) |

### 模型训练 GUI

| 脚本 | 模型框架 | 输出目录 |
|------|----------|----------|
| `train_model_lightgbm_gui.py` | LightGBM | `memory_light/` |
| `train_model_xgboost_gui.py` | XGBoost | `memory_xgboost/` |
| `train_model_catboost_gui.py` | CatBoost | `memory_catboost/` |
| `train_model_gbm_gui.py` | Scikit-learn GBM | `memory_gbm/` |

**GUI 功能特性：**
- 选择训练数据与验证集比例
- 配置预处理方法（滚动平均 / Savitzky-Golay 滤波）
- 调整模型超参数
- 实时监控训练/验证 Loss 曲线
- 自动计算 Accuracy / Macro-F1 / Log Loss
- 保存模型、特征重要性、混淆矩阵

### 后处理

| 脚本 | 功能 |
|------|------|
| `add_keypoint_column.py` | 从 `status_predict` 序列自动识别井段关键点 |

## 使用流程

### 第一步：数据准备

```bash
# 1. 计算狗腿度
python calculate_dogleg.py

# 2. 生成状态标签（需有关键点标注）
python add_status_column.py
```

### 第二步：数据探索（可选）

```bash
python explore_data.py
```

### 第三步：模型训练

```bash
# 使用 LightGBM（推荐）
python train_model_lightgbm_gui.py

# 或使用其他模型
python train_model_xgboost_gui.py
python train_model_catboost_gui.py
python train_model_gbm_gui.py
```

### 第四步：生成关键点（可选）

```bash
python add_keypoint_column.py
```

关键点标注规则：

| 状态跳变 | 关键点值 | 含义 |
|----------|----------|------|
| 0 → 1 | 1 | 直井 → 造斜 |
| 1 → 2 | 2 | 造斜 → 稳斜 |
| 2 → 3 | 3 | 稳斜 → 降斜 |

## 项目结构

```
timeorder/
├── calculate_dogleg.py          # 狗腿度计算
├── add_status_column.py         # 状态标签生成
├── explore_data.py              # 数据探索与可视化
├── add_keypoint_column.py       # 关键点生成
├── train_model_*_gui.py         # 模型训练 GUI（4种模型）
├── data/                        # 数据目录
├── Models/                      # 模型存储目录
├── memory_*/                    # 训练过程中间结果
├── pyproject.toml               # 项目配置
└── requirements.txt             # 依赖列表
```

## 文档

- [Methodology.md](Methodology.md) - 方法论详细说明
- [feature_engineering_documentation.md](feature_engineering_documentation.md) - 特征工程文档
- [Experiments and Results Analysis.md](Experiments%20and%20Results%20Analysis.md) - 实验结果分析

## 许可证

本项目采用 [MIT License](https://opensource.org/licenses/MIT) 许可证。

## 作者

CCQ - 1873475824@qq.com
