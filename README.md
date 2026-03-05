+## 项目简介
+
+本项目 **timeorder** 是一个用于 **井眼轨迹状态识别** 的机器学习系统，面向油气钻井场景。  
+它基于井斜角、狗腿度等测井数据，结合梯度提升类集成模型（GBM / LightGBM / XGBoost / CatBoost）和专家规则修正，实现对井眼轨迹四种状态的自动识别：
+
+- **0：直井段 (Vertical)**
+- **1：造斜段 (Build-up)**
+- **2：稳斜段 (Hold)**
+- **3：降斜段 (Drop-off)**
+
+核心输出为每个深度点的 `status` / `status_predict`，以及根据预测结果自动标注的关键点（造斜开始、稳斜开始、降斜开始）。
+
+---
+
+## 环境与安装
+
+- **Python**：推荐 Python 3.10 及以上（`pyproject.toml` 要求 `>=3.10`）
+- **依赖安装方式一（推荐，直接按项目要求）**：
+
+```bash
+pip install -r requirements.txt
+```
+
+- **依赖安装方式二（以包形式安装）**：
+
+```bash
+pip install .
+```
+
+`pyproject.toml` 中还定义了一些可选依赖（如 `optuna`、Jupyter 等），若需要做进一步实验/调参，可通过：
+
+```bash
+pip install .[ml]
+```
+
+---
+
+## 数据文件概览（仓库中已有示例）
+
+- `data/data.csv`：原始按深度采样的定向数据（包含井斜、方位角等），用于计算狗腿度。
+- `data001.csv`：已经汇总好的建模数据示例，通常包含：
+  - `序号`、`转换后JH`（井号）、`JX`（井斜角）、`Dogleg Severity`、`status` 等。
+- `validation_without_label_predictions.csv`：仅含预测 `status_predict` 的验证集结果（无真实标签），用于后处理生成关键点。
+
+具体字段及特征工程细节可参考：
+
+- `Methodology.md`
+- `feature_engineering_documentation.md`
+- `Experiments and Results Analysis.md`
+
+---
+
+## 主要脚本说明
+
+- **数据预处理与特征相关**
+  - `calculate_dogleg.py`  
+    - 从 `data/data.csv` 读取原始井斜 / 方位数据，按井号和深度排序，计算相邻测点的 **狗腿度 (Dogleg Severity)**。  
+    - 输出：`data/data_with_dogleg.csv`。
+  - `add_status_column.py`  
+    - 针对已经有人为或算法标注好 `关键点` 的数据，按 **井号 + 关键点序号** 规则生成离散状态 `status`（0/1/2/3）。  
+    - 默认输入：`data/data_with_dogleg.csv`，输出：`data_with_status.csv`。
+  - `explore_data.py`  
+    - 对数据进行 **EDA 和可视化分析**，包括：
+      - 缺失值、统计描述
+      - Status 分布、按井分布
+      - 井斜与狗腿度的分布和随深度变化曲线
+      - 特征相关性热力图  
+    - 默认从根目录读取 `data001.csv`。
+
+- **模型训练 GUI（交互式训练与验证）**
+  - `train_model_lightgbm_gui.py`：使用 **LightGBM** 进行训练与预测 GUI。
+  - `train_model_xgboost_gui.py`：使用 **XGBoost**。
+  - `train_model_catboost_gui.py`：使用 **CatBoost**。
+  - `train_model_gbm_gui.py`：使用 **Scikit-learn GradientBoosting (GBM)**。
+
+  这些脚本共性：
+  - 打开一个 Tkinter 图形界面，允许：
+    - 选择训练数据（默认 `data001.csv`，需包含 `Dogleg Severity` 与 `status`）
+    - 选择验证集比例
+    - 配置预处理（滚动平均 / Savitzky-Golay 滤波）和模型超参数
+    - 监控训练/验证 Loss 曲线
+    - 自动计算 Accuracy / Macro-F1 / Log Loss
+    - 保存模型、特征重要性、混淆矩阵与分井结果（`memory_*` 目录）
+
+- **预测后处理与关键点生成**
+  - `add_keypoint_column.py`  
+    - 针对一个已经包含预测列 `status_predict` 的数据文件（例如模型在无标签验证集上的预测结果）：
+      - 按井号，检测 `status_predict` 从 0→1、1→2、2→3 的跳变位置
+      - 在这些位置标注 `关键点` 分别为 1 / 2 / 3
+    - 默认输入：`validation_without_label_predictions.csv`  
+      输出：`validation_with_keypoints.csv`。
+
+---
+
+## 推荐执行流程（程序执行顺序）
+
+下面给出从 **原始方向数据** 到 **模型训练、预测与关键点提取** 的一条典型工作流。  
+根据你已有的数据情况，可以从中间任一步开始。
+
+### 一、从原始测井数据开始（如有）
+
+1. **计算狗腿度**
+
+   ```bash
+   python calculate_dogleg.py
+   ```
+
+   - 读入：`data/data.csv`
+   - 写出：`data/data_with_dogleg.csv`
+
+2. **基于关键点生成状态标签（如已有人为标注关键点）**
+
+   编辑 `add_status_column.py` 中的路径或直接使用默认路径，然后执行：
+
+   ```bash
+   python add_status_column.py
+   ```
+
+   - 读入：`data/data_with_dogleg.csv`（需要包含列 `转换后JH`、`关键点` 等）
+   - 写出：`data_with_status.csv`（新增 `status` 列）
+
+3. **整理为建模数据 `data001.csv`（一次性操作）**
+
+   - 将包含 `序号`、`转换后JH`、`JX`、`Dogleg Severity`、`status` 等字段的样本整合为一个 CSV，命名为 `data001.csv`，放在项目根目录。
+   - 仓库中已经提供了一个示例 `data001.csv`，可直接用于尝试和调试。
+
+### 二、探索数据与理解特征（可选但推荐）
+
+4. **运行数据探索脚本**
+
+   ```bash
+   python explore_data.py
+   ```
+
+   - 默认读入：`data001.csv`
+   - 输出若干统计信息和图片文件：
+     - `status_distribution.png`
+     - `feature_distributions.png`
+     - `well_trajectories.png`
+     - `correlation_heatmap.png`
+
+### 三、训练井段状态预测模型（任选一种或多种模型）
+
+5. **使用 LightGBM GUI 训练（示例）**
+
+   ```bash
+   python train_model_lightgbm_gui.py
+   ```
+
+   在弹出的界面中：
+
+   - 选择训练数据文件（默认 `data001.csv`）
+   - 设置验证集比例（例如 0.2）
+   - 设置平滑方法（`none` / `rolling` / `savgol`）
+   - 配置超参数（学习率、轮数、叶子数等）
+   - 点击“开始训练”，观察训练/验证曲线及评估指标  
+   - 训练完成后，GUI 会：
+     - 将模型保存到 `Models/`（例如 `well_status_model.txt` 及对应的 `_scaler.pkl` 和 `_params.json`）
+     - 将训练集/验证集的预测结果与混淆矩阵等保存到 `memory_light/` 目录
+
+6. **使用 XGBoost / CatBoost / GBM 版本训练**
+
+   步骤与 LightGBM 类似，只是调用的脚本不同：
+
+   ```bash
+   # XGBoost 版本
+   python train_model_xgboost_gui.py
+
+   # CatBoost 版本
+   python train_model_catboost_gui.py
+
+   # GBM 版本
+   python train_model_gbm_gui.py
+   ```
+
+   不同模型会分别在 `memory_xgboost/`、`memory_catboost/`、`memory_gbm/` 下生成对应的中间结果。
+
+### 四、对新数据进行状态预测
+
+7. **在 GUI 中进行预测**
+
+   以 LightGBM 为例（其余脚本流程基本一致）：
+
+   - 运行：
+
+     ```bash
+     python train_model_lightgbm_gui.py
+     ```
+
+   - 在 GUI 中点击“模型预测”按钮：
+     1. 选择已经训练好的模型文件（位于 `Models/`）
+     2. 选择待预测的 CSV 文件（需包含 `序号`、`转换后JH`、`JX`、`Dogleg Severity` 等列）
+     3. 选择输出文件保存路径，例如：`some_data_predictions.csv`
+
+   - 程序会：
+     - 输出整体预测结果（含 `origin_status_predict`、`status_predict`）
+     - 在 `predict_light/`（或 `predict_xgboost` / `predict_catboost` / `predict_gbm`）下按井号保存分井 CSV
+     - 若数据中有真实 `status` 列，还会输出混淆矩阵 CSV / PNG
+
+### 五、对预测结果自动生成关键点（可选步骤）
+
+8. **从预测的 `status_predict` 中提取关键点**
+
+   当你拿到一个仅有 `status_predict` 的结果（例如验证集无真实标签，只运行了模型预测），可以使用：
+
+   ```bash
+   python add_keypoint_column.py
+   ```
+
+   - 默认：
+     - 读入：`validation_without_label_predictions.csv`
+     - 写出：`validation_with_keypoints.csv`（新增 `关键点` 列）
+   - 规则：
+     - 同一口井内，`status_predict` 从 0→1 时记为 `关键点 = 1`（直井 -> 造斜）
+     - 从 1→2 时记为 `关键点 = 2`（造斜 -> 稳斜）
+     - 从 2→3 时记为 `关键点 = 3`（稳斜 -> 降斜）
+
+---
+
+## 项目结构摘要
+
+- `calculate_dogleg.py`：按井计算狗腿度，输出 `data/data_with_dogleg.csv`。
+- `add_status_column.py`：基于 `关键点` 列生成离散状态 `status`。
+- `explore_data.py`：数据探索与可视化。
+- `train_model_lightgbm_gui.py` / `train_model_xgboost_gui.py` / `train_model_catboost_gui.py` / `train_model_gbm_gui.py`：四种集成模型的 GUI 训练与预测脚本。
+- `add_keypoint_column.py`：从 `status_predict` 序列自动识别井段关键点。
+- `Methodology.md`、`feature_engineering_documentation.md`、`Experiments and Results Analysis.md`：论文/报告风格的详细方法论、特征工程和实验结果说明文档。
+
+如需扩展或集成到其他系统，可以直接复用上述预处理、特征工程与模型脚本，并参考 `pyproject.toml` 中的依赖与入口定义。
+
*** End Patch