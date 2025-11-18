# Predict.py 使用说明

## 概述

`predict.py` 是井段状态预测工具，支持两种使用模式：
1. **GUI 模式**（推荐）：通过文件选择对话框选择文件
2. **命令行模式**：通过命令行参数指定文件

## 使用方法

### 方法 1：GUI 模式（推荐，最简单）

直接运行程序，会自动弹出文件选择对话框：

```bash
python predict.py
```

**操作步骤：**
1. 运行程序后，会弹出文件选择对话框
2. 选择要预测的 CSV 文件
3. 然后会弹出保存对话框
4. 选择预测结果的保存位置和文件名
5. 程序自动进行预测并保存结果

**注意：** 默认输出文件名会基于输入文件名自动生成（例如：`data_predictions.csv`）

### 方法 2：命令行模式

通过命令行参数指定文件：

```bash
python predict.py --input data.csv --output results.csv
```

**命令行参数：**

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--input` | 输入CSV文件路径 | 无（会打开GUI） | `--input data.csv` |
| `--output` | 输出CSV文件路径 | 无（会打开GUI） | `--output results.csv` |
| `--model` | 模型文件路径 | `well_status_model.txt` | `--model my_model.txt` |
| `--visualize` | 是否生成可视化图表 | False | `--visualize` |
| `--well` | 指定可视化的井号 | 第一口井 | `--well B16-X1` |
| `--gui` | 强制使用GUI模式 | False | `--gui` |

### 方法 3：混合模式

可以指定部分参数，未指定的会使用GUI：

```bash
# 只指定输入文件，输出文件通过GUI选择
python predict.py --input data.csv

# 强制使用GUI选择所有文件
python predict.py --gui

# 预测并生成可视化
python predict.py --visualize
```

## 使用示例

### 示例 1：基本预测（GUI模式）

```bash
python predict.py
```

程序会：
1. 弹出对话框选择输入CSV
2. 弹出对话框选择输出位置
3. 自动进行预测
4. 保存结果

### 示例 2：命令行指定文件

```bash
python predict.py --input data001.csv --output predictions.csv
```

### 示例 3：预测并可视化

```bash
python predict.py --input data001.csv --output predictions.csv --visualize
```

生成的文件：
- `predictions.csv` - 预测结果
- `prediction_visualization_[井号].png` - 可视化图表

### 示例 4：指定特定井进行可视化

```bash
python predict.py --input data001.csv --visualize --well B16-X1
```

### 示例 5：使用自定义模型

```bash
python predict.py --input new_data.csv --model custom_model.txt
```

## 输入文件要求

CSV文件必须包含以下列：

| 列名 | 说明 | 必需 |
|------|------|------|
| `序号` | 数据点序号 | 是 |
| `转换后JH` | 井号 | 是 |
| `JX` | 井斜角 | 是 |
| `dy➗dx` | 井斜角变化率 | 是 |
| `status` | 实际状态（如果有） | 否 |

**示例数据格式：**
```csv
序号,转换后JH,JX,dy➗dx
1,B16-X1,0.0,
2,B16-X1,0.21,0.018229167
3,B16-X1,0.30,0.003208556
...
```

## 输出文件说明

预测结果CSV文件包含：
- 所有输入列
- 生成的60+个特征列
- `predicted_status` - 预测的井段状态（0-3）
  - 0: 直井段 (Vertical)
  - 1: 造斜段 (Build-up)
  - 2: 稳斜段 (Hold)
  - 3: 降斜段 (Drop-off)

## 可视化输出

如果使用 `--visualize` 参数，会为每口井生成可视化图表：

**图表包含3个子图：**
1. 井斜角曲线
2. dy/dx 曲线
3. 状态对比（如果有真实标签）

**图表特点：**
- 字体：Arial
- 语言：English
- 高分辨率：300 DPI
- 格式：PNG

## 错误处理

### 常见错误及解决方案

**1. 找不到模型文件**
```
Error: Model file not found
```
**解决：** 确保 `well_status_model.txt` 存在于当前目录，或使用 `--model` 指定路径

**2. 输入文件格式错误**
```
Error: Required column not found
```
**解决：** 检查CSV文件是否包含必需的列（序号、转换后JH、JX、dy➗dx）

**3. 取消文件选择**
```
No file selected. Exiting...
```
**解决：** 正常退出，重新运行程序并选择文件

## 前置条件

在使用预测功能前，必须：
1. 已运行 `train_model.py` 训练模型
2. 存在 `well_status_model.txt` 模型文件
3. 已安装所有依赖包（参见 `requirements.txt`）

## 性能说明

- **预测速度**：约 1000-2000 条数据/秒
- **内存占用**：通常 < 1GB
- **模型加载时间**：< 1秒

## 注意事项

1. ⚠️ **模型完整性**：当前版本的模型保存/加载机制需要改进。建议在同一个Python会话中训练和预测。

2. ⚠️ **数据一致性**：预测数据的格式和特征必须与训练数据一致。

3. ⚠️ **文件编码**：输出文件使用 UTF-8 with BOM 编码，确保中文显示正确。

4. ⚠️ **GUI依赖**：GUI模式需要tkinter支持（Python标准库，通常已安装）。

## 高级用法

### 批量预测多个文件

创建批处理脚本：

```bash
# Windows (batch file)
@echo off
for %%f in (*.csv) do (
    python predict.py --input %%f --output %%~nf_predictions.csv
)
```

```bash
# Linux/Mac (shell script)
for file in *.csv; do
    python predict.py --input "$file" --output "${file%.csv}_predictions.csv"
done
```

### Python 脚本中调用

```python
from predict import predict_new_data, visualize_predictions

# 预测
df_result = predict_new_data('data.csv', 'output.csv')

# 可视化
visualize_predictions(df_result, well_name='B16-X1')
```

## 技术支持

如有问题，请查看：
- `README_ML.md` - 完整文档
- `ml_solution.md` - 方案设计
- `CHART_LANGUAGE_SUMMARY.md` - 图表说明

---

**版本**: 1.0
**更新日期**: 2024-10

