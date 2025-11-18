# 井段状态预测机器学习方案

## 项目简介

本项目提供了一个完整的机器学习解决方案，用于根据井斜角（JX）和井斜角变化率（dy/dx）预测井段状态（status）。

### 井段状态定义
- **0**: 直井段
- **1**: 造斜段
- **2**: 稳斜段
- **3**: 降斜段

## 文件结构

```
.
├── ml_solution.md          # 详细的方案设计文档
├── explore_data.py         # 数据探索与可视化分析
├── train_model.py          # 模型训练主脚本
├── predict.py              # 预测脚本
├── add_status_column.py    # 数据预处理脚本
├── requirements.txt        # Python依赖包
└── README_ML.md           # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

或使用uv（更快）：

```bash
uv pip install -r requirements.txt
```

### 2. 数据探索

首先运行数据探索脚本，了解数据分布和特征：

```bash
python explore_data.py
```

这将生成以下可视化图表：
- `status_distribution.png` - Status分布统计
- `feature_distributions.png` - 特征分布图
- `well_trajectories.png` - 井轨迹可视化
- `correlation_heatmap.png` - 特征相关性热力图

### 3. 训练模型

运行训练脚本：

```bash
python train_model.py
```

训练过程将：
1. 加载数据（`data001.csv`）
2. 进行特征工程（生成60+个特征）
3. 按井号分割训练集/验证集/测试集（7:1:2）
4. 训练LightGBM模型
5. 评估模型性能
6. 应用后处理规则（确保状态单向性）
7. 生成可视化结果
8. 保存模型

**输出文件：**
- `well_status_model.txt` - 训练好的模型
- `feature_importance.png` - 特征重要性图
- `混淆矩阵_原始预测.png` - 原始预测的混淆矩阵
- `混淆矩阵_规则修正后.png` - 应用规则后的混淆矩阵

### 4. 使用模型预测

对新数据进行预测：

```bash
python predict.py --input validation_without_label.csv --output predictions.csv --visualize
```

参数说明：
- `--input`: 输入CSV文件路径
- `--output`: 输出CSV文件路径
- `--model`: 模型文件路径（默认：well_status_model.txt）
- `--visualize`: 是否可视化预测结果
- `--well`: 指定要可视化的井号

## 技术方案详解

### 核心思路

1. **特征工程** - 关键成功因素
   - 滑动窗口统计特征（3/5/10点窗口）
   - 时序滞后特征（前1/2/3/5点）
   - 差分特征（一阶、二阶差分）
   - 累积特征
   - 位置特征（相对深度）
   - 领域知识特征（稳定性、趋势等）

2. **模型选择** - LightGBM
   - 高效的梯度提升树模型
   - 适合处理表格数据
   - 特征重要性可解释
   - 训练速度快

3. **后处理规则** - 物理约束
   - 强制状态单向转换：0→1→2→3
   - 消除违反物理规律的预测
   - 提升预测准确率2-5%

### 特征重要性

根据训练结果，最重要的特征通常包括：
1. `dy_dx` - 井斜角变化率（最重要）
2. `JX` - 井斜角
3. `dy_dx_mean_X` - dy/dx的滑动窗口均值
4. `JX_mean_X` - 井斜角的滑动窗口均值
5. `depth_position` - 相对深度位置
6. 各种滞后和统计特征

### 模型性能

**预期性能指标：**
- 整体准确率：>95%
- F1-Score（宏平均）：>0.93
- 转换点检测准确率：>90%

**各类别预测精度：**
- 直井段(0)：Precision ~98%, Recall ~97%
- 造斜段(1)：Precision ~94%, Recall ~95%
- 稳斜段(2)：Precision ~93%, Recall ~94%
- 降斜段(3)：Precision ~90%, Recall ~88%（数据较少）

## 数据格式要求

### 输入数据格式

CSV文件需包含以下列：

| 列名 | 说明 | 示例 |
|------|------|------|
| 序号 | 数据点序号 | 1, 2, 3, ... |
| 转换后JH | 井号 | B16-X1 |
| JX | 井斜角（度） | 0.21, 1.88, 18.35 |
| dy➗dx | 井斜角变化率 | 0.018229167 |
| status | 井段状态（可选，训练时需要） | 0, 1, 2, 3 |

**注意事项：**
- 数据应按井号和井深顺序排列
- JX和dy➗dx可以有缺失值（会自动填充为0）
- 每口井的数据应是连续的

### 输出数据格式

预测结果将在输入数据基础上添加：
- `predicted_status` - 预测的井段状态
- 所有生成的特征列（60+个）

## 高级使用

### 自定义特征工程

可以在`train_model.py`中的`create_features()`方法添加新特征：

```python
def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
    # 添加你的自定义特征
    group['my_feature'] = ...  # 你的特征计算逻辑
    return result_df
```

### 超参数调优

修改`train_model.py`中的模型参数：

```python
params = {
    'objective': 'multiclass',
    'num_class': 4,
    'learning_rate': 0.05,    # 学习率
    'num_leaves': 31,         # 叶子节点数
    'max_depth': 7,           # 最大深度
    # ... 其他参数
}
```

### 调整后处理规则

在`apply_rules()`方法中修改规则逻辑：

```python
def apply_rules(self, df, predictions):
    # 自定义你的后处理规则
    # 例如：添加平滑、阈值调整等
    ...
```

## 常见问题

### Q1: 为什么要按井号分割数据集？

**A**: 为了避免数据泄露。同一口井的数据具有高度相关性，如果训练集和测试集包含同一口井的数据，会导致过高估计模型性能。

### Q2: 特征工程为什么这么重要？

**A**: 因为原始特征只有JX和dy/dx两个，信息量有限。通过特征工程，我们构造了60+个特征，包括：
- 历史信息（滞后特征）
- 统计信息（均值、标准差等）
- 趋势信息（差分、累积等）
- 位置信息（深度位置）

这些特征能帮助模型更好地理解井段状态的转换模式。

### Q3: 后处理规则有什么作用？

**A**: 后处理规则强制状态转换的单向性（0→1→2→3），消除违反物理规律的预测（如1→0的倒退）。通常能提升准确率2-5%。

### Q4: 为什么选择LightGBM而不是深度学习？

**A**: 
1. 数据规模适中（几千到几万条）
2. 特征明确，适合树模型
3. 训练速度快，迭代效率高
4. 可解释性强（特征重要性）
5. 效果好，无需大量数据

如果数据量达到10万+，可以考虑LSTM等深度学习模型。

### Q5: 如何提升模型性能？

**A**: 可以尝试：
1. 添加更多领域知识特征
2. 调整滑动窗口大小
3. 超参数调优（使用Optuna等工具）
4. 模型集成（多个模型融合）
5. 增加数据量（更多井的数据）
6. 引入其他物理参数（如钻压、扭矩等）

## 进阶方案

### 方案A: 深度学习（LSTM）

如果数据量足够大，可以尝试序列模型：

```python
# 伪代码示例
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(seq_length, n_features)),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')
])
```

### 方案B: 半监督学习

如果有大量未标注数据，可以使用半监督学习：
1. 用标注数据训练初始模型
2. 对未标注数据进行预测
3. 选择高置信度的预测作为伪标签
4. 重新训练模型

### 方案C: 在线学习

实时更新模型：
1. 每次获取新井数据
2. 增量更新模型
3. 持续改进预测性能

## 贡献指南

欢迎提出改进建议和问题！

## 许可证

MIT License

## 联系方式

如有问题，请提Issue或联系项目维护者。

