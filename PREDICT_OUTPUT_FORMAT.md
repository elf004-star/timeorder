# 预测输出格式说明

## 输出文件格式

### 简洁模式（当前默认）

预测结果CSV文件**仅包含**：
- ✅ **所有原始输入列**（保持原样）
- ✅ **一个新列：`status_predict`**（预测的井段状态）

**不包含**：
- ❌ 60+个特征工程列
- ❌ 中间计算列
- ❌ 其他冗余数据

### 输出示例

**输入文件（data.csv）：**
```csv
序号,转换后JH,JX,dy➗dx
1,B16-X1,0.0,
2,B16-X1,0.21,0.018229167
3,B16-X1,0.30,0.003208556
4,B16-X1,0.48,0.006526468
...
```

**输出文件（data_predictions.csv）：**
```csv
序号,转换后JH,JX,dy➗dx,status_predict
1,B16-X1,0.0,,0
2,B16-X1,0.21,0.018229167,0
3,B16-X1,0.30,0.003208556,0
4,B16-X1,0.48,0.006526468,1
...
```

### status_predict 列说明

`status_predict` 列的值表示预测的井段状态：

| 值 | 状态名称（中文） | 状态名称（英文） | 描述 |
|----|----------------|-----------------|------|
| 0  | 直井段          | Vertical        | 井眼接近垂直 |
| 1  | 造斜段          | Build-up        | 井斜角逐渐增大 |
| 2  | 稳斜段          | Hold            | 井斜角保持稳定 |
| 3  | 降斜段          | Drop-off        | 井斜角逐渐减小 |

## 文件保存位置

### 默认保存位置

输出文件**自动保存在输入文件的同一目录**下。

**示例：**
```
输入文件： C:\Data\welldata.csv
输出文件： C:\Data\welldata_predictions.csv
```

### 文件命名规则

输出文件名 = `[输入文件名]_predictions.csv`

**示例：**
- 输入：`data001.csv` → 输出：`data001_predictions.csv`
- 输入：`my_well_data.csv` → 输出：`my_well_data_predictions.csv`
- 输入：`test.csv` → 输出：`test_predictions.csv`

### 自定义保存位置

如果在保存对话框中选择了其他位置，将保存到指定位置。

## 使用流程

### 1. 启动程序
```bash
python predict.py
```

### 2. 选择输入文件
- 文件选择对话框打开
- 选择要预测的CSV文件

### 3. 确认输出位置
- 保存对话框打开
- **默认位置：输入文件同目录**
- **默认文件名：输入文件名_predictions.csv**
- 可以修改文件名或选择其他目录

### 4. 等待预测完成
- 程序自动进行特征工程
- 进行预测
- 保存结果

### 5. 查看结果
结果文件包含：
- 所有原始列
- 新增的 `status_predict` 列

## 数据编码

- **编码格式**：UTF-8 with BOM
- **兼容性**：Excel 可直接打开，中文正常显示
- **分隔符**：逗号（,）

## 列顺序

输出文件保持原始列的顺序，`status_predict` 列添加在最后：

```
[原始列1], [原始列2], ..., [原始列N], status_predict
```

## 数据完整性

- ✅ 保留所有原始数据行
- ✅ 不修改原始数据值
- ✅ 保持原始数据顺序
- ✅ 保留缺失值（NaN）
- ✅ 每口井的预测遵循单向性规则（0→1→2→3）

## 注意事项

### 1. 文件覆盖

如果输出文件已存在，保存时会提示是否覆盖。

### 2. 权限问题

确保有写入目标目录的权限。

### 3. 磁盘空间

输出文件大小约为输入文件的 1.1 倍（仅增加一列）。

### 4. 数据质量

- 如果输入数据缺失值较多，预测质量可能受影响
- 建议输入数据符合格式要求

## 与之前版本的区别

| 特性 | 旧版本 | 新版本 |
|------|--------|--------|
| 输出列数 | 原始列 + 60+ 特征列 + 预测列 | 原始列 + 1 预测列 |
| 文件大小 | 很大（10-20倍） | 小（1.1倍） |
| 可读性 | 差（太多列） | 好（简洁清晰） |
| 保存位置 | 当前目录 | 输入文件同目录 |
| 列名 | `predicted_status` | `status_predict` |

## 技术细节

### 预测流程

```
输入数据
    ↓
数据预处理（转换数值类型）
    ↓
特征工程（生成60+特征，内存中）
    ↓
模型预测
    ↓
后处理规则（单向性约束）
    ↓
提取预测结果（仅保留预测列）
    ↓
添加到原始数据
    ↓
保存CSV（原始列 + status_predict）
```

### 内存使用

- 特征工程在内存中进行
- 不会保存到输出文件
- 内存使用后自动释放

## 示例代码

### Python 脚本调用

```python
from predict import predict_new_data

# 预测并保存（简洁输出）
df_result = predict_new_data(
    input_file='data.csv',
    output_file='data_predictions.csv'
)

# 输出文件只包含原始列 + status_predict
print(df_result.columns)
# 输出：Index(['序号', '转换后JH', 'JX', 'dy➗dx', 'status_predict'], dtype='object')
```

### 批量处理

```python
import os
from predict import predict_new_data

# 批量预测目录下所有CSV文件
data_dir = 'C:/Data/'
for filename in os.listdir(data_dir):
    if filename.endswith('.csv') and not filename.endswith('_predictions.csv'):
        input_path = os.path.join(data_dir, filename)
        output_path = os.path.join(data_dir, filename.replace('.csv', '_predictions.csv'))
        
        print(f"Processing: {filename}")
        predict_new_data(input_path, output_path)
        print(f"Saved: {output_path}\n")
```

## 常见问题

### Q1: 为什么只有一个新列？

**A**: 为了保持输出文件简洁清晰，便于在Excel中查看和分析。特征工程的中间列仅用于预测，不需要保存。

### Q2: 如果需要所有特征列怎么办？

**A**: 可以修改代码，将 `df_with_features` 保存即可。但通常不需要这些列。

### Q3: 预测列名为什么是 status_predict？

**A**: 
- 清晰明确：`status_predict` 表示这是预测的状态
- 与原始 `status` 列区分
- 符合命名习惯

### Q4: 可以修改输出列名吗？

**A**: 可以，修改代码中的列名即可：
```python
df_original['status_predict'] = ...
# 改为
df_original['predicted_status'] = ...
```

---

**版本**: 2.0 (简洁版)  
**更新日期**: 2024-10

