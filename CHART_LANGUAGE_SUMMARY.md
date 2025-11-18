# Chart Language Summary

All chart outputs have been modified to use **English only** with **Arial font** and **no special symbols**.

## Modified Files

### 1. explore_data.py
- Font: Arial
- Language: English only

**Generated Charts:**
- `status_distribution.png`
  - Overall Status Distribution
  - Status Distribution by Well (Top 10)
  
- `feature_distributions.png`
  - Distribution of Inclination Angle
  - Distribution of dy/dx
  - Inclination Angle by Status
  - dy/dx by Status
  
- `well_trajectories.png`
  - Well trajectories showing:
    - Inclination Angle
    - dy/dx
    - Wellbore Status
  
- `correlation_heatmap.png`
  - Feature Correlation Heatmap

### 2. train_model.py
- Font: Arial
- Language: English only

**Generated Charts:**
- `feature_importance.png`
  - Top N Feature Importance
  - X-axis: Importance Score
  - Y-axis: Feature Name
  
- `Confusion_Matrix_Raw.png`
  - Confusion Matrix (Raw Prediction)
  - Labels: Vertical(0), Build-up(1), Hold(2), Drop-off(3)
  
- `Confusion_Matrix_Corrected.png`
  - Confusion Matrix (After Rule Correction)
  - Labels: Vertical(0), Build-up(1), Hold(2), Drop-off(3)

## Terminology Mapping

| Chinese | English |
|---------|---------|
| 直井段 | Vertical |
| 造斜段 | Build-up |
| 稳斜段 | Hold |
| 降斜段 | Drop-off |
| 井斜角 | Inclination Angle |
| 井号 | Well ID |
| 序号 | Index |
| 特征重要性 | Feature Importance |
| 混淆矩阵 | Confusion Matrix |
| 真实标签 | True Label |
| 预测标签 | Predicted Label |
| 准确率 | Accuracy |
| 相关性热力图 | Correlation Heatmap |

## Key Changes

1. **Font Settings:**
   ```python
   from matplotlib import rcParams
   rcParams['font.family'] = 'Arial'
   rcParams['font.sans-serif'] = ['Arial']
   rcParams['axes.unicode_minus'] = False
   ```

2. **No Special Symbols:**
   - Replaced `dy➗dx` with `dy/dx` in chart labels
   - Used standard ASCII characters only

3. **All Text in English:**
   - Chart titles
   - Axis labels
   - Legend entries
   - Status labels
   - Console output for charts

## Notes

- Console output (print statements) may still contain Chinese for user feedback
- Data column names in CSV files remain unchanged
- Only chart/figure output has been modified to English
- All generated PNG files now use English labels exclusively

