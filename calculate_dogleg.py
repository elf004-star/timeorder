import pandas as pd
import numpy as np
import math

# 读取数据
df = pd.read_csv('data/data.csv')

# 将角度转换为弧度
def deg_to_rad(deg):
    return math.radians(deg)

# 保留3位有效数字
def round_to_3_sig_figs(value):
    """
    将数值保留3位有效数字，返回浮点数
    """
    if value == 0.0:
        return 0.0
    
    # 使用字符串格式化来确保3位有效数字
    # 使用g格式，自动选择最合适的表示方式
    formatted = f"{value:.3g}"
    
    # 转换回浮点数
    return float(formatted)

# 计算狗腿度
def calculate_dogleg(d1, theta1, phi1, d2, theta2, phi2):
    """
    计算两个连续测点之间的狗腿度
    
    参数:
    d1, d2: 测量深度 (米)
    theta1, theta2: 井斜角 (度)
    phi1, phi2: 方位角 (度)
    
    返回:
    DLS: 狗腿度 (deg/30m)
    """
    # 转换为弧度
    theta1_rad = deg_to_rad(theta1)
    theta2_rad = deg_to_rad(theta2)
    phi1_rad = deg_to_rad(phi1)
    phi2_rad = deg_to_rad(phi2)
    
    # 计算总角度变化 α
    delta_theta = theta2_rad - theta1_rad
    delta_phi = phi2_rad - phi1_rad
    
    # 使用Lubinski公式
    sin_squared_half_theta = math.sin(delta_theta / 2) ** 2
    sin_squared_half_phi = math.sin(delta_phi / 2) ** 2
    
    inside_sqrt = sin_squared_half_theta + math.sin(theta1_rad) * math.sin(theta2_rad) * sin_squared_half_phi
    
    # 避免数值误差导致sqrt内为负数
    inside_sqrt = max(0, min(1, inside_sqrt))
    
    alpha = 2 * math.asin(math.sqrt(inside_sqrt))
    
    # 计算深度差
    delta_d = d2 - d1
    
    # 避免除零
    if delta_d == 0:
        return 0.0
    
    # 计算DLS (deg/30m)
    dls = (alpha * 30 / delta_d) * (180 / math.pi)
    
    return dls

# 按井号分组处理
result_rows = []

for well_name in df['转换后JH'].unique():
    well_data = df[df['转换后JH'] == well_name].copy()
    well_data = well_data.sort_values('XJS').reset_index(drop=True)
    
    dls_values = []
    
    # 计算每个点的狗腿度（除了最后一个点）
    for i in range(len(well_data) - 1):
        d1 = well_data.iloc[i]['XJS']
        theta1 = well_data.iloc[i]['JX']
        phi1 = well_data.iloc[i]['FW']
        
        d2 = well_data.iloc[i + 1]['XJS']
        theta2 = well_data.iloc[i + 1]['JX']
        phi2 = well_data.iloc[i + 1]['FW']
        
        dls = calculate_dogleg(d1, theta1, phi1, d2, theta2, phi2)
        dls_values.append(dls)
    
    # 最后一个点沿用上一个点的狗腿度
    if len(dls_values) > 0:
        dls_values.append(dls_values[-1])
    else:
        # 如果只有一个点，狗腿度为0
        dls_values.append(0.0)
    
    # 构建结果行
    for idx, (_, row) in enumerate(well_data.iterrows()):
        result_rows.append({
            '序号': row['序号'],
            '转换后JH': row['转换后JH'],
            'JX': row['JX'],
            'Dogleg Severity': round_to_3_sig_figs(dls_values[idx]),
            '关键点': row['关键点']
        })

# 创建结果DataFrame
result_df = pd.DataFrame(result_rows)

# 保存结果
result_df.to_csv('data/data_with_dogleg.csv', index=False, encoding='utf-8-sig')

print(f"计算完成！共处理 {len(result_df)} 条记录")
print(f"结果已保存到: data/data_with_dogleg.csv")
print(f"\n前10条结果预览:")
print(result_df.head(10))

