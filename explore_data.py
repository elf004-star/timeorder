"""
数据探索与可视化分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# 设置字体为Arial
rcParams['font.family'] = 'Arial'
rcParams['font.sans-serif'] = ['Arial']
rcParams['axes.unicode_minus'] = False

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# 忽略警告
import warnings
warnings.filterwarnings('ignore')


def load_data(file_path: str) -> pd.DataFrame:
    """加载数据"""
    print("="*60)
    print("数据加载")
    print("="*60)
    df = pd.read_csv(file_path)
    
    # 转换Dogleg Severity为数值类型
    if 'Dogleg Severity' in df.columns:
        df['Dogleg Severity'] = pd.to_numeric(df['Dogleg Severity'], errors='coerce')
    
    print(f"\n数据文件: {file_path}")
    print(f"数据形状: {df.shape}")
    print(f"\n列名: {list(df.columns)}")
    return df


def basic_statistics(df: pd.DataFrame):
    """基本统计信息"""
    print("\n" + "="*60)
    print("基本统计信息")
    print("="*60)
    
    # 数据概览
    print("\n数据概览:")
    print(df.head())
    
    # 数据类型
    print("\n数据类型:")
    print(df.dtypes)
    
    # 缺失值
    print("\n缺失值统计:")
    missing = df.isnull().sum()
    missing_pct = 100 * df.isnull().sum() / len(df)
    missing_table = pd.DataFrame({
        '缺失数量': missing,
        '缺失比例(%)': missing_pct
    })
    print(missing_table[missing_table['缺失数量'] > 0])
    
    # 数值列统计
    print("\n数值列统计:")
    print(df.describe())
    
    # 井号信息
    print("\n井号信息:")
    print(f"总井数: {df['转换后JH'].nunique()}")
    print(f"井号列表: {df['转换后JH'].unique()[:10]}...")  # 显示前10个
    
    # 每口井的数据点数量
    well_counts = df.groupby('转换后JH').size()
    print(f"\n每口井的平均数据点数: {well_counts.mean():.1f}")
    print(f"最少数据点: {well_counts.min()}")
    print(f"最多数据点: {well_counts.max()}")
    
    # Status分布
    if 'status' in df.columns:
        print("\nStatus分布:")
        status_counts = df['status'].value_counts().sort_index()
        for status, count in status_counts.items():
            pct = count / len(df) * 100
            status_name = ['直井段', '造斜段', '稳斜段', '降斜段'][int(status)]
            print(f"  {status} ({status_name}): {count} ({pct:.2f}%)")


def plot_status_distribution(df: pd.DataFrame):
    """Plot status distribution"""
    if 'status' not in df.columns:
        print("No status column in data, skipping this analysis")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Overall distribution
    status_counts = df['status'].value_counts().sort_index()
    status_labels = ['Vertical(0)', 'Build-up(1)', 'Hold(2)', 'Drop-off(3)']
    
    axes[0].bar(range(len(status_counts)), status_counts.values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
    axes[0].set_xticks(range(len(status_counts)))
    axes[0].set_xticklabels([status_labels[i] for i in status_counts.index])
    axes[0].set_ylabel('Number of Data Points', fontsize=18)
    axes[0].set_title('Overall Status Distribution', fontsize=19, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].tick_params(axis='both', labelsize=18)
    
    # Add value labels
    for i, v in enumerate(status_counts.values):
        axes[0].text(i, v + max(status_counts.values)*0.01, str(v), 
                    ha='center', va='bottom', fontweight='bold', fontsize=18)
    
    # 2. Status distribution per well
    well_status = df.groupby(['转换后JH', 'status']).size().unstack(fill_value=0)
    well_status_pct = well_status.div(well_status.sum(axis=1), axis=0) * 100
    
    # Show top 10 wells
    well_status_pct_top10 = well_status_pct.head(10)
    well_status_pct_top10.plot(kind='bar', stacked=True, ax=axes[1],
                               color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
    axes[1].set_ylabel('Percentage (%)', fontsize=18)
    axes[1].set_xlabel('Well ID', fontsize=18)
    axes[1].set_title('Status Distribution by Well (Top 10)', fontsize=19, fontweight='bold')
    axes[1].legend(['Vertical(0)', 'Build-up(1)', 'Hold(2)', 'Drop-off(3)'], 
                   loc='upper right', fontsize=14)
    axes[1].tick_params(axis='x', rotation=45, labelsize=18)
    axes[1].tick_params(axis='y', labelsize=18)
    
    plt.tight_layout()
    plt.savefig('status_distribution.png', dpi=300, bbox_inches='tight')
    print("\nStatus distribution plot saved: status_distribution.png")


def plot_feature_distributions(df: pd.DataFrame):
    """Plot feature distributions"""
    # Ensure correct data types
    df_plot = df.copy()
    df_plot['Dogleg Severity'] = pd.to_numeric(df_plot['Dogleg Severity'], errors='coerce')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Inclination angle distribution
    axes[0, 0].hist(df_plot['JX'].dropna(), bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Inclination Angle (degree)', fontsize=20)
    axes[0, 0].set_ylabel('Frequency', fontsize=20)
    axes[0, 0].set_title('Distribution of Inclination Angle', fontsize=22, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].tick_params(axis='both', labelsize=20)
    
    # 2. Dogleg Severity distribution
    axes[0, 1].hist(df_plot['Dogleg Severity'].dropna(), bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Dogleg Severity', fontsize=20)
    axes[0, 1].set_ylabel('Frequency', fontsize=20)
    axes[0, 1].set_title('Distribution of Dogleg Severity', fontsize=22, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].tick_params(axis='both', labelsize=20)
    
    # 3. Inclination angle by Status boxplot
    if 'status' in df_plot.columns:
        status_labels = ['Vertical\n(0)', 'Build-up\n(1)', 'Hold\n(2)', 'Drop-off\n(3)']
        df_plot['status_label'] = df_plot['status'].map({0: status_labels[0], 
                                                          1: status_labels[1], 
                                                          2: status_labels[2], 
                                                          3: status_labels[3]})
        
        sns.boxplot(data=df_plot, x='status_label', y='JX', ax=axes[1, 0], 
                   hue='status_label', palette=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'], 
                   legend=False)
        axes[1, 0].set_xlabel('Status', fontsize=20)
        axes[1, 0].set_ylabel('Inclination Angle (degree)', fontsize=20)
        axes[1, 0].set_title('Inclination Angle by Status', fontsize=22, fontweight='bold')
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].tick_params(axis='both', labelsize=20)
        
        # 4. Dogleg Severity by Status boxplot
        sns.boxplot(data=df_plot, x='status_label', y='Dogleg Severity', ax=axes[1, 1],
                   hue='status_label', palette=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'], 
                   legend=False)
        axes[1, 1].set_xlabel('Status', fontsize=20)
        axes[1, 1].set_ylabel('Dogleg Severity', fontsize=20)
        axes[1, 1].set_title('Dogleg Severity by Status', fontsize=22, fontweight='bold')
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].tick_params(axis='both', labelsize=20)
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    print("Feature distribution plot saved: feature_distributions.png")


def plot_well_trajectory(df: pd.DataFrame, well_name: str = None, n_wells: int = 3):
    """Plot well trajectories"""
    # Select wells
    if well_name is not None:
        wells = [well_name]
    else:
        wells = df['转换后JH'].unique()[:n_wells]
    
    n_wells_actual = len(wells)
    fig, axes = plt.subplots(n_wells_actual, 3, figsize=(15, 5*n_wells_actual))
    
    if n_wells_actual == 1:
        axes = axes.reshape(1, -1)
    
    for idx, well in enumerate(wells):
        well_data = df[df['转换后JH'] == well].copy().sort_values('序号')
        
        # Ensure numeric types
        well_data['Dogleg Severity'] = pd.to_numeric(well_data['Dogleg Severity'], errors='coerce')
        
        # 1. Inclination angle curve
        axes[idx, 0].plot(well_data['序号'], well_data['JX'], 'b-', linewidth=2)
        axes[idx, 0].set_xlabel('Index', fontsize=22)
        axes[idx, 0].set_ylabel('Inclination Angle (degree)', fontsize=22)
        axes[idx, 0].set_title(f'Well: {well} - Inclination Angle', fontsize=24, fontweight='bold')
        axes[idx, 0].grid(alpha=0.3)
        axes[idx, 0].tick_params(axis='both', labelsize=22)
        
        # 2. Dogleg Severity curve
        axes[idx, 1].plot(well_data['序号'], well_data['Dogleg Severity'], 'g-', linewidth=2)
        axes[idx, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[idx, 1].set_xlabel('Index', fontsize=22)
        axes[idx, 1].set_ylabel('Dogleg Severity', fontsize=22)
        axes[idx, 1].set_title('Dogleg Severity', fontsize=24, fontweight='bold')
        axes[idx, 1].grid(alpha=0.3)
        axes[idx, 1].tick_params(axis='both', labelsize=22)
        
        # 3. Status curve
        if 'status' in well_data.columns:
            # Fill different status regions with colors
            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
            for status in range(4):
                mask = well_data['status'] == status
                if mask.any():
                    axes[idx, 2].scatter(well_data[mask]['序号'], 
                                       well_data[mask]['status'],
                                       c=colors[status], s=20, alpha=0.6,
                                       label=['Vertical', 'Build-up', 'Hold', 'Drop-off'][status])
            
            axes[idx, 2].plot(well_data['序号'], well_data['status'], 'k-', 
                            linewidth=1.5, alpha=0.5)
            axes[idx, 2].set_xlabel('Index', fontsize=22)
            axes[idx, 2].set_ylabel('Status', fontsize=22)
            axes[idx, 2].set_title('Wellbore Status', fontsize=24, fontweight='bold')
            axes[idx, 2].set_yticks([0, 1, 2, 3])
            axes[idx, 2].set_yticklabels(['Vertical', 'Build-up', 'Hold', 'Drop-off'], fontsize=22)
            axes[idx, 2].legend(loc='best', fontsize=18)
            axes[idx, 2].grid(alpha=0.3)
            axes[idx, 2].tick_params(axis='x', labelsize=22)
    
    plt.tight_layout()
    filename = 'well_trajectories.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Well trajectory plot saved: {filename}")


def analyze_status_transitions(df: pd.DataFrame):
    """分析Status转换模式"""
    if 'status' not in df.columns:
        print("数据中没有status列，跳过此分析")
        return
    
    print("\n" + "="*60)
    print("Status转换模式分析")
    print("="*60)
    
    transition_patterns = []
    
    for well_name, group in df.groupby('转换后JH'):
        group = group.sort_values('序号')
        status_sequence = group['status'].values
        
        # 找到状态转换点
        transitions = []
        prev_status = status_sequence[0]
        for status in status_sequence[1:]:
            if status != prev_status:
                transitions.append(f"{prev_status}->{status}")
                prev_status = status
        
        # 完整的状态序列
        unique_statuses = []
        prev = -1
        for s in status_sequence:
            if s != prev:
                unique_statuses.append(s)
                prev = s
        
        pattern = '->'.join(map(str, unique_statuses))
        transition_patterns.append({
            '井号': well_name,
            '转换模式': pattern,
            '转换次数': len(transitions)
        })
    
    patterns_df = pd.DataFrame(transition_patterns)
    
    # 统计转换模式
    print("\n转换模式统计:")
    pattern_counts = patterns_df['转换模式'].value_counts()
    for pattern, count in pattern_counts.head(10).items():
        print(f"  {pattern}: {count} 口井")
    
    # 转换次数统计
    print("\n转换次数统计:")
    print(patterns_df['转换次数'].describe())
    
    return patterns_df


def correlation_analysis(df: pd.DataFrame):
    """Correlation analysis"""
    print("\n" + "="*60)
    print("Feature Correlation Analysis")
    print("="*60)
    
    # Select numeric columns and ensure numeric types
    df_numeric = df.copy()
    df_numeric['Dogleg Severity'] = pd.to_numeric(df_numeric['Dogleg Severity'], errors='coerce')
    
    numeric_cols = ['JX', 'Dogleg Severity']
    if 'status' in df.columns:
        numeric_cols.append('status')
    
    corr_matrix = df_numeric[numeric_cols].corr()
    
    print("\nCorrelation Matrix:")
    print(corr_matrix)
    
    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, square=True, linewidths=1)
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("\nCorrelation heatmap saved: correlation_heatmap.png")


def main():
    """主函数"""
    # 加载数据
    df = load_data('data001.csv')
    
    # 基本统计
    basic_statistics(df)
    
    # Status分布
    plot_status_distribution(df)
    
    # 特征分布
    plot_feature_distributions(df)
    
    # Well trajectory visualization
    print("\nPlotting well trajectories...")
    plot_well_trajectory(df, n_wells=3)
    
    # Status transition analysis
    patterns_df = analyze_status_transitions(df)
    
    # Correlation analysis
    correlation_analysis(df)
    
    print("\n" + "="*60)
    print("Data Exploration Completed!")
    print("="*60)
    print("\nGenerated plot files:")
    print("  - status_distribution.png")
    print("  - feature_distributions.png")
    print("  - well_trajectories.png")
    print("  - correlation_heatmap.png")


if __name__ == "__main__":
    main()

