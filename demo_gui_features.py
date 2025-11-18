"""
演示GUI程序的主要功能（不实际启动GUI窗口）
"""

import pandas as pd
import numpy as np

def demo_predictor_features():
    """演示预测器的功能"""
    print("="*60)
    print("井段状态预测器功能演示")
    print("="*60)
    
    try:
        from train_model_gui import WellStatusPredictor
        
        # 创建预测器实例
        print("\n1. 创建预测器实例...")
        predictor = WellStatusPredictor()
        print("   ✓ 预测器创建成功")
        
        # 加载数据
        print("\n2. 加载数据...")
        df = pd.read_csv('data001.csv')
        print(f"   ✓ 数据加载成功: {df.shape}")
        
        # 转换数据类型
        if 'dy➗dx' in df.columns:
            df['dy➗dx'] = pd.to_numeric(df['dy➗dx'], errors='coerce')
        
        print(f"   - 井号数量: {df['转换后JH'].nunique()}")
        print(f"   - Status分布:")
        for status, count in df['status'].value_counts().sort_index().items():
            print(f"     {status}: {count}")
        
        # 特征工程（只处理前2口井作为演示）
        print("\n3. 特征工程演示（前2口井）...")
        sample_wells = df['转换后JH'].unique()[:2]
        sample_df = df[df['转换后JH'].isin(sample_wells)].copy()
        
        df_with_features = predictor.create_features(sample_df)
        print(f"   ✓ 特征工程完成")
        print(f"   - 原始特征数: {len(sample_df.columns)}")
        print(f"   - 扩展后特征数: {len(df_with_features.columns)}")
        
        # 列出一些生成的特征
        new_features = [col for col in df_with_features.columns if col not in sample_df.columns]
        print(f"   - 生成的特征示例（前10个）:")
        for feat in new_features[:10]:
            print(f"     • {feat}")
        
        print("\n✓ 预测器功能正常")
        return True
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_gui_features():
    """演示GUI功能特性"""
    print("\n" + "="*60)
    print("GUI训练系统功能特性")
    print("="*60)
    
    features = [
        "✓ 可视化界面，易于操作",
        "✓ 文件选择功能（默认data001.csv）",
        "✓ 可调节的数据集划分比例",
        "  - 测试集比例：0.1-0.4（默认0.2）",
        "  - 验证集比例：0.05-0.3（默认0.1）",
        "✓ 可配置的超参数",
        "  - 学习率（默认0.05）",
        "  - 叶子节点数（默认31）",
        "  - 最大深度（默认7）",
        "  - 训练轮数（默认500）",
        "  - 早停轮数（默认50）",
        "  - 特征采样率（默认0.9）",
        "  - 样本采样率（默认0.8）",
        "✓ 实时训练监控",
        "  - 训练损失曲线",
        "  - 验证损失曲线",
        "  - 动态更新显示",
        "✓ 详细训练日志",
        "✓ 完整的评估结果",
        "✓ 自动模型保存"
    ]
    
    for feature in features:
        print(f"  {feature}")
    
    print("\n启动方式:")
    print("  python train_model_gui.py")

def main():
    """主函数"""
    print("\n井段状态预测模型训练系统 - GUI版本")
    print("功能演示与验证")
    print()
    
    # 演示预测器功能
    predictor_ok = demo_predictor_features()
    
    # 展示GUI功能
    demo_gui_features()
    
    # 总结
    print("\n" + "="*60)
    print("演示总结")
    print("="*60)
    
    if predictor_ok:
        print("✓ 核心功能验证通过")
        print("\n可以启动GUI进行完整训练:")
        print("  1. 运行: python train_model_gui.py")
        print("  2. 在GUI中选择数据文件")
        print("  3. 调整参数（可选）")
        print("  4. 点击'开始训练'按钮")
        print("  5. 观察训练过程和结果")
    else:
        print("✗ 功能验证失败，请检查错误信息")
    
    print("="*60)

if __name__ == "__main__":
    main()

