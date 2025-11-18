"""
快速开始示例 - 一键运行完整流程
"""

import os
import sys


def check_dependencies():
    """检查依赖包是否安装"""
    print("="*60)
    print("检查依赖包...")
    print("="*60)
    
    required_packages = [
        'pandas',
        'numpy',
        'sklearn',
        'lightgbm',
        'matplotlib',
        'seaborn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (未安装)")
            missing_packages.append(package)
    
    if missing_packages:
        print("\n请先安装缺失的包:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("\n所有依赖包已安装!")
    return True


def check_data_file():
    """检查数据文件是否存在"""
    print("\n" + "="*60)
    print("检查数据文件...")
    print("="*60)
    
    data_file = 'data001.csv'
    if not os.path.exists(data_file):
        print(f"✗ 数据文件 {data_file} 不存在")
        print("\n请确保 data001.csv 文件在当前目录下")
        return False
    
    print(f"✓ 数据文件 {data_file} 已找到")
    return True


def run_data_exploration():
    """运行数据探索"""
    print("\n" + "="*60)
    print("步骤 1: 数据探索")
    print("="*60)
    
    try:
        from explore_data import main as explore_main
        explore_main()
        print("\n✓ 数据探索完成")
        return True
    except Exception as e:
        print(f"\n✗ 数据探索出错: {e}")
        return False


def run_model_training():
    """运行模型训练"""
    print("\n" + "="*60)
    print("步骤 2: 模型训练")
    print("="*60)
    
    try:
        from train_model import main as train_main
        predictor, results = train_main()
        print("\n✓ 模型训练完成")
        return True, predictor, results
    except Exception as e:
        print(f"\n✗ 模型训练出错: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def print_summary(results):
    """打印训练结果摘要"""
    print("\n" + "="*60)
    print("训练结果摘要")
    print("="*60)
    
    if results:
        print(f"\n原始预测:")
        print(f"  准确率: {results.get('accuracy', 0):.4f}")
        print(f"  F1-Score: {results.get('f1_macro', 0):.4f}")
        
        if 'accuracy_corrected' in results:
            print(f"\n应用规则后:")
            print(f"  准确率: {results.get('accuracy_corrected', 0):.4f} "
                  f"(提升: {results.get('accuracy_corrected', 0) - results.get('accuracy', 0):+.4f})")
            print(f"  F1-Score: {results.get('f1_macro_corrected', 0):.4f} "
                  f"(提升: {results.get('f1_macro_corrected', 0) - results.get('f1_macro', 0):+.4f})")


def print_generated_files():
    """列出生成的文件"""
    print("\n" + "="*60)
    print("生成的文件")
    print("="*60)
    
    files = [
        ('well_status_model.txt', '训练好的模型'),
        ('feature_importance.png', '特征重要性图'),
        ('混淆矩阵_原始预测.png', '原始预测混淆矩阵'),
        ('混淆矩阵_规则修正后.png', '规则修正后混淆矩阵'),
        ('status_distribution.png', 'Status分布图'),
        ('feature_distributions.png', '特征分布图'),
        ('well_trajectories.png', '井轨迹图'),
        ('correlation_heatmap.png', '相关性热力图')
    ]
    
    print("\n生成的文件列表:")
    for filename, description in files:
        if os.path.exists(filename):
            print(f"  ✓ {filename:<35} - {description}")
        else:
            print(f"  ✗ {filename:<35} - {description} (未生成)")


def main():
    """主函数"""
    print("\n" + "🚀"*30)
    print("井段状态预测 - 快速开始")
    print("🚀"*30 + "\n")
    
    # 1. 检查依赖
    if not check_dependencies():
        return
    
    # 2. 检查数据文件
    if not check_data_file():
        return
    
    # 3. 询问用户是否继续
    print("\n" + "="*60)
    print("准备运行完整流程:")
    print("  1. 数据探索与可视化")
    print("  2. 特征工程")
    print("  3. 模型训练")
    print("  4. 模型评估")
    print("="*60)
    
    response = input("\n是否继续? (y/n): ").strip().lower()
    if response != 'y':
        print("已取消")
        return
    
    # 4. 数据探索
    print("\n" + "🔍"*30)
    if not run_data_exploration():
        print("\n数据探索失败，是否继续训练模型？")
        response = input("继续? (y/n): ").strip().lower()
        if response != 'y':
            return
    
    # 5. 模型训练
    print("\n" + "🤖"*30)
    success, predictor, results = run_model_training()
    
    if not success:
        print("\n❌ 训练失败")
        return
    
    # 6. 打印摘要
    print_summary(results)
    
    # 7. 列出生成的文件
    print_generated_files()
    
    # 8. 下一步指引
    print("\n" + "="*60)
    print("🎉 完成！")
    print("="*60)
    
    print("\n下一步操作:")
    print("1. 查看生成的图表文件了解数据和模型性能")
    print("2. 使用训练好的模型进行预测:")
    print("   python predict.py --input validation_without_label.csv --output predictions.csv")
    print("3. 查看详细文档: README_ML.md")
    print("4. 查看方案设计: ml_solution.md")
    
    print("\n模型文件: well_status_model.txt")
    print("可以使用此模型对新数据进行预测")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

