"""快速测试"""
print("测试开始...")

# 测试导入
try:
    import train_model_gui
    print("✓ GUI模块导入成功")
    
    from train_model_gui import WellStatusPredictor, TrainingGUI
    print("✓ 类导入成功")
    
    predictor = WellStatusPredictor()
    print("✓ 预测器实例化成功")
    
    print("\n所有测试通过！")
    print("可以运行: python train_model_gui.py")
except Exception as e:
    print(f"✗ 错误: {e}")
    import traceback
    traceback.print_exc()

