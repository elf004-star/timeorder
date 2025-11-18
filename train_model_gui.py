"""
井段状态预测模型训练脚本 - GUI版本
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import lightgbm as lgb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import os

# Set Arial font
from matplotlib import rcParams
rcParams['font.family'] = 'Arial'
rcParams['font.sans-serif'] = ['Arial']
rcParams['axes.unicode_minus'] = False


class WellStatusPredictor:
    """井段状态预测器"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
    
    def load_model(self, model_path: str):
        """加载已训练的模型及相关配置"""
        import pickle
        
        # 加载模型
        self.model = lgb.Booster(model_file=model_path)
        print(f"模型已加载: {model_path}")
        
        # 尝试加载scaler和feature_columns
        scaler_path = model_path.replace('.txt', '_scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                data = pickle.load(f)
                self.scaler = data['scaler']
                self.feature_columns = data['feature_columns']
            print(f"Scaler已加载: {scaler_path}")
        else:
            print(f"警告: 未找到scaler文件 {scaler_path}，预测可能不准确")
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        特征工程：为每口井创建时序特征
        """
        print("开始特征工程...")
        
        # 按井号分组处理
        features_list = []
        
        for well_name, group in df.groupby('转换后JH'):
            group = group.copy().sort_values('序号').reset_index(drop=True)
            
            # 基础特征 - 确保数值类型
            group['JX'] = pd.to_numeric(group['JX'], errors='coerce').fillna(0)
            group['dy_dx'] = pd.to_numeric(group['dy➗dx'], errors='coerce').fillna(0)
            
            # 1. 滑动窗口统计特征（窗口大小：3, 5, 10）
            for window in [3, 5, 10]:
                # JX的统计特征
                group[f'JX_mean_{window}'] = group['JX'].rolling(window=window, min_periods=1).mean()
                group[f'JX_std_{window}'] = group['JX'].rolling(window=window, min_periods=1).std().fillna(0)
                group[f'JX_max_{window}'] = group['JX'].rolling(window=window, min_periods=1).max()
                group[f'JX_min_{window}'] = group['JX'].rolling(window=window, min_periods=1).min()
                
                # dy/dx的统计特征
                group[f'dy_dx_mean_{window}'] = group['dy_dx'].rolling(window=window, min_periods=1).mean()
                group[f'dy_dx_std_{window}'] = group['dy_dx'].rolling(window=window, min_periods=1).std().fillna(0)
                group[f'dy_dx_max_{window}'] = group['dy_dx'].rolling(window=window, min_periods=1).max()
                group[f'dy_dx_min_{window}'] = group['dy_dx'].rolling(window=window, min_periods=1).min()
            
            # 2. 滞后特征（前N个点的值）
            for lag in [1, 2, 3, 5]:
                group[f'JX_lag_{lag}'] = group['JX'].shift(lag).fillna(0)
                group[f'dy_dx_lag_{lag}'] = group['dy_dx'].shift(lag).fillna(0)
            
            # 3. 差分特征
            group['JX_diff_1'] = group['JX'].diff().fillna(0)
            group['JX_diff_2'] = group['JX'].diff(2).fillna(0)
            group['dy_dx_diff_1'] = group['dy_dx'].diff().fillna(0)
            
            # 4. 累积特征
            group['JX_cumsum'] = group['JX'].cumsum()
            group['dy_dx_cumsum'] = group['dy_dx'].cumsum()
            
            # 5. 位置特征
            group['depth_position'] = np.arange(len(group)) / len(group)  # 相对深度位置
            group['depth_from_start'] = np.arange(len(group))  # 从井口的距离
            
            # 6. 领域知识特征
            group['JX_abs'] = group['JX'].abs()
            group['dy_dx_abs'] = group['dy_dx'].abs()
            group['is_stable'] = (group['dy_dx_abs'] < 0.01).astype(int)  # 井斜角是否稳定
            group['is_increasing'] = (group['dy_dx'] > 0.02).astype(int)  # 是否在造斜
            group['is_decreasing'] = (group['dy_dx'] < -0.01).astype(int)  # 是否在降斜
            
            # 7. 趋势特征（最近N个点的趋势）
            for window in [3, 5]:
                group[f'JX_trend_{window}'] = group['JX'].rolling(window=window, min_periods=1).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                ).fillna(0)
            
            features_list.append(group)
        
        result_df = pd.concat(features_list, ignore_index=True)
        print(f"特征工程完成，生成 {len(result_df.columns)} 个特征")
        
        return result_df
    
    def prepare_data(self, df: pd.DataFrame, val_size: float = 0.2) -> Tuple:
        """
        准备训练数据：按井号分割数据集（只划分训练集和验证集）
        """
        print("准备数据...")
        
        # 获取所有井号
        well_names = df['转换后JH'].unique()
        n_wells = len(well_names)
        
        print(f"总井数: {n_wells}")
        
        # 计算分割数量
        n_val = int(n_wells * val_size)
        n_train = n_wells - n_val
        
        # 随机打乱井号
        np.random.seed(42)
        shuffled_wells = np.random.permutation(well_names)
        
        # 分配井号到不同集合（只有训练集和验证集）
        train_wells = shuffled_wells[:n_train]
        val_wells = shuffled_wells[n_train:]
        
        # 根据井号分割数据
        train_df = df[df['转换后JH'].isin(train_wells)].copy()
        val_df = df[df['转换后JH'].isin(val_wells)].copy()
        
        print(f"训练集: {len(train_df)} 样本 ({len(train_wells)} 口井)")
        print(f"验证集: {len(val_df)} 样本 ({len(val_wells)} 口井)")
        
        # 提取特征和标签
        exclude_cols = ['序号', '转换后JH', 'status', 'dy➗dx']
        self.feature_columns = [col for col in train_df.columns if col not in exclude_cols]
        
        X_train = train_df[self.feature_columns].values
        y_train = train_df['status'].values
        
        X_val = val_df[self.feature_columns].values
        y_val = val_df['status'].values
        
        # 标准化特征
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        
        return (X_train, y_train), (X_val, y_val), val_df
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray, params: Dict,
              callback=None) -> Dict:
        """
        训练LightGBM模型
        """
        print("\n开始训练模型...")
        
        # 创建数据集
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # 训练历史记录
        train_losses = []
        val_losses = []
        
        # 自定义回调函数
        def record_eval(env):
            train_losses.append(env.evaluation_result_list[0][2])
            val_losses.append(env.evaluation_result_list[1][2])
            if callback:
                callback(len(train_losses), train_losses[-1], val_losses[-1])
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=params.get('early_stopping_rounds', 50)),
            lgb.log_evaluation(period=10),
            record_eval
        ]
        
        # 训练模型
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=params.get('num_boost_round', 500),
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=callbacks
        )
        
        print("模型训练完成")
        
        # 返回训练历史
        return {
            'best_iteration': self.model.best_iteration,
            'best_score': self.model.best_score,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        y_pred_proba = self.model.predict(X, num_iteration=self.model.best_iteration)
        return np.argmax(y_pred_proba, axis=1)
    
    def apply_rules(self, df: pd.DataFrame, predictions: np.ndarray) -> np.ndarray:
        """
        应用后处理规则：确保状态转换的单向性
        """
        print("\n应用后处理规则...")
        
        corrected_predictions = predictions.copy()
        
        # 按井号分组处理
        start_idx = 0
        for well_name, group in df.groupby('转换后JH'):
            end_idx = start_idx + len(group)
            well_pred = predictions[start_idx:end_idx]
            
            # 强制单向性：0 -> 1 -> 2 -> 3
            corrected = []
            current_max = 0
            
            for pred in well_pred:
                if pred >= current_max:
                    current_max = pred
                corrected.append(current_max)
            
            corrected_predictions[start_idx:end_idx] = corrected
            start_idx = end_idx
        
        # 统计修正的数量
        n_corrected = np.sum(predictions != corrected_predictions)
        print(f"修正了 {n_corrected} 个预测 ({n_corrected/len(predictions)*100:.2f}%)")
        
        return corrected_predictions
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 y_pred_corrected: np.ndarray = None) -> Dict:
        """
        评估模型性能
        """
        print("\n" + "="*60)
        print("Model Evaluation Results")
        print("="*60)
        
        # Raw prediction evaluation
        print("\n[Raw Prediction]")
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        
        print(f"Accuracy: {acc:.4f}")
        print(f"F1-Score (macro): {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, 
                                   target_names=['Vertical(0)', 'Build-up(1)', 'Hold(2)', 'Drop-off(3)'],
                                   zero_division=0))
        
        results = {
            'accuracy': acc,
            'f1_macro': f1,
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        # Corrected prediction evaluation
        if y_pred_corrected is not None:
            print("\n[After Rule Correction]")
            acc_corrected = accuracy_score(y_true, y_pred_corrected)
            f1_corrected = f1_score(y_true, y_pred_corrected, average='macro')
            
            print(f"Accuracy: {acc_corrected:.4f} (improvement: {acc_corrected-acc:+.4f})")
            print(f"F1-Score (macro): {f1_corrected:.4f} (improvement: {f1_corrected-f1:+.4f})")
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred_corrected,
                                       target_names=['Vertical(0)', 'Build-up(1)', 'Hold(2)', 'Drop-off(3)'],
                                       zero_division=0))
            
            results['accuracy_corrected'] = acc_corrected
            results['f1_macro_corrected'] = f1_corrected
            results['confusion_matrix_corrected'] = confusion_matrix(y_true, y_pred_corrected)
        
        return results
    
    def save_model(self, filename: str = "well_status_model.txt", params: Dict = None):
        """保存模型到Models文件夹"""
        import json
        import pickle
        from datetime import datetime
        
        # 创建Models文件夹
        models_dir = "Models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            print(f"创建文件夹: {models_dir}")
        
        # 保存模型到Models文件夹
        model_path = os.path.join(models_dir, filename)
        self.model.save_model(model_path)
        print(f"\n模型已保存到: {model_path}")
        
        # 保存scaler和feature_columns
        scaler_filename = filename.replace('.txt', '_scaler.pkl')
        scaler_path = os.path.join(models_dir, scaler_filename)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }, f)
        print(f"Scaler已保存到: {scaler_path}")
        
        # 保存训练参数
        if params is not None:
            params_info = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_file': filename,
                'best_iteration': self.model.best_iteration,
                'best_score': self.model.best_score,
                'parameters': params
            }
            
            params_filename = filename.replace('.txt', '_params.json')
            params_path = os.path.join(models_dir, params_filename)
            
            with open(params_path, 'w', encoding='utf-8') as f:
                json.dump(params_info, f, indent=4, ensure_ascii=False)
            
            print(f"参数已保存到: {params_path}")


class TrainingGUI:
    """训练界面GUI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("井段状态预测模型训练系统")
        self.root.geometry("1200x800")
        
        self.predictor = WellStatusPredictor()
        self.training_thread = None
        self.is_training = False
        
        # 训练数据
        self.train_losses = []
        self.val_losses = []
        
        self.setup_ui()
        
    def setup_ui(self):
        """设置UI界面"""
        
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # ========== 左侧控制面板 ==========
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        row = 0
        
        # 文件选择
        ttk.Label(control_frame, text="数据文件:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.file_var = tk.StringVar(value="data001.csv")
        file_entry = ttk.Entry(control_frame, textvariable=self.file_var, width=30)
        file_entry.grid(row=row, column=1, pady=5, padx=5)
        ttk.Button(control_frame, text="浏览...", command=self.browse_file).grid(row=row, column=2, pady=5)
        row += 1
        
        # 分隔线
        ttk.Separator(control_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        # 数据集划分
        ttk.Label(control_frame, text="数据集划分", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=5)
        row += 1
        
        ttk.Label(control_frame, text="验证集比例:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.val_size_var = tk.DoubleVar(value=0.2)
        
        # 创建滑动条，范围0.1-0.9，步进0.1
        val_scale = ttk.Scale(
            control_frame, 
            from_=0.1, 
            to=0.9, 
            variable=self.val_size_var, 
            orient=tk.HORIZONTAL,
            length=150
        )
        # 拖动时自动取整到0.1
        val_scale.configure(command=lambda v: self.val_size_var.set(round(float(v) * 10) / 10))
        val_scale.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)
        
        # 显示标签，格式化为一位小数
        self.val_label = ttk.Label(control_frame, text="0.2", width=5)
        self.val_label.grid(row=row, column=2, pady=5)
        self.val_size_var.trace('w', lambda *args: self.val_label.config(text=f"{self.val_size_var.get():.1f}"))
        row += 1
        
        # 添加说明文字
        hint_label = ttk.Label(control_frame, text="(可选: 0.1~0.9, 步进0.1)", font=('Arial', 8), foreground='gray')
        hint_label.grid(row=row, column=1, sticky=tk.W, padx=5)
        row += 1
        
        # 分隔线
        ttk.Separator(control_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        # 超参数设置
        ttk.Label(control_frame, text="模型超参数", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=5)
        row += 1
        
        # Learning Rate
        ttk.Label(control_frame, text="学习率:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.lr_var = tk.DoubleVar(value=0.05)
        ttk.Entry(control_frame, textvariable=self.lr_var, width=15).grid(row=row, column=1, sticky=tk.W, pady=5, padx=5)
        row += 1
        
        # Num Leaves
        ttk.Label(control_frame, text="叶子节点数:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.num_leaves_var = tk.IntVar(value=31)
        ttk.Entry(control_frame, textvariable=self.num_leaves_var, width=15).grid(row=row, column=1, sticky=tk.W, pady=5, padx=5)
        row += 1
        
        # Max Depth
        ttk.Label(control_frame, text="最大深度:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.max_depth_var = tk.IntVar(value=7)
        ttk.Entry(control_frame, textvariable=self.max_depth_var, width=15).grid(row=row, column=1, sticky=tk.W, pady=5, padx=5)
        row += 1
        
        # Num Boost Round
        ttk.Label(control_frame, text="训练轮数:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.num_boost_var = tk.IntVar(value=500)
        ttk.Entry(control_frame, textvariable=self.num_boost_var, width=15).grid(row=row, column=1, sticky=tk.W, pady=5, padx=5)
        row += 1
        
        # Early Stopping
        ttk.Label(control_frame, text="早停轮数:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.early_stop_var = tk.IntVar(value=50)
        ttk.Entry(control_frame, textvariable=self.early_stop_var, width=15).grid(row=row, column=1, sticky=tk.W, pady=5, padx=5)
        row += 1
        
        # Min Child Samples
        ttk.Label(control_frame, text="最小样本数:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.min_child_var = tk.IntVar(value=20)
        ttk.Entry(control_frame, textvariable=self.min_child_var, width=15).grid(row=row, column=1, sticky=tk.W, pady=5, padx=5)
        row += 1
        
        # Feature Fraction
        ttk.Label(control_frame, text="特征采样率:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.feature_frac_var = tk.DoubleVar(value=0.9)
        ttk.Entry(control_frame, textvariable=self.feature_frac_var, width=15).grid(row=row, column=1, sticky=tk.W, pady=5, padx=5)
        row += 1
        
        # Bagging Fraction
        ttk.Label(control_frame, text="样本采样率:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.bagging_frac_var = tk.DoubleVar(value=0.8)
        ttk.Entry(control_frame, textvariable=self.bagging_frac_var, width=15).grid(row=row, column=1, sticky=tk.W, pady=5, padx=5)
        row += 1
        
        # 分隔线
        ttk.Separator(control_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        # 开始训练按钮
        self.train_button = ttk.Button(control_frame, text="开始训练", command=self.start_training)
        self.train_button.grid(row=row, column=0, columnspan=3, pady=10)
        row += 1
        
        # 分隔线
        ttk.Separator(control_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        # 预测功能
        ttk.Label(control_frame, text="模型预测", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=5)
        row += 1
        
        # 预测按钮
        self.predict_button = ttk.Button(control_frame, text="选择模型并预测", command=self.start_prediction)
        self.predict_button.grid(row=row, column=0, columnspan=3, pady=5)
        row += 1
        
        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        # 状态标签
        self.status_var = tk.StringVar(value="准备就绪")
        status_label = ttk.Label(control_frame, textvariable=self.status_var, foreground="blue")
        status_label.grid(row=row, column=0, columnspan=3, pady=5)
        row += 1
        
        # ========== 右侧显示面板 ==========
        display_frame = ttk.Frame(main_frame)
        display_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        display_frame.rowconfigure(0, weight=1)
        display_frame.columnconfigure(0, weight=1)
        
        # 创建notebook（多标签页）
        notebook = ttk.Notebook(display_frame)
        notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 训练监控标签页
        train_frame = ttk.Frame(notebook)
        notebook.add(train_frame, text="训练监控")
        
        # 创建matplotlib图表
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('Iteration')
        self.ax.set_ylabel('Loss (Log Loss)')
        self.ax.set_title('Training and Validation Loss')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(['Train Loss', 'Validation Loss'])
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=train_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 日志标签页
        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text="训练日志")
        
        # 创建文本框和滚动条
        log_scroll = ttk.Scrollbar(log_frame)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_text = tk.Text(log_frame, wrap=tk.WORD, yscrollcommand=log_scroll.set)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        log_scroll.config(command=self.log_text.yview)
        
        # 结果标签页
        result_frame = ttk.Frame(notebook)
        notebook.add(result_frame, text="训练结果")
        
        self.result_text = tk.Text(result_frame, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
    def browse_file(self):
        """浏览文件"""
        filename = filedialog.askopenfilename(
            title="选择CSV文件",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=os.getcwd()
        )
        if filename:
            self.file_var.set(filename)
    
    def log(self, message):
        """记录日志"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        print(message)  # 同时打印到控制台
    
    def update_plot(self, iteration, train_loss, val_loss):
        """更新训练曲线"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        self.ax.clear()
        self.ax.plot(self.train_losses, label='Train Loss', color='blue', linewidth=2)
        self.ax.plot(self.val_losses, label='Validation Loss', color='red', linewidth=2)
        self.ax.set_xlabel('Iteration')
        self.ax.set_ylabel('Loss (Log Loss)')
        self.ax.set_title('Training and Validation Loss')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        self.canvas.draw()
        
        # 更新进度
        progress = (iteration / self.num_boost_var.get()) * 100
        self.progress_var.set(progress)
        self.status_var.set(f"训练中... 迭代 {iteration}/{self.num_boost_var.get()}")
    
    def start_training(self):
        """开始训练"""
        if self.is_training:
            messagebox.showwarning("警告", "训练正在进行中！")
            return
        
        # 检查文件是否存在
        if not os.path.exists(self.file_var.get()):
            messagebox.showerror("错误", f"文件不存在: {self.file_var.get()}")
            return
        
        # 禁用训练按钮
        self.train_button.config(state='disabled')
        self.is_training = True
        
        # 清空之前的数据
        self.train_losses = []
        self.val_losses = []
        self.log_text.delete(1.0, tk.END)
        self.result_text.delete(1.0, tk.END)
        
        # 在新线程中运行训练
        self.training_thread = threading.Thread(target=self.train_model)
        self.training_thread.start()
    
    def train_model(self):
        """训练模型（在独立线程中运行）"""
        try:
            # 1. 加载数据
            self.log("="*60)
            self.log("井段状态预测模型训练")
            self.log("="*60)
            self.log("\n1. 加载数据...")
            
            df = pd.read_csv(self.file_var.get())
            
            # 转换dy➗dx为数值类型
            if 'dy➗dx' in df.columns:
                df['dy➗dx'] = pd.to_numeric(df['dy➗dx'], errors='coerce')
            
            self.log(f"数据形状: {df.shape}")
            self.log(f"\nStatus分布:")
            self.log(str(df['status'].value_counts().sort_index()))
            self.log(f"\n井号数量: {df['转换后JH'].nunique()}")
            
            # 2. 特征工程
            self.log("\n2. 特征工程...")
            self.status_var.set("特征工程中...")
            df_with_features = self.predictor.create_features(df)
            self.log(f"特征工程完成，生成 {len(df_with_features.columns)} 个特征")
            
            # 3. 准备数据
            self.log("\n3. 准备数据...")
            self.status_var.set("准备数据中...")
            val_size = self.val_size_var.get()
            
            (X_train, y_train), (X_val, y_val), val_df = \
                self.predictor.prepare_data(df_with_features, val_size=val_size)
            
            self.log(f"训练集样本数: {len(X_train)}")
            self.log(f"验证集样本数: {len(X_val)}")
            
            # 4. 设置参数
            params = {
                'objective': 'multiclass',
                'num_class': 4,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': self.num_leaves_var.get(),
                'learning_rate': self.lr_var.get(),
                'feature_fraction': self.feature_frac_var.get(),
                'bagging_fraction': self.bagging_frac_var.get(),
                'bagging_freq': 5,
                'max_depth': self.max_depth_var.get(),
                'min_child_samples': self.min_child_var.get(),
                'verbose': -1,
                'num_boost_round': self.num_boost_var.get(),
                'early_stopping_rounds': self.early_stop_var.get()
            }
            
            # 5. 训练模型
            self.log("\n4. 开始训练模型...")
            self.status_var.set("训练中...")
            
            history = self.predictor.train(
                X_train, y_train, X_val, y_val, params,
                callback=lambda iter, train_loss, val_loss: self.root.after(0, self.update_plot, iter, train_loss, val_loss)
            )
            
            self.log(f"\n训练完成！")
            self.log(f"最佳迭代: {history['best_iteration']}")
            self.log(f"最佳分数: {history['best_score']}")
            
            # 6. 验证集评估
            self.log("\n5. 验证集评估...")
            self.status_var.set("评估中...")
            
            y_val_pred = self.predictor.predict(X_val)
            y_val_pred_corrected = self.predictor.apply_rules(val_df, y_val_pred)
            val_results = self.predictor.evaluate(y_val, y_val_pred, y_val_pred_corrected)
            
            # 显示结果
            result_text = "="*60 + "\n"
            result_text += "验证集评估结果\n"
            result_text += "="*60 + "\n\n"
            result_text += "[原始预测]\n"
            result_text += f"准确率: {val_results['accuracy']:.4f}\n"
            result_text += f"F1分数 (macro): {val_results['f1_macro']:.4f}\n\n"
            
            if 'accuracy_corrected' in val_results:
                result_text += "[规则修正后]\n"
                result_text += f"准确率: {val_results['accuracy_corrected']:.4f}\n"
                result_text += f"F1分数 (macro): {val_results['f1_macro_corrected']:.4f}\n"
                result_text += f"改进: {val_results['accuracy_corrected']-val_results['accuracy']:+.4f}\n"
            
            self.result_text.insert(tk.END, result_text)
            
            # 7. 保存模型
            self.log("\n6. 保存模型...")
            self.predictor.save_model(params=params)
            
            self.log("\n" + "="*60)
            self.log("训练完成！")
            self.log("="*60)
            
            self.status_var.set("训练完成！")
            self.progress_var.set(100)
            
            messagebox.showinfo("成功", "模型训练完成！")
            
        except Exception as e:
            self.log(f"\n错误: {str(e)}")
            messagebox.showerror("错误", f"训练过程中出现错误:\n{str(e)}")
            self.status_var.set("训练失败")
        
        finally:
            # 重新启用训练按钮
            self.train_button.config(state='normal')
            self.is_training = False
    
    def start_prediction(self):
        """开始预测流程"""
        # 1. 选择模型文件
        model_path = filedialog.askopenfilename(
            title="选择训练好的模型文件",
            filetypes=[("Model files", "*.txt"), ("All files", "*.*")],
            initialdir="Models" if os.path.exists("Models") else os.getcwd()
        )
        
        if not model_path:
            return
        
        # 2. 选择待预测的CSV文件
        data_path = filedialog.askopenfilename(
            title="选择需要预测的CSV文件",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=os.getcwd()
        )
        
        if not data_path:
            return
        
        # 3. 选择输出文件位置
        output_path = filedialog.asksaveasfilename(
            title="保存预测结果",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=os.path.basename(data_path).replace('.csv', '_predictions.csv')
        )
        
        if not output_path:
            return
        
        # 在新线程中运行预测
        prediction_thread = threading.Thread(
            target=self.run_prediction,
            args=(model_path, data_path, output_path)
        )
        prediction_thread.start()
    
    def run_prediction(self, model_path: str, data_path: str, output_path: str):
        """运行预测（在独立线程中）"""
        try:
            self.predict_button.config(state='disabled')
            self.status_var.set("预测中...")
            
            self.log("\n" + "="*60)
            self.log("开始预测")
            self.log("="*60)
            self.log(f"\n模型文件: {model_path}")
            self.log(f"数据文件: {data_path}")
            self.log(f"输出文件: {output_path}")
            
            # 1. 加载数据
            self.log("\n1. 加载数据...")
            df = pd.read_csv(data_path)
            self.log(f"   数据形状: {df.shape}")
            
            # 检查必需的列
            required_cols = ['序号', '转换后JH', 'JX', 'dy➗dx']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"数据文件缺少必需的列: {missing_cols}")
            
            # 转换数据类型
            if 'dy➗dx' in df.columns:
                df['dy➗dx'] = pd.to_numeric(df['dy➗dx'], errors='coerce')
            
            self.log(f"   井号数量: {df['转换后JH'].nunique()}")
            
            # 2. 加载模型
            self.log("\n2. 加载模型...")
            self.predictor.load_model(model_path)
            self.log("   ✓ 模型加载成功")
            
            # 3. 特征工程
            self.log("\n3. 特征工程...")
            df_with_features = self.predictor.create_features(df)
            self.log(f"   ✓ 生成 {len(df_with_features.columns)} 个特征")
            
            # 4. 准备预测数据
            self.log("\n4. 准备预测数据...")
            
            # 使用加载的feature_columns
            if not self.predictor.feature_columns:
                raise ValueError("未找到特征列信息，请使用完整的模型文件（包含_scaler.pkl）")
            
            X_pred = df_with_features[self.predictor.feature_columns].values
            
            # 使用加载的scaler进行标准化
            X_pred = self.predictor.scaler.transform(X_pred)
            
            self.log(f"   ✓ 准备了 {len(X_pred)} 个样本，使用 {len(self.predictor.feature_columns)} 个特征")
            
            # 5. 预测
            self.log("\n5. 执行预测...")
            y_pred = self.predictor.predict(X_pred)
            self.log(f"   ✓ 预测完成")
            
            # 6. 应用规则修正
            self.log("\n6. 应用后处理规则...")
            y_pred_corrected = self.predictor.apply_rules(df_with_features, y_pred)
            
            # 7. 保存结果
            self.log("\n7. 保存预测结果...")
            result_df = df[['序号', '转换后JH', 'JX', 'dy➗dx']].copy()
            result_df['status_predict'] = y_pred_corrected
            
            result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            self.log(f"   ✓ 结果已保存到: {output_path}")
            
            # 显示预测结果统计
            self.log("\n预测结果统计:")
            status_counts = pd.Series(y_pred_corrected).value_counts().sort_index()
            status_names = {0: '直井段', 1: '造斜段', 2: '稳斜段', 3: '降斜段'}
            for status, count in status_counts.items():
                self.log(f"   {status_names.get(status, status)}: {count} ({count/len(y_pred_corrected)*100:.1f}%)")
            
            self.log("\n" + "="*60)
            self.log("预测完成！")
            self.log("="*60)
            
            self.status_var.set("预测完成！")
            messagebox.showinfo("成功", f"预测完成！\n结果已保存到:\n{output_path}")
            
        except Exception as e:
            self.log(f"\n错误: {str(e)}")
            messagebox.showerror("错误", f"预测过程中出现错误:\n{str(e)}")
            self.status_var.set("预测失败")
            import traceback
            traceback.print_exc()
        
        finally:
            self.predict_button.config(state='normal')


def main():
    """主函数"""
    root = tk.Tk()
    app = TrainingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

