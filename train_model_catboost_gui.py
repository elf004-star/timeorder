"""
井段状态预测模型训练脚本 - CatBoost GUI版本
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import catboost as cb
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
import os
import json
import pickle
from datetime import datetime

# Set Arial font
from matplotlib import rcParams
rcParams['font.family'] = 'Arial'
rcParams['font.sans-serif'] = ['Arial']
rcParams['axes.unicode_minus'] = False


class WellStatusPredictor:
    """井段状态预测器 (CatBoost)"""
    
    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()
        self.feature_columns = []
    
    def load_model(self, model_path: str):
        """加载已训练的模型及相关配置"""
        
        # 加载模型
        self.model = cb.CatBoostClassifier()
        self.model.load_model(model_path)
        print(f"模型已加载: {model_path}")
        
        # 尝试加载scaler和feature_columns
        scaler_path = model_path.replace('.cbm', '_scaler.pkl')
        if not os.path.exists(scaler_path):
             # 兼容旧命名习惯
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
            group['dy_dx'] = pd.to_numeric(group['Dogleg Severity'], errors='coerce').fillna(0)
            
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
        exclude_cols = ['序号', '转换后JH', 'status', 'Dogleg Severity']
        self.feature_columns = [col for col in train_df.columns if col not in exclude_cols]
        
        X_train = train_df[self.feature_columns].values
        y_train = train_df['status'].values
        
        X_val = val_df[self.feature_columns].values
        y_val = val_df['status'].values
        
        # 标准化特征
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        
        return (X_train, y_train), (X_val, y_val), train_df, val_df
    
    def save_results_by_well(self, df: pd.DataFrame, origin_predictions: np.ndarray, corrected_predictions: np.ndarray, output_dir: str):
        """保存预测结果，按井号分文件"""
        import os
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Add prediction to a copy of df
        result_df = df.copy()
        result_df['origin_status_predict'] = origin_predictions
        result_df['status_predict'] = corrected_predictions
        
        # Keep only required columns
        required_cols = ['序号', '转换后JH', 'JX', 'Dogleg Severity', 'status', 'origin_status_predict', 'status_predict']
        # Check if columns exist
        available_cols = [col for col in required_cols if col in result_df.columns]
        
        result_df = result_df[available_cols]
        
        # Save by well
        count = 0
        for well_name, group in result_df.groupby('转换后JH'):
            # Clean filename
            safe_name = str(well_name).replace('/', '_').replace('\\', '_')
            file_path = os.path.join(output_dir, f"{safe_name}.csv")
            group.to_csv(file_path, index=False, encoding='utf-8-sig')
            count += 1
            
        print(f"已保存 {count} 个井的文件到 {output_dir}")

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray, params: Dict,
              callback=None) -> Dict:
        """
        训练CatBoost模型
        """
        print("\n开始训练模型...")
        
        # 准备CatBoost参数
        cb_params = {
            'iterations': params.get('iterations', 1000),
            'learning_rate': params.get('learning_rate', 0.05),
            'depth': params.get('depth', 6),
            'l2_leaf_reg': params.get('l2_leaf_reg', 3),
            'min_data_in_leaf': params.get('min_data_in_leaf', 1),
            'auto_class_weights': 'Balanced',  # 自动处理类别不平衡
            'loss_function': 'MultiClass',
            'eval_metric': 'MultiClass',
            'random_seed': 42,
            'early_stopping_rounds': params.get('early_stopping_rounds', 50),
            'verbose': 100,  # 每100轮打印一次
            'allow_writing_files': False
        }
        
        self.model = cb.CatBoostClassifier(**cb_params)
        
        # 训练模型
        # 注意：CatBoost的fit是阻塞的，这里无法像LightGBM那样简单地通过callback进行实时绘图更新
        # 我们将在训练完成后统一绘制曲线
        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True
        )
        
        print("模型训练完成")
        
        # 获取训练历史
        evals_result = self.model.get_evals_result()
        # CatBoost evals_result 结构: {'learn': {'MultiClass': [...]}, 'validation': {'MultiClass': [...]}}
        
        train_losses = evals_result['learn']['MultiClass']
        val_losses = evals_result['validation']['MultiClass']
        
        # 如果有回调，我们在最后调用一次以更新图表
        if callback:
            # 模拟逐步更新的效果可能比较卡，直接传完整数据
            # 但为了保持接口一致，我们可以在这里不做任何事，图表更新在GUI层处理
            pass

        return {
            'best_iteration': self.model.get_best_iteration(),
            'best_score': self.model.get_best_score()['validation']['MultiClass'],
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        y_pred_proba = self.model.predict_proba(X)
        return np.argmax(y_pred_proba, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """获取预测概率"""
        return self.model.predict_proba(X)
    
    def apply_rules(self, df: pd.DataFrame, predictions: np.ndarray, prediction_proba: np.ndarray = None) -> np.ndarray:
        """
        应用后处理规则：
        1. 当一口井最大状态是2的时候，即使预测的是3，也当作2处理 (仅当有真实标签时)
        2. 确保状态转换的单向性 (0 -> 1 -> 2 -> 3)，使用最小修改算法
        3. 如果修改次数相同，优先修改置信度低的地方
        """
        print("\n应用后处理规则...")
        
        # 创建副本以避免修改原始数据
        temp_df = df.copy()
        
        # 必须确保temp_df和predictions是对齐的
        # 添加一个0..N的索引列，用于后续将结果放回数组的正确位置
        temp_df['__row_idx__'] = range(len(temp_df))
        temp_df['__pred__'] = predictions
        
        # 如果没有提供概率，使用默认置信度（所有位置相等）
        if prediction_proba is None:
            # 使用预测值作为置信度的简单代理（实际应该用概率）
            confidence = np.ones(len(predictions))
        else:
            # 使用预测类别的概率作为置信度
            confidence = np.max(prediction_proba, axis=1)
        
        temp_df['__confidence__'] = confidence
        
        corrected_predictions = predictions.copy()
        
        # 按井号分组处理
        for well_name, group in temp_df.groupby('转换后JH'):
            # 获取该组在原始数组中的索引位置
            indices = group['__row_idx__'].values
            preds = group['__pred__'].values.copy()
            confidences = group['__confidence__'].values.copy()
            
            # 规则1: 如果真实最大状态为2，则将预测的3改为2
            # 只有当DataFrame包含'status'列时才能应用此规则
            if 'status' in group.columns:
                max_status = group['status'].max()
                if max_status == 2:
                    preds[preds == 3] = 2
            
            # 规则2: 使用最小修改算法确保单向性 0 -> 1 -> 2 -> 3
            corrected = self._minimize_modifications(preds, confidences)
            
            # 将修正后的结果放回对应位置
            corrected_predictions[indices] = corrected
        
        # 统计修正的数量
        n_corrected = np.sum(predictions != corrected_predictions)
        print(f"修正了 {n_corrected} 个预测 ({n_corrected/len(predictions)*100:.2f}%)")
        
        return corrected_predictions
    
    def _minimize_modifications(self, preds: np.ndarray, confidences: np.ndarray) -> np.ndarray:
        """
        找到最小修改方案，使序列单调递增（0 -> 1 -> 2 -> 3）
        如果修改次数相同，优先修改置信度低的地方
        
        使用动态规划算法
        """
        n = len(preds)
        if n == 0:
            return preds
        
        # 状态：dp[i][s] = 到位置i，状态为s的最小修改代价
        # s可以是0,1,2,3
        # 代价 = 修改次数 + (1 - 置信度) * 0.1（置信度低的优先修改）
        
        dp = np.full((n, 4), np.inf)
        parent = np.full((n, 4), -1, dtype=int)
        
        # 初始化：第一个位置
        for s in range(4):
            if preds[0] == s:
                cost = 0  # 不需要修改
            else:
                cost = 1 + (1 - confidences[0]) * 0.1  # 修改代价
            dp[0][s] = cost
        
        # 动态规划
        for i in range(1, n):
            for s in range(4):
                # 当前状态必须是s，且必须 >= 前一个位置的状态
                min_cost = np.inf
                best_prev = -1
                
                for prev_s in range(s + 1):  # 前一个状态可以是0到s
                    prev_cost = dp[i-1][prev_s]
                    if prev_cost == np.inf:
                        continue
                    
                    # 计算修改代价
                    if preds[i] == s:
                        modify_cost = 0
                    else:
                        modify_cost = 1 + (1 - confidences[i]) * 0.1
                    
                    total_cost = prev_cost + modify_cost
                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_prev = prev_s
                
                dp[i][s] = min_cost
                parent[i][s] = best_prev
        
        # 回溯找到最优路径
        best_final_state = np.argmin(dp[n-1])
        result = np.zeros(n, dtype=int)
        result[n-1] = best_final_state
        
        for i in range(n-2, -1, -1):
            result[i] = parent[i+1][result[i+1]]
        
        return result
    
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
    
    def save_model(self, filename: str = "well_status_catboost_model.cbm", params: Dict = None):
        """保存模型到Models文件夹"""
        
        # 创建Models文件夹
        models_dir = "Models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            print(f"创建文件夹: {models_dir}")
        
        # 确保文件名后缀正确
        if not filename.endswith('.cbm'):
            filename = os.path.splitext(filename)[0] + '.cbm'
            
        # 保存模型到Models文件夹
        model_path = os.path.join(models_dir, filename)
        self.model.save_model(model_path)
        print(f"\n模型已保存到: {model_path}")
        
        # 保存scaler和feature_columns
        scaler_filename = filename.replace('.cbm', '_scaler.pkl')
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
                'best_iteration': self.model.get_best_iteration(),
                'best_score': self.model.get_best_score()['validation']['MultiClass'],
                'parameters': params
            }
            
            params_filename = filename.replace('.cbm', '_params.json')
            params_path = os.path.join(models_dir, params_filename)
            
            with open(params_path, 'w', encoding='utf-8') as f:
                json.dump(params_info, f, indent=4, ensure_ascii=False)
            
            print(f"参数已保存到: {params_path}")


class TrainingGUI:
    """训练界面GUI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("井段状态预测模型训练系统 (CatBoost)")
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
        ttk.Label(control_frame, text="CatBoost 超参数", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=5)
        row += 1
        
        # Learning Rate
        ttk.Label(control_frame, text="学习率:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.lr_var = tk.DoubleVar(value=0.05)
        ttk.Entry(control_frame, textvariable=self.lr_var, width=15).grid(row=row, column=1, sticky=tk.W, pady=5, padx=5)
        row += 1
        
        # Depth
        ttk.Label(control_frame, text="树深度 (Depth):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.depth_var = tk.IntVar(value=6)
        ttk.Entry(control_frame, textvariable=self.depth_var, width=15).grid(row=row, column=1, sticky=tk.W, pady=5, padx=5)
        row += 1
        
        # Iterations
        ttk.Label(control_frame, text="迭代次数:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.iterations_var = tk.IntVar(value=1000)
        ttk.Entry(control_frame, textvariable=self.iterations_var, width=15).grid(row=row, column=1, sticky=tk.W, pady=5, padx=5)
        row += 1
        
        # Early Stopping
        ttk.Label(control_frame, text="早停轮数:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.early_stop_var = tk.IntVar(value=50)
        ttk.Entry(control_frame, textvariable=self.early_stop_var, width=15).grid(row=row, column=1, sticky=tk.W, pady=5, padx=5)
        row += 1
        
        # L2 Leaf Reg
        ttk.Label(control_frame, text="L2正则化:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.l2_reg_var = tk.DoubleVar(value=3.0)
        ttk.Entry(control_frame, textvariable=self.l2_reg_var, width=15).grid(row=row, column=1, sticky=tk.W, pady=5, padx=5)
        row += 1
        
        # Min Data In Leaf
        ttk.Label(control_frame, text="叶子最小样本:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.min_data_var = tk.IntVar(value=1)
        ttk.Entry(control_frame, textvariable=self.min_data_var, width=15).grid(row=row, column=1, sticky=tk.W, pady=5, padx=5)
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
        self.ax.set_ylabel('Loss (MultiClass)')
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
    
    def update_plot(self, train_losses, val_losses):
        """更新训练曲线"""
        self.ax.clear()
        self.ax.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
        self.ax.plot(val_losses, label='Validation Loss', color='red', linewidth=2)
        self.ax.set_xlabel('Iteration')
        self.ax.set_ylabel('Loss (MultiClass)')
        self.ax.set_title('Training and Validation Loss')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        self.canvas.draw()
    
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
        self.progress_var.set(0)
        
        # 在新线程中运行训练
        self.training_thread = threading.Thread(target=self.train_model)
        self.training_thread.start()
    
    def train_model(self):
        """训练模型（在独立线程中运行）"""
        try:
            # 1. 加载数据
            self.log("="*60)
            self.log("井段状态预测模型训练 (CatBoost)")
            self.log("="*60)
            self.log("\n1. 加载数据...")
            
            df = pd.read_csv(self.file_var.get())
            
            # 转换Dogleg Severity为数值类型
            if 'Dogleg Severity' in df.columns:
                df['Dogleg Severity'] = pd.to_numeric(df['Dogleg Severity'], errors='coerce')
            
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
            
            (X_train, y_train), (X_val, y_val), train_df, val_df = \
                self.predictor.prepare_data(df_with_features, val_size=val_size)
            
            self.log(f"训练集样本数: {len(X_train)}")
            self.log(f"验证集样本数: {len(X_val)}")
            
            # 4. 设置参数
            params = {
                'iterations': self.iterations_var.get(),
                'learning_rate': self.lr_var.get(),
                'depth': self.depth_var.get(),
                'l2_leaf_reg': self.l2_reg_var.get(),
                'min_data_in_leaf': self.min_data_var.get(),
                'early_stopping_rounds': self.early_stop_var.get()
            }
            
            # 5. 训练模型
            self.log("\n4. 开始训练模型...")
            self.status_var.set("训练中...")
            
            # 由于CatBoost的fit是阻塞的，我们不传入实时回调，而是训练完后绘制
            history = self.predictor.train(
                X_train, y_train, X_val, y_val, params,
                callback=None
            )
            
            # 更新图表
            self.root.after(0, self.update_plot, history['train_losses'], history['val_losses'])
            
            self.log(f"\n训练完成！")
            self.log(f"最佳迭代: {history['best_iteration']}")
            self.log(f"最佳分数: {history['best_score']:.6f}")
            
            # 6. 验证集评估
            self.log("\n5. 验证集评估...")
            self.status_var.set("评估中...")
            
            y_val_pred_proba = self.predictor.predict_proba(X_val)
            y_val_pred = np.argmax(y_val_pred_proba, axis=1)
            y_val_pred_corrected = self.predictor.apply_rules(val_df, y_val_pred, y_val_pred_proba)
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
            
            # 6. 保存中间结果
            self.log("\n6. 保存中间结果...")
            self.status_var.set("保存中间结果...")
            
            # Predict on training set
            y_train_pred_proba = self.predictor.predict_proba(X_train)
            y_train_pred = np.argmax(y_train_pred_proba, axis=1)
            y_train_pred_corrected = self.predictor.apply_rules(train_df, y_train_pred, y_train_pred_proba)
            
            # Define memory directory
            base_dir = os.path.dirname(self.file_var.get()) if self.file_var.get() else os.getcwd()
            memory_dir = os.path.join(base_dir, "memory_catboost")
            
            self.predictor.save_results_by_well(train_df, y_train_pred, y_train_pred_corrected, os.path.join(memory_dir, "train"))
            self.predictor.save_results_by_well(val_df, y_val_pred, y_val_pred_corrected, os.path.join(memory_dir, "test"))
            
            self.log(f"中间结果已保存到 {memory_dir}")

            # 7. 保存模型
            self.log("\n7. 保存模型...")
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
            import traceback
            traceback.print_exc()
        
        finally:
            # 重新启用训练按钮
            self.train_button.config(state='normal')
            self.is_training = False
    
    def start_prediction(self):
        """开始预测流程"""
        # 1. 选择模型文件
        model_path = filedialog.askopenfilename(
            title="选择训练好的模型文件",
            filetypes=[("CatBoost Model", "*.cbm"), ("All files", "*.*")],
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
            required_cols = ['序号', '转换后JH', 'JX', 'Dogleg Severity']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"数据文件缺少必需的列: {missing_cols}")
            
            # 转换数据类型
            if 'Dogleg Severity' in df.columns:
                df['Dogleg Severity'] = pd.to_numeric(df['Dogleg Severity'], errors='coerce')
            
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
            y_pred_proba = self.predictor.predict_proba(X_pred)
            y_pred = np.argmax(y_pred_proba, axis=1)
            self.log(f"   ✓ 预测完成")
            
            # 6. 应用规则修正
            self.log("\n6. 应用后处理规则...")
            y_pred_corrected = self.predictor.apply_rules(df_with_features, y_pred, y_pred_proba)
            
            # 7. 保存结果
            self.log("\n7. 保存预测结果...")
            result_df = df[['序号', '转换后JH', 'JX', 'Dogleg Severity']].copy()
            result_df['origin_status_predict'] = y_pred
            result_df['status_predict'] = y_pred_corrected
            
            result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            self.log(f"   ✓ 结果已保存到: {output_path}")
            
            # 7.5 保存分井结果
            self.log("\n7.5 保存分井结果...")
            data_dir = os.path.dirname(data_path) if data_path else os.getcwd()
            predict_dir = os.path.join(data_dir, "predict_catboost")
            self.predictor.save_results_by_well(df, y_pred, y_pred_corrected, predict_dir)
            self.log(f"   ✓ 分井结果已保存到: {predict_dir}")
            
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
    try:
        import catboost
    except ImportError:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("错误", "未检测到 catboost 库。\n请运行 pip install catboost 安装。")
        return

    root = tk.Tk()
    app = TrainingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
