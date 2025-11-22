"""
井段状态预测模型训练脚本 - GUI版本 (XGBoost)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')
from scipy.signal import savgol_filter
from scipy import stats

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
    """井段状态预测器 (XGBoost版)"""
    
    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()
        self.feature_columns = []
        # 默认预处理参数
        self.preprocessing_params = {
            'method': 'none',      # none, rolling, savgol
            'window_size': 5,
            'poly_order': 2
        }
    
    def load_model(self, model_path: str):
        """加载已训练的模型及相关配置"""
        import pickle
        
        # 自动纠正路径：如果用户选择了参数文件(_params.json)，尝试切换到模型文件(.json)
        if model_path.endswith('_params.json'):
            adjusted_path = model_path.replace('_params.json', '.json')
            if os.path.exists(adjusted_path):
                print(f"提示: 检测到选择了参数文件，自动切换到模型文件: {adjusted_path}")
                model_path = adjusted_path

        # 加载模型
        self.model = xgb.Booster()
        try:
            self.model.load_model(model_path)
        except Exception as e:
            if 'Invalid cast' in str(e):
                raise ValueError(f"模型文件格式错误: {model_path}\n请确保选择了正确的模型文件(.json)，而不是参数文件(_params.json)。")
            raise e
            
        print(f"模型已加载: {model_path}")
        
        # 尝试加载scaler和feature_columns
        # 兼容 .json, .model, .txt 等后缀
        base_path = os.path.splitext(model_path)[0]
        scaler_path = base_path + '_scaler.pkl'
        
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                data = pickle.load(f)
                self.scaler = data['scaler']
                self.feature_columns = data['feature_columns']
                # 加载预处理参数，如果不存在则使用默认值
                self.preprocessing_params = data.get('preprocessing_params', {
                    'method': 'none',
                    'window_size': 5,
                    'poly_order': 2
                })
            print(f"配置已加载: {scaler_path}")
            print(f"预处理参数: {self.preprocessing_params}")
        else:
            print(f"警告: 未找到scaler文件 {scaler_path}，预测可能不准确")
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        特征工程：为每口井创建时序特征 (共28个推荐特征)
        """
        print("开始特征工程...")
        print(f"使用预处理参数: {self.preprocessing_params}")
        
        # 按井号分组处理
        features_list = []
        
        for well_name, group in df.groupby('转换后JH'):
            group = group.copy().sort_values('序号').reset_index(drop=True)
            
            # 确保基础列存在且为数值
            group['JX'] = pd.to_numeric(group['JX'], errors='coerce').fillna(0)
            if 'Dogleg Severity' in group.columns:
                group['dogl'] = pd.to_numeric(group['Dogleg Severity'], errors='coerce').fillna(0)
            else:
                group['dogl'] = 0
            
            # ==========================================
            # 1. 数据去噪 (Smoothing)
            # ==========================================
            raw_jx = group['JX'].values
            smoothed_jx = raw_jx.copy()
            
            method = self.preprocessing_params.get('method', 'none')
            window = int(self.preprocessing_params.get('window_size', 5))
            poly = int(self.preprocessing_params.get('poly_order', 2))
            
            if len(group) >= window:
                if method == 'rolling':
                    # 移动平均 (Center=True保持相位)
                    smoothed_jx = pd.Series(raw_jx).rolling(window=window, center=True, min_periods=1).mean().values
                elif method == 'savgol':
                    # Savitzky-Golay 滤波器
                    try:
                        if window % 2 == 0: window += 1 # SG要求窗口为奇数
                        if window > len(group): window = len(group) if len(group) % 2 != 0 else len(group) - 1
                        if window > poly:
                            smoothed_jx = savgol_filter(raw_jx, window, poly)
                    except Exception as e:
                        print(f"SG Filter error for well {well_name}: {e}")
            
            # 使用平滑后的数据进行特征计算
            # 注意：后续特征计算基于 smoothed_jx，但为了保留原始信息，可以保留raw列
            group['JX_smooth'] = smoothed_jx
            # 用平滑后的值替换原值进行计算（或者新建一列，这里根据需求主要用平滑后的分析趋势）
            # 题目要求：在提取特征前，必须进行两步操作。所以我们基于 JX_smooth 计算特征
            s_series = pd.Series(smoothed_jx) 
            
            # 辅助函数：计算滚动斜率 (提前定义)
            def calc_slope(x):
                if len(x) < 2: return 0
                return np.polyfit(np.arange(len(x)), x, 1)[0]
            
            # ==========================================
            # 1.5 构建 dogl_s 及其特征
            # ==========================================
            # 使用平滑后的JX判断趋势
            jx_next = s_series.shift(-1)
            
            # 判断符号: 下一点 >= 当前点 => 1 (增加/不变), 下一点 < 当前点 => -1 (减少)
            # 使用np.where处理
            signs = np.where(jx_next < s_series, -1, 1)
            
            group['dogl_s'] = group['dogl'] * signs
            
            # 最后一个点沿用上一个点的dogl_s
            if len(group) > 1:
                group.loc[group.index[-1], 'dogl_s'] = group.loc[group.index[-2], 'dogl_s']
            
            # dogl_s 的 6个时序特征
            # 1. dogl_s_mean_5
            group['dogl_s_mean_5'] = group['dogl_s'].rolling(window=5, min_periods=1).mean()
            # 2. dogl_s_std_5
            group['dogl_s_std_5'] = group['dogl_s'].rolling(window=5, min_periods=1).std().fillna(0)
            # 3. dogl_s_trend_5 (Slope)
            group['dogl_s_trend_5'] = group['dogl_s'].rolling(window=5, min_periods=2).apply(calc_slope).fillna(0)
            # 4. dogl_s_diff_1
            group['dogl_s_diff_1'] = group['dogl_s'].diff().fillna(0)
            # 5. dogl_s_cumsum
            group['dogl_s_cumsum'] = group['dogl_s'].cumsum()
            # 6. dogl_s_lag_1
            group['dogl_s_lag_1'] = group['dogl_s'].shift(1).fillna(0)
            
            # ==========================================
            # 2. 组内归一化 (Group-wise Normalization)
            # ==========================================
            g_min = s_series.min()
            g_max = s_series.max()
            g_range = g_max - g_min if (g_max - g_min) > 1e-6 else 1.0
            
            # 第一类：位置与状态特征 (5个)
            # 1. norm_value
            group['norm_value'] = (s_series - g_min) / g_range
            
            # 2. rel_pos
            group['rel_pos'] = (group.index + 1) / len(group)
            
            # 3. dist_to_max
            group['dist_to_max'] = g_max - s_series
            
            # 4. is_max_region (是否大于组内最大值的95%)
            group['is_max_region'] = (s_series > (g_min + 0.95 * g_range)).astype(int)
            
            # 5. accum_ratio (当前累计和占总和的比例)
            group['accum_ratio'] = s_series.cumsum() / (s_series.sum() + 1e-6)
            
            # ==========================================
            # 第二类：速度与趋势特征 (一阶导数相关) (8个)
            # ==========================================
            
            # 6. diff_1
            group['diff_1'] = s_series.diff().fillna(0)
            
            # 7. diff_3
            group['diff_3'] = s_series.diff(3).fillna(0)
            
            # 辅助函数：计算滚动斜率
            def calc_slope(x):
                if len(x) < 2: return 0
                return np.polyfit(np.arange(len(x)), x, 1)[0]
            
            # 8. trend_slope_5
            group['trend_slope_5'] = s_series.rolling(5, min_periods=2).apply(calc_slope).fillna(0)
            
            # 9. trend_slope_10
            group['trend_slope_10'] = s_series.rolling(10, min_periods=2).apply(calc_slope).fillna(0)
            
            # 10. ema_divergence (当前值 - EMA)
            ema = s_series.ewm(span=10, adjust=False).mean()
            group['ema_divergence'] = s_series - ema
            
            # 11. pct_change
            group['pct_change'] = s_series.pct_change().fillna(0).replace([np.inf, -np.inf], 0)
            
            # 12 & 13. consecutive_up/down (连续上升/下降)
            # 这是一个比较耗时的操作，使用向量化方法近似
            diff = group['diff_1']
            
            # 识别上升(+1)和下降(-1)
            direction = np.sign(diff)
            
            # 计算连续上升
            # 利用groupby和小技巧
            up_groups = (direction <= 0).cumsum()
            group['consecutive_up'] = group.groupby(up_groups).cumcount()
            # 修正：如果当前不是上升，置为0
            group.loc[direction <= 0, 'consecutive_up'] = 0
            
            # 计算连续下降
            down_groups = (direction >= 0).cumsum()
            group['consecutive_down'] = group.groupby(down_groups).cumcount()
            group.loc[direction >= 0, 'consecutive_down'] = 0
            
            # ==========================================
            # 第三类：拐点与加速度特征 (二阶导数相关) (7个)
            # ==========================================
            
            # 14. diff_2
            group['diff_2'] = group['diff_1'].diff().fillna(0)
            
            # 15. slope_change
            group['slope_change'] = group['trend_slope_5'].diff().fillna(0)
            
            # 16. curvature (局部曲率 - 简化版：三点计算)
            # k = |x''| / (1 + x'^2)^(3/2)
            # 这里x是index，y是值。x' = diff_1, x'' = diff_2
            # 假设dx=1
            group['curvature'] = group['diff_2'].abs() / ((1 + group['diff_1']**2)**1.5)
            group['curvature'] = group['curvature'].fillna(0)
            
            # 17. peak_accel_loc (窗口内是否是二阶导数的最大值点 - 窗口5)
            roll_max_acc = group['diff_2'].abs().rolling(5, center=True, min_periods=1).max()
            group['peak_accel_loc'] = (group['diff_2'].abs() == roll_max_acc).astype(int)
            
            # 18. std_5
            group['std_5'] = s_series.rolling(5, min_periods=1).std().fillna(0)
            
            # 19. std_10
            group['std_10'] = s_series.rolling(10, min_periods=1).std().fillna(0)
            
            # 20. z_score_local (过去10个点)
            roll_mean_10 = s_series.rolling(10, min_periods=1).mean()
            roll_std_10 = s_series.rolling(10, min_periods=1).std().replace(0, 1) # 避免除0
            group['z_score_local'] = (s_series - roll_mean_10) / roll_std_10
            group['z_score_local'] = group['z_score_local'].fillna(0)
            
            # ==========================================
            # 第四类：双向上下文特征 (Look-ahead) (8个)
            # ==========================================
            
            # 21. lead_diff_1 (未来1个点 - 当前) = (t+1) - t
            group['lead_diff_1'] = s_series.shift(-1) - s_series
            
            # 22. lead_diff_3
            group['lead_diff_3'] = s_series.shift(-3) - s_series
            
            # 23. lead_slope_5 (未来5个点的斜率)
            # 将序列反转，计算past slope，再反转回来，并shift
            rev_s = s_series.iloc[::-1]
            rev_slope = rev_s.rolling(5, min_periods=2).apply(calc_slope).fillna(0)
            # 反转回来是对应"未来"的，但索引是对齐的吗？
            # rolling是取"过去"5个。
            # 对t而言，future slope是 t, t+1, ..., t+4 的斜率
            # 可以用shift后的数据算
            # 简化：对shift(-5)的rolling(5)不对，应该是反向rolling
            # 这里用一个近似：shift(-1)后计算rolling(5)，但rolling是向后的
            # 正确做法：使用indexer或shift
            # 简单做法：shift(-4)然后取rolling(5)的slope? No.
            # 最好的办法：利用shift构造matrix然后polyfit，但太慢。
            # 既然已有 trend_slope_5 (looking back)，我们把整个序列反转计算trend_slope_5，再反转回来，就是looking forward
            
            # 反转计算
            slope_rev = rev_s.rolling(5, min_periods=2).apply(calc_slope).fillna(0)
            # 反转回来
            slope_forward = slope_rev.iloc[::-1].values
            # 注意：反转后的rolling代表的是"未来"向"现在"看。
            # 比如反转后第0个点是原最后一点。rolling(5)用了原最后5点。
            # 所以反转回来的第i点，包含了 i, i+1, i+2... 
            # 但方向是反的。斜率符号要取反？
            # 如果原序列是递增，反转后是递减，斜率为负。所以要取反。
            group['lead_slope_5'] = -slope_forward
            
            # 24. center_diff ( (t+1) - (t-1) ) / 2
            group['center_diff'] = (s_series.shift(-1) - s_series.shift(1)) / 2
            
            # 25. pre_post_ratio (未来5点均值 / 过去5点均值)
            # 过去5点均值
            past_mean = s_series.rolling(5, min_periods=1).mean()
            # 未来5点均值：反转计算rolling mean再反转
            future_mean = s_series.iloc[::-1].rolling(5, min_periods=1).mean().iloc[::-1]
            
            group['pre_post_ratio'] = future_mean / (past_mean + 1e-6) # 避免除0
            
            # 26. lag_rolling_mean_5
            group['lag_rolling_mean_5'] = past_mean
            
            # 27. lead_rolling_mean_5
            group['lead_rolling_mean_5'] = future_mean
            
            # 28. cross_region (lead > lag)
            group['cross_region'] = (future_mean > (past_mean * 1.02)).astype(int) # 稍微加点阈值
            
            # 填充由于shift产生的NaN
            group = group.fillna(0)
            
            features_list.append(group)
        
        result_df = pd.concat(features_list, ignore_index=True)
        print(f"特征工程完成，生成 {len(result_df.columns)} 个列")
        
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
        训练XGBoost模型
        """
        print("\n开始训练模型(XGBoost)...")
        
        # 计算样本权重以处理类别不平衡
        print("计算样本权重以处理类别不平衡...")
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
        
        # 创建数据矩阵
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # 训练历史记录
        train_losses = []
        val_losses = []
        
        # 自定义回调函数，用于记录和更新GUI
        class GUICallback(xgb.callback.TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                # evals_log 结构: {'train': {'mlogloss': [...]}, 'valid': {'mlogloss': [...]}}
                try:
                    # 获取最新的loss值
                    # 注意: 不同版本的xgboost evals_log结构可能略有不同，这里假设标准结构
                    train_loss = evals_log['train']['mlogloss'][-1]
                    val_loss = evals_log['valid']['mlogloss'][-1]
                    
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    
                    if callback:
                        callback(epoch, train_loss, val_loss)
                except Exception as e:
                    # 忽略回调中的错误，以免中断训练
                    pass
                return False

        # 训练模型
        evals = [(dtrain, 'train'), (dval, 'valid')]
        
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=params.get('num_boost_round', 500),
            evals=evals,
            callbacks=[GUICallback()],
            early_stopping_rounds=params.get('early_stopping_rounds', 50),
            verbose_eval=10
        )
        
        print("模型训练完成")
        
        return {
            'best_iteration': self.model.best_iteration,
            'best_score': self.model.best_score,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        dtest = xgb.DMatrix(X)
        # XGBoost predict对于多分类默认返回 (N, n_classes) 的概率矩阵 (使用multi:softprob)
        y_pred_proba = self.model.predict(dtest)
        return np.argmax(y_pred_proba, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """获取预测概率"""
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
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
    
    def save_model(self, filename: str = "well_status_xgboost.json", params: Dict = None):
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
        # 强制使用json后缀如果用户没有指定
        if not filename.endswith('.json'):
            filename = os.path.splitext(filename)[0] + '.json'
            
        model_path = os.path.join(models_dir, filename)
        self.model.save_model(model_path)
        print(f"\n模型已保存到: {model_path}")
        
        # 保存scaler和feature_columns
        scaler_filename = filename.replace('.json', '_scaler.pkl')
        scaler_path = os.path.join(models_dir, scaler_filename)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'preprocessing_params': self.preprocessing_params
            }, f)
        print(f"Scaler及配置已保存到: {scaler_path}")
        
        # 保存训练参数
        if params is not None:
            params_info = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_file': filename,
                'best_iteration': self.model.best_iteration,
                'best_score': self.model.best_score,
                'parameters': params
            }
            
            params_filename = filename.replace('.json', '_params.json')
            params_path = os.path.join(models_dir, params_filename)
            
            with open(params_path, 'w', encoding='utf-8') as f:
                json.dump(params_info, f, indent=4, ensure_ascii=False)
            
            print(f"参数已保存到: {params_path}")

    def save_feature_importance(self, memory_dir: str):
        """保存特征重要性到CSV和生成图表"""
        if self.model is None or not self.feature_columns:
            print("警告: 模型未训练或特征列未设置，无法获取特征重要性")
            return
        
        # 获取特征重要性 (XGBoost)
        importance_dict = self.model.get_score(importance_type='gain')
        
        features = []
        importances = []
        
        for i, col_name in enumerate(self.feature_columns):
            # XGBoost default feature name is f{i}
            f_key = f'f{i}'
            score = importance_dict.get(f_key, 0)
            features.append(col_name)
            importances.append(score)
            
        # 创建DataFrame
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # 创建features文件夹
        features_dir = os.path.join(memory_dir, "features")
        if not os.path.exists(features_dir):
            os.makedirs(features_dir)
        
        # 保存CSV
        csv_path = os.path.join(features_dir, "feature_importance.csv")
        importance_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"特征重要性CSV已保存到: {csv_path}")
        
        # Generate chart - top 10 features
        plt.figure(figsize=(10, 6))
        top10 = importance_df.head(10)
        plt.barh(range(len(top10)), top10['importance'].values, color='steelblue')
        plt.yticks(range(len(top10)), top10['feature'].values, fontsize=11)
        plt.xticks(fontsize=11)
        plt.xlabel('Feature Importance (Gain)', fontsize=14)
        plt.ylabel('Feature Name', fontsize=14)
        plt.title('Top 10 Features by Importance', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()  # Most important at the top
        plt.tight_layout()
        top10_path = os.path.join(features_dir, "feature_importance_top10.png")
        plt.savefig(top10_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature importance chart (top 10) saved to: {top10_path}")

        # Generate chart - top 20 features
        plt.figure(figsize=(10, 8))
        top20 = importance_df.head(20)
        plt.barh(range(len(top20)), top20['importance'].values, color='steelblue')
        plt.yticks(range(len(top20)), top20['feature'].values)
        plt.xlabel('Feature Importance (Gain)', fontsize=14)
        plt.ylabel('Feature Name', fontsize=14)
        plt.title('Top 20 Features', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()  # Most important at the top
        plt.tight_layout()
        top20_path = os.path.join(features_dir, "feature_importance_top20.png")
        plt.savefig(top20_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature Importance Chart (Top 20) saved to: {top20_path}")
        
        # Generate chart - all features
        plt.figure(figsize=(12, max(8, len(importance_df) * 0.3)))
        plt.barh(range(len(importance_df)), importance_df['importance'].values, color='steelblue')
        plt.yticks(range(len(importance_df)), importance_df['feature'].values, fontsize=8)
        plt.xlabel('Feature Importance (Gain)', fontsize=14)
        plt.ylabel('Feature Name', fontsize=14)
        plt.title('All Features', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()  # Most important at the top
        plt.tight_layout()
        all_path = os.path.join(features_dir, "feature_importance_all.png")
        plt.savefig(all_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature Importance Chart (All) saved to: {all_path}")

class TrainingGUI:
    """训练界面GUI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("井段状态预测模型训练系统 (XGBoost)")
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
        
        val_scale = ttk.Scale(
            control_frame, 
            from_=0.1, 
            to=0.9, 
            variable=self.val_size_var, 
            orient=tk.HORIZONTAL,
            length=150
        )
        val_scale.configure(command=lambda v: self.val_size_var.set(round(float(v) * 10) / 10))
        val_scale.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)
        
        self.val_label = ttk.Label(control_frame, text="0.2", width=5)
        self.val_label.grid(row=row, column=2, pady=5)
        self.val_size_var.trace('w', lambda *args: self.val_label.config(text=f"{self.val_size_var.get():.1f}"))
        row += 1
        
        hint_label = ttk.Label(control_frame, text="(可选: 0.1~0.9, 步进0.1)", font=('Arial', 8), foreground='gray')
        hint_label.grid(row=row, column=1, sticky=tk.W, padx=5)
        row += 1
        
        # 预处理设置
        ttk.Label(control_frame, text="预处理设置", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=5)
        row += 1
        
        # Smoothing Method
        ttk.Label(control_frame, text="去噪方法:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.smooth_method_var = tk.StringVar(value="none")
        method_cb = ttk.Combobox(control_frame, textvariable=self.smooth_method_var, values=["none", "rolling", "savgol"], width=13)
        method_cb.grid(row=row, column=1, sticky=tk.W, pady=5, padx=5)
        row += 1
        
        # Window Size
        ttk.Label(control_frame, text="窗口大小:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.window_size_var = tk.IntVar(value=5)
        ttk.Entry(control_frame, textvariable=self.window_size_var, width=15).grid(row=row, column=1, sticky=tk.W, pady=5, padx=5)
        row += 1
        
        # Poly Order (SG only)
        ttk.Label(control_frame, text="多项式阶数:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.poly_order_var = tk.IntVar(value=2)
        ttk.Entry(control_frame, textvariable=self.poly_order_var, width=15).grid(row=row, column=1, sticky=tk.W, pady=5, padx=5)
        row += 1
        
        # 分隔线
        ttk.Separator(control_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        # 超参数设置
        ttk.Label(control_frame, text="XGBoost 超参数", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=5)
        row += 1
        
        # Learning Rate (eta)
        ttk.Label(control_frame, text="学习率 (eta):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.lr_var = tk.DoubleVar(value=0.1)
        ttk.Entry(control_frame, textvariable=self.lr_var, width=15).grid(row=row, column=1, sticky=tk.W, pady=5, padx=5)
        row += 1
        
        # Max Leaves
        ttk.Label(control_frame, text="叶子节点数 (0=不限):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.max_leaves_var = tk.IntVar(value=0)
        ttk.Entry(control_frame, textvariable=self.max_leaves_var, width=15).grid(row=row, column=1, sticky=tk.W, pady=5, padx=5)
        row += 1
        
        # Max Depth
        ttk.Label(control_frame, text="最大深度:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.max_depth_var = tk.IntVar(value=6)
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
        
        # Min Child Weight
        ttk.Label(control_frame, text="最小样本数(权重):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.min_child_var = tk.IntVar(value=1)
        ttk.Entry(control_frame, textvariable=self.min_child_var, width=15).grid(row=row, column=1, sticky=tk.W, pady=5, padx=5)
        row += 1
        
        # Colsample By Tree (Feature Fraction)
        ttk.Label(control_frame, text="特征采样率:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.feature_frac_var = tk.DoubleVar(value=0.8)
        ttk.Entry(control_frame, textvariable=self.feature_frac_var, width=15).grid(row=row, column=1, sticky=tk.W, pady=5, padx=5)
        row += 1
        
        # Subsample (Bagging Fraction)
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
        self.ax.set_box_aspect(1)
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax.set_xlabel('Iteration', fontsize=14)
        self.ax.set_ylabel('Loss (Log Loss)', fontsize=14)
        self.ax.set_title('Training and Validation Loss', fontsize=16)
        self.ax.grid(True, alpha=0.3)
        self.ax.tick_params(labelsize=14)
        self.ax.legend(['Train Loss', 'Validation Loss'], fontsize=14)
        
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
        self.ax.set_xlabel('Iteration', fontsize=14)
        self.ax.set_ylabel('Loss (Log Loss)', fontsize=14)
        self.ax.set_title('Training and Validation Loss', fontsize=16)
        self.ax.grid(True, alpha=0.3)
        self.ax.tick_params(labelsize=14)
        self.ax.legend(fontsize=14)
        # 设置正方形比例和整数X轴
        self.ax.set_box_aspect(1)
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
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
            self.log("井段状态预测模型训练 (XGBoost)")
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
            
            # 1.5 设置预处理参数
            self.predictor.preprocessing_params = {
                'method': self.smooth_method_var.get(),
                'window_size': self.window_size_var.get(),
                'poly_order': self.poly_order_var.get()
            }
            self.log(f"\n预处理参数: {self.predictor.preprocessing_params}")
            
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
            
            # 4. 设置参数 (XGBoost)
            params = {
                'objective': 'multi:softprob',  # 多分类概率
                'num_class': 4,
                'eval_metric': 'mlogloss',
                'booster': 'gbtree',
                'grow_policy': 'lossguide',
                'eta': self.lr_var.get(),  # 学习率
                'max_leaves': self.max_leaves_var.get(),
                'max_depth': self.max_depth_var.get(),
                'min_child_weight': self.min_child_var.get(),
                'subsample': self.bagging_frac_var.get(),
                'colsample_bytree': self.feature_frac_var.get(),
                'verbosity': 1,
                'nthread': -1,
                'num_boost_round': self.num_boost_var.get(),
                'early_stopping_rounds': self.early_stop_var.get()
            }
            
            self.log("\n参数设置:")
            for k, v in params.items():
                self.log(f"{k}: {v}")
            
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
            memory_dir = os.path.join(base_dir, "memory_xgboost")
            
            self.predictor.save_results_by_well(train_df, y_train_pred, y_train_pred_corrected, os.path.join(memory_dir, "train"))
            self.predictor.save_results_by_well(val_df, y_val_pred, y_val_pred_corrected, os.path.join(memory_dir, "test"))
            
            self.log(f"中间结果已保存到 {memory_dir}")
            
            # 6.5. 保存特征重要性
            self.log("\n6.5. 保存特征重要性...")
            self.status_var.set("保存特征重要性...")
            self.predictor.save_feature_importance(memory_dir)
            self.log("特征重要性已保存")

            # 7. 保存模型
            self.log("\n7. 保存模型...")
            self.predictor.save_model(filename="well_status_xgboost.json", params=params)
            
            self.log("\n" + "="*60)
            self.log("训练完成！")
            self.log("="*60)
            
            self.status_var.set("训练完成！")
            self.progress_var.set(100)
            
            messagebox.showinfo("成功", "模型训练完成！")
            
        except Exception as e:
            self.log(f"\n错误: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
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
            filetypes=[("Model files", "*.json"), ("All files", "*.*")],
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
            initialfile=os.path.basename(data_path).replace('.csv', '_xgboost_predictions.csv')
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
            self.log("开始预测 (XGBoost)")
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
            predict_dir = os.path.join(data_dir, "predict_xgboost")
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
    root = tk.Tk()
    app = TrainingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()