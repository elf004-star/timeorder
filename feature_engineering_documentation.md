# 井段状态预测模型特征工程文档

本文档详细整理了模型训练中使用的特征工程逻辑。所有特征均基于按井号分组后的时序数据计算。

## 0. 基础符号定义

*   $t$: 当前时间步（深度点序号）
*   $JX$: 原始井斜数据序列
*   $S$: 平滑处理后的井斜序列 (`JX_smooth`)，是后续大部分特征的基础输入
    *   *预处理参数默认值：Window=5, Poly=2*
*   $D$: 原始狗腿度序列 (`Dogleg Severity`)
*   $N$: 该口井的总数据点数
*   $\mu_{win}$: 指定窗口大小的移动平均值
*   $\sigma_{win}$: 指定窗口大小的移动标准差
*   Slope: 线性回归斜率 ($y = kx + b$ 中的 $k$)

---

## 1. 基础与导向特征 (Dogleg & Direction)

这些特征结合了狗腿度大小与井斜变化的方向，用于捕捉井眼轨迹变化的力度和方向。

| 特征名 (Symbol) | 含义 (Description) | 计算公式 / 逻辑 (Formula) |
| :--- | :--- | :--- |
| **dogl_s** | 带符号的狗腿度 | $D_t \times \text{sign}(S_{t+1} - S_t)$ <br> (若 $S_{t+1} < S_t$ 取-1，否则取1) |
| **dogl_s_mean_5** | 符号狗腿度均值 | $\text{SMA}_5(\text{dogl\_s})$ |
| **dogl_s_std_5** | 符号狗腿度波动 | $\text{Std}_5(\text{dogl\_s})$ |
| **dogl_s_trend_5** | 符号狗腿度趋势 | $\text{Slope}_5(\text{dogl\_s})$ |
| **dogl_s_diff_1** | 符号狗腿度变化量 | $\text{dogl\_s}_t - \text{dogl\_s}_{t-1}$ |
| **dogl_s_cumsum** | 累计符号狗腿度 | $\sum_{i=0}^t \text{dogl\_s}_i$ |
| **dogl_s_lag_1** | 上一点的符号狗腿度 | $\text{dogl\_s}_{t-1}$ |

---

## 2. 位置与状态特征 (Location & Status)

基于组内统计值的归一化特征，用于判断当前点在整口井中的相对位置和状态。

| 特征名 (Symbol) | 含义 (Description) | 计算公式 / 逻辑 (Formula) |
| :--- | :--- | :--- |
| **norm_value** | 组内归一化井斜 | $(S_t - S_{\min}) / (S_{\max} - S_{\min})$ |
| **rel_pos** | 相对深度位置 | $(t + 1) / N$ |
| **dist_to_max** | 距离最大井斜的差值 | $S_{\max} - S_t$ |
| **is_max_region** | 是否处于最大值区域 | $1 \text{ if } S_t > (S_{\min} + 0.95(S_{\max} - S_{\min})) \text{ else } 0$ |
| **accum_ratio** | 累计井斜占比 | $\sum_{i=0}^t S_i / \sum_{i=0}^{N-1} S_i$ |

---

## 3. 速度与趋势特征 (Velocity & Trend)

基于一阶导数及趋势线，反映井斜变化的快慢和方向。

| 特征名 (Symbol) | 含义 (Description) | 计算公式 / 逻辑 (Formula) |
| :--- | :--- | :--- |
| **diff_1** | 一阶差分 (瞬时速度) | $S_t - S_{t-1}$ |
| **diff_3** | 三阶跨度差分 | $S_t - S_{t-3}$ |
| **trend_slope_5** | 5点局部趋势斜率 | $\text{Slope}(S_{t-4}, \dots, S_t)$ |
| **trend_slope_10** | 10点局部趋势斜率 | $\text{Slope}(S_{t-9}, \dots, S_t)$ |
| **ema_divergence** | EMA偏离度 | $S_t - \text{EMA}_{10}(S)$ |
| **pct_change** | 变化率 | $(S_t - S_{t-1}) / S_{t-1}$ |
| **consecutive_up** | 连续上升计数 | 截止当前连续 $S_t > S_{t-1}$ 的次数 |
| **consecutive_down** | 连续下降计数 | 截止当前连续 $S_t < S_{t-1}$ 的次数 |

---

## 4. 拐点与加速度特征 (Inflection & Acceleration)

基于二阶导数及统计波动，反映轨迹变化的剧烈程度和弯曲情况。

| 特征名 (Symbol) | 含义 (Description) | 计算公式 / 逻辑 (Formula) |
| :--- | :--- | :--- |
| **diff_2** | 二阶差分 (加速度) | $\text{diff\_1}_t - \text{diff\_1}_{t-1}$ |
| **slope_change** | 斜率变化率 | $\text{trend\_slope\_5}_t - \text{trend\_slope\_5}_{t-1}$ |
| **curvature** | 局部曲率 (简化版) | $|S''_t| / (1 + (S'_t)^2)^{1.5}$ <br> (其中 $S' \approx \text{diff\_1}, S'' \approx \text{diff\_2}$) |
| **peak_accel_loc** | 是否加速度峰值 | $1 \text{ if } |\text{diff\_2}_t| = \max(|\text{diff\_2}|_{t-2:t+2}) \text{ else } 0$ |
| **std_5** | 5点局部波动率 | $\sigma_5(S_{t-4}, \dots, S_t)$ |
| **std_10** | 10点局部波动率 | $\sigma_{10}(S_{t-9}, \dots, S_t)$ |
| **z_score_local** | 局部Z-Score | $(S_t - \mu_{10}) / \sigma_{10}$ |

---

## 5. 双向上下文特征 (Look-ahead Context)

利用未来数据点的信息（Look-ahead），这对离线分析或回顾性预测非常重要，能捕捉即将发生的变化。

| 特征名 (Symbol) | 含义 (Description) | 计算公式 / 逻辑 (Formula) |
| :--- | :--- | :--- |
| **lead_diff_1** | 未来一步差分 | $S_{t+1} - S_t$ |
| **lead_diff_3** | 未来三步差分 | $S_{t+3} - S_t$ |
| **lead_slope_5** | 未来趋势斜率 | $\text{Slope}(S_t, \dots, S_{t+4})$ |
| **center_diff** | 中心差分 | $(S_{t+1} - S_{t-1}) / 2$ |
| **pre_post_ratio** | 前后均值比 | $\mu_{\text{future\_5}} / \mu_{\text{past\_5}}$ |
| **lag_rolling_mean_5** | 过去均值 | $\mu_{\text{past\_5}} = \text{Mean}(S_{t-4}, \dots, S_t)$ |
| **lead_rolling_mean_5** | 未来均值 | $\mu_{\text{future\_5}} = \text{Mean}(S_t, \dots, S_{t+4})$ |
| **cross_region** | 趋势穿越标识 | $1 \text{ if } \mu_{\text{future\_5}} > 1.02 \times \mu_{\text{past\_5}} \text{ else } 0$ |
