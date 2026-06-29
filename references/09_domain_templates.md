---
name: domain-templates
description: 五大领域自适应分析模板 —— 根据 Phase 0 识别的应用领域（碳排放/水环境/风险评估/时空格局/生物炭材料），自动切换对应的特征工程、建模策略、SHAP 解释重点、情景模拟和可视化方案。
---

# Phase 9：领域自适应分析与情景模拟

## 9.0a 领域上下文优先级

在使用任何领域模板前，先读取 `domain_context.json`：

1. 若用户明确指定领域，以用户指定为准。
2. 若 `domain_context.json.confidence >= 0.55`，使用检测领域。
3. 若置信度不足，向用户确认 top candidates 后再继续。
4. 若 `nearest_recent_papers` 非空，用三篇近邻文献中的方法流程和可视化习惯覆盖本文件的默认建议。
5. 若无实时文献检索，使用本文件默认模板，但在报告中说明检索不可用。

领域模板的作用是给出默认研究路径；最终执行路径应以 `domain_context.json.domain_method_adjustments` 和 `method_decision_log.json` 为准。

## 9.0 领域调度逻辑

```python
def detect_domain(df, user_override=None):
    """根据数据列名和内容自动推断应用领域"""
    if user_override:
        return user_override

    col_names = ' '.join(df.columns).lower()

    # 碳排放
    carbon_keywords = ['carbon', 'co2', 'emission', '碳排放', '碳达峰', '碳中和',
                       '能耗', '能源', '碳强度', '碳储量']
    if any(k in col_names for k in carbon_keywords):
        return 'carbon'

    # 水环境
    water_keywords = ['do', 'cod', 'nh3', 'tp', 'tn', 'ph', 'bod', '溶解氧',
                      '水质', '污染物', '浓度', '浊度', 'tss', 'ec', '盐度']
    if any(k in col_names for k in water_keywords):
        return 'water'

    # 风险评估
    risk_keywords = ['risk', 'hazard', 'fire', 'flood', '火灾', '洪水', '滑坡',
                     '地震', '易发性', '风险', 'susceptibility', 'vulnerability']
    if any(k in col_names for k in risk_keywords):
        return 'risk'

    # 时空格局
    spatial_keywords = ['ndvi', 'landscape', '景观', '生态韧性', '格局', 'landuse',
                        '土地利用', '生物量', 'biomass', '城市化', 'urban']
    if any(k in col_names for k in spatial_keywords):
        return 'spatial_temporal'

    # 生物炭材料
    material_keywords = ['biochar', 'adsorption', 'pyrolysis', '吸附', '热解',
                         '生物炭', '去除率', 'sorption', '表面积', 'bet']
    if any(k in col_names for k in material_keywords):
        return 'materials'

    return 'general'
```

---

## 9.1 碳排放专项模板

### 9.1.1 特征工程特定策略

```python
# 1. 碳排放因素分解（LMDI / Kaya）
# C = P × (GDP/P) × (E/GDP) × (C/E)
#   人口 × 人均GDP   × 能源强度 × 碳排放系数
def lmdi_decomposition(df):
    """对数平均迪氏指数分解法"""
    df['energy_intensity'] = df['energy_consumption'] / df['GDP']
    df['carbon_coefficient'] = df['carbon_emission'] / df['energy_consumption']
    df['gdp_per_capita'] = df['GDP'] / df['population']
    # 各因素贡献
    factors = ['population', 'gdp_per_capita', 'energy_intensity', 'carbon_coefficient']
    return df, factors

# 2. 时间滞后特征（碳排放的惯性效应）
def create_carbon_lag_features(df, target_col, lags=[1, 2, 3, 5]):
    for lag in lags:
        df[f'{target_col}_lag{lag}'] = df.groupby('region')[target_col].shift(lag)
    return df

# 3. 政策虚拟变量（碳交易试点、环保督察等）
def add_policy_dummies(df, year_col, policy_years):
    for policy_name, pyear in policy_years.items():
        df[f'policy_{policy_name}'] = (df[year_col] >= pyear).astype(int)
    return df
```

### 9.1.2 多情景碳达峰预测

```python
import numpy as np

def carbon_peak_scenario_simulation(model, base_features, scenarios, n_years=20):
    """
    多情景碳达峰预测
    scenarios = {
        'baseline':    {'gdp_growth': 0.05, 'energy_efficiency_improve': 0.01},
        'low_carbon':  {'gdp_growth': 0.04, 'energy_efficiency_improve': 0.03},
        'aggressive':  {'gdp_growth': 0.03, 'energy_efficiency_improve': 0.05},
    }
    """
    results = {}
    for scenario_name, params in scenarios.items():
        trajectory = []
        features = base_features.copy()
        for t in range(n_years):
            pred = model.predict(features.reshape(1, -1))[0]
            trajectory.append(pred)
            # 更新特征（GDP增长、能效提升等）
            features = update_features_for_scenario(features, params, t)
        peak_year = np.argmax(trajectory)  # or argmin depending
        peak_value = np.max(trajectory)
        results[scenario_name] = {
            'trajectory': trajectory,
            'peak_year': peak_year,
            'peak_value': peak_value,
        }
    return results
```

### 9.1.3 SHAP 解释重点

- **全局层面**：GDP、能源强度、产业结构、城镇化率的贡献排序
- **非线性阈值**：城镇化率 ~65%、人均 GDP ~$15,000 处的碳排放拐点
- **交互效应**：经济增长 × 能源结构的交互——"增长是否必然带来排放"
- **空间异质性**：东中西三大区域的驱动因素差异

### 9.1.4 特色输出图表

```python
# 图 1：多情景碳达峰路径图（3 条彩色曲线 + 达峰标记）
# 图 2：碳排放驱动因素 SHAP 贡献分解瀑布图
# 图 3：城镇化率-碳排放 SHAP 依赖图（标注阈值拐点）
# 图 4：城市碳减排策略矩阵（排放量 vs 排放强度，4 象限）
# 图 5：碳达峰概率空间分布图
```

---

## 9.2 水环境专项模板

### 9.2.1 特征工程特定策略

```python
# 1. 水质综合指数计算
def calculate_wqi(df):
    """加权水质指数"""
    weights = {'DO': 0.17, 'COD': 0.12, 'NH3': 0.15, 'TP': 0.10,
               'TN': 0.10, 'pH': 0.08, 'Turbidity': 0.08, 'EC': 0.08}
    # 标准化后加权求和
    df['WQI'] = sum(df[c] * w for c, w in weights.items() if c in df.columns)
    return df

# 2. 水质等级标签（基于国标 GB 3838-2002）
def classify_water_quality(df, param_cols):
    """根据各项指标最差值归类 I-V 类"""
    thresholds = {
        'DO':    [7.5, 6, 5, 3, 2],      # I II III IV V
        'COD':   [15, 15, 20, 30, 40],
        'NH3':   [0.15, 0.5, 1.0, 1.5, 2.0],
        'TP':    [0.02, 0.1, 0.2, 0.3, 0.4],
    }
    # 取各指标最差等级
    class_labels = []
    for i in range(len(df)):
        worst_class = 1
        for param, thresh in thresholds.items():
            if param in df.columns:
                val = df[param].iloc[i]
                for cls_idx, t in enumerate(thresh):
                    if (param == 'DO' and val >= t) or (param != 'DO' and val <= t):
                        break
                worst_class = max(worst_class, cls_idx + 1)
        class_labels.append(worst_class)
    return class_labels

# 3. 季节/水文期标记
def add_hydrological_period(df, month_col):
    """丰水期(6-9)/平水期(3-5,10-11)/枯水期(12-2)"""
    conditions = [
        df[month_col].isin([6,7,8,9]),
        df[month_col].isin([3,4,5,10,11]),
        df[month_col].isin([12,1,2]),
    ]
    choices = ['wet', 'normal', 'dry']
    df['hydro_period'] = np.select(conditions, choices)
    return df
```

### 9.2.2 SHAP 解释重点

- **关键参数排序**：识别主导水质退化/改善的参数
- **非线性响应**：DO 对温度/NH3 的非线性响应曲线
- **污染源解析**：通过 SHAP 交互效应推测污染来源（点源 vs 面源）
- **季节异质性**：丰/平/枯水期的驱动因素差异

### 9.2.3 特色输出图表

```python
# 图 1：水质等级空间分布地图
# 图 2：关键参数 SHAP 蜂群图（着色 = 水文期）
# 图 3：DO 浓度对温度的 SHAP 依赖图（着色 = NH3）
# 图 4：污染源贡献比例玫瑰图（按 SHAP 推算）
# 图 5：水质超标风险概率热力图（时间×空间）
```

---

## 9.3 风险评估专项模板

### 9.3.1 特征工程特定策略

```python
# 1. 多源数据融合
def fuse_risk_features(df, dem_path, landuse_path, climate_path):
    """融合地形、土地利用、气象等多源数据"""
    # DEM 衍生特征：高程、坡度、坡向、地形起伏度
    # 土地利用：建成区比例、绿地比例、水域比例（缓冲区统计）
    # 气象：年均降水、极端降水日数、高温日数
    # 基础设施：路网密度、排水管网密度、消防站可达性
    # 社会经济：人口密度、GDP 密度、老龄化比例
    pass

# 2. 样本不均衡处理（灾害/风险数据典型问题）
def handle_imbalanced_risk_data(X, y, method='smote'):
    """风险样本通常严重不均衡（灾害点 << 非灾害点）"""
    if method == 'smote':
        from imblearn.over_sampling import SMOTE
        X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X, y)
    elif method == 'weighted':
        scale_pos_weight = (y == 0).sum() / (y == 1).sum()
        model_params['scale_pos_weight'] = scale_pos_weight
    return X_resampled, y_resampled

# 3. 概率校准（风险概率可解释性）
def calibrate_risk_probability(model, X_calib, y_calib):
    from sklearn.calibration import CalibratedClassifierCV
    calibrated = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
    calibrated.fit(X_calib, y_calib)
    return calibrated
```

### 9.3.2 风险分区与不确定性

```python
# 风险等级划分（自然断点法）
from jenkspy import jenks_breaks
breaks = jenks_breaks(risk_probs, n_classes=5)

# 不确定性量化
risk_ci_lower = np.percentile(bootstrap_preds, 2.5, axis=0)
risk_ci_upper = np.percentile(bootstrap_preds, 97.5, axis=0)
uncertainty = risk_ci_upper - risk_ci_lower
```

### 9.3.3 SHAP 解释重点

- **驱动因子排序**：什么因素最能区分高风险和低风险区域
- **交互效应**：降水×地形的交互如何放大风险
- **空间聚集**：SHAP 值的 Moran's I 检验风险驱动因子的空间溢出效应
- **子群体分析**：不同风险等级的样本 SHAP 模式差异

### 9.3.4 特色输出图表

```python
# 图 1：风险等级分区地图（5 级自然断点着色）
# 图 2：SHAP 蜂群图（按风险等级着色）
# 图 3：风险象限散点图（预测概率 vs Top 特征 SHAP）
# 图 4：不确定性空间分布热力图
# 图 5：Top-3 风险因子依赖图（着色=降水/地形/人口）
```

---

## 9.4 时空格局专项模板

### 9.4.1 时空特征构造

```python
# 1. 时空立方体构建
def build_space_time_cube(df, x_col, y_col, t_col, value_col):
    """构建 (空间单元 × 时间点) 的完整网格"""
    space_time = df.pivot_table(
        index=[x_col, y_col],
        columns=t_col,
        values=value_col
    )
    return space_time

# 2. 时空滞后特征
def create_spatial_lag(df, value_col, x_col, y_col, k=5):
    """空间滞后 = KNN 邻居的加权均值"""
    from sklearn.neighbors import NearestNeighbors
    coords = df[[x_col, y_col]].values
    nn = NearestNeighbors(n_neighbors=k+1).fit(coords)
    distances, indices = nn.kneighbors(coords)
    spatial_lag = np.zeros(len(df))
    for i in range(len(df)):
        weights = 1.0 / (distances[i, 1:] + 1e-10)
        weights /= weights.sum()
        spatial_lag[i] = np.average(df[value_col].iloc[indices[i, 1:]], weights=weights)
    return spatial_lag

# 3. 时空交互项
df['space_time_interaction'] = df['spatial_lag'] * df['temporal_lag']
```

### 9.4.2 时空马尔可夫链

```python
def spatial_markov_chain(values, classes, spatial_lag):
    """
    时空马尔可夫链：考虑空间滞后条件下的类别转移概率
    经典论文方法：Rey (2001), 广泛应用于时空格局论文
    """
    n_classes = len(classes)
    # 条件转移矩阵：(当前类别 × 空间滞后类别) → 下一期类别
    trans_matrix = np.zeros((n_classes, n_classes, n_classes))

    for t in range(len(values) - 1):
        current_class = np.digitize(values[t], classes)
        next_class = np.digitize(values[t+1], classes)
        lag_class = np.digitize(spatial_lag[t], classes)
        trans_matrix[current_class, lag_class, next_class] += 1

    # 行归一化
    for i in range(n_classes):
        for j in range(n_classes):
            row_sum = trans_matrix[i, j, :].sum()
            if row_sum > 0:
                trans_matrix[i, j, :] /= row_sum
    return trans_matrix
```

### 9.4.3 冷热点分析（Getis-Ord Gi*）

```python
# 识别 SHAP 值的空间聚集热点
from pysal.explore import esda
from pysal.lib import weights

# 基于 KNN 的空间权重
w = weights.KNN.from_dataframe(df[['lon', 'lat']], k=8)
# Getis-Ord Gi* 统计量
gi_star = esda.G_Local(df['shap_value'].values, w, star=True)
# 识别显著热点（Z > 1.96）和冷点（Z < -1.96）
df['hotspot'] = np.where(gi_star.Zs > 1.96, 'Hot',
                np.where(gi_star.Zs < -1.96, 'Cold', 'Not significant'))
```

### 9.4.4 特色输出图表

```python
# 图 1：时空立方体可视化（3D 或 Hovmöller 图）
# 图 2：冷热点时空演化序列图
# 图 3：主导特征空间分布图（每个空间单元标注最强影响特征）
# 图 4：特征贡献月度/季节热力图（特征×时间）
# 图 5：时空转移概率弦图（马尔可夫链）
```

---

## 9.5 生物炭材料专项模板

### 9.5.1 材料特征工程

```python
# 1. 材料特性工程
def engineer_biochar_features(df):
    """从原始实验数据构造材料特征"""
    # 原子比（关键！）
    df['H_C_ratio'] = df['H_content'] / df['C_content']
    df['O_C_ratio'] = df['O_content'] / df['C_content']
    df['N_C_ratio'] = df.get('N_content', 0) / df['C_content']
    # 极性指数
    df['polarity_index'] = (df['O_content'] + df['N_content']) / df['C_content']
    # 灰分修正比表面积
    if 'ash_content' in df.columns:
        df['ash_corrected_SSA'] = df['SSA'] * (1 - df['ash_content'] / 100)
    return df

# 2. 原料类型编码（重要类别特征）
BIOMASS_TYPES = {
    'agricultural_waste':    ['rice_husk', 'wheat_straw', 'corn_stover'],
    'woody_biomass':         ['pine', 'oak', 'bamboo', 'poplar'],
    'animal_waste':          ['chicken_manure', 'cow_dung', 'pig_manure'],
    'sewage_sludge':         ['municipal_sludge', 'industrial_sludge'],
    'algae':                 ['chlorella', 'spirulina', 'macroalgae'],
}
```

### 9.5.2 CatBoost 优先策略

生物炭材料数据通常包含大量类别型特征（原料类型、活化剂种类、改性方法等），CatBoost 天然支持类别特征，无需手动编码：

```python
# CatBoost 直接处理类别特征
cat_features = ['biomass_type', 'activation_agent', 'modification_method',
                'pyrolysis_atmosphere']
model = CatBoostRegressor(
    cat_features=cat_features,
    depth=6, learning_rate=0.1, n_estimators=200,
    verbose=0, random_seed=42
)
```

### 9.5.3 PDP 工艺参数优化

```python
from sklearn.inspection import partial_dependence

# 二维 PDP：热解温度 × 停留时间
fig, ax = plt.subplots(figsize=(5, 4))
pdp = partial_dependence(model, X, features=[('pyrolysis_temp', 'residence_time')],
                         grid_resolution=30)

# 等高线 + 最优区域标记
X_grid, Y_grid = np.meshgrid(pdp['values'][0], pdp['values'][1])
Z = pdp['average'][0]
contour = ax.contourf(X_grid, Y_grid, Z, levels=15, cmap='RdYlBu_r')
ax.contour(X_grid, Y_grid, Z, levels=8, colors='black', linewidths=0.3)

# 标记最优区域（前 10% 的 Z 值）
optimal_mask = Z >= np.percentile(Z, 90)
ax.contourf(X_grid, Y_grid, optimal_mask, levels=1,
            colors='none', hatches=['//'], alpha=0.3)
```

### 9.5.4 特色输出图表

```python
# 图 1：材料特性相关性热力图（原子比、SSA、官能团等）
# 图 2：CatBoost vs XGBoost vs RF 性能对比
# 图 3：关键特性 SHAP 蜂群图（按原料类型着色）
# 图 4：热解温度 × 停留时间 PDP 二维等高线图（目标：最大吸附容量）
# 图 5：最优工艺参数组合推荐雷达图
```

---

## 9.6 通用输出规范

所有领域模板的输出保存到 `05_Results/09_领域专项分析/`：

```
05_Results/09_领域专项分析/
├── 碳排放/
│   ├── carbon_peak_scenarios.csv
│   ├── lmdi_decomposition.csv
│   ├── city_strategy_matrix.csv
│   └── Pictures/
├── 水环境/
│   ├── water_quality_classification.csv
│   ├── pollution_source_analysis.csv
│   └── Pictures/
├── 风险评估/
│   ├── risk_zoning.csv
│   ├── uncertainty_map_data.csv
│   └── Pictures/
├── 时空格局/
│   ├── spatial_markov_matrix.csv
│   ├── hotspot_analysis.csv
│   └── Pictures/
└── 生物炭材料/
    ├── parameter_optimization.csv
    ├── material_design_guide.csv
    └── Pictures/
```
