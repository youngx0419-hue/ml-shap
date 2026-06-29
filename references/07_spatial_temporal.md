---
name: spatial-temporal-shap
description: 时空 SHAP 分析模块 ML-SHAP — 分层分析架构(strata, not features) + Global/Local Moran's I + Getis-Ord Gi* + KDE + 时空马尔可夫链 + 城市SHAP热力图 + 主导特征空间分布 + 时间序列交叉验证 + GTWR对比。空间/时间变量不作为模型特征，而是作为事后分层变量。
---

# Phase 7：时空 SHAP 异质性分析（ML-SHAP 分层架构）

## 7.0 核心原则：分层分析 (Strata)，非特征编码 [ML-SHAP CRITICAL]

### 严禁的做法

```python
# ❌ 错误: 将城市/时间编码为模型特征
df['City_Code'] = LabelEncoder().fit_transform(df['City'])
X = df[['PM2.5', 'CO', 'City_Code', 'Month']]  # City_Code 不是污染物!
model.fit(X, y)
shap_values = explainer.shap_values(X)
# 输出: "城市编码的 SHAP = 5.2" — 无物理意义, 无法指导决策
```

### 正确的做法

```python
# ✅ 正确: 城市/时间作为分层变量, 模型只用因果特征
X = df[['PM2.5', 'CO', 'NO2', ...]]  # 仅污染物
model.fit(X, y)  # 模型不接触城市/时间列
shap_values = explainer.shap_values(X)

# 分层分析: 在各城市/季节内部分别统计 SHAP 模式
for city in df['City'].unique():
    city_idx = df['City'] == city
    city_shap = shap_values[city_idx].mean(axis=0)
    print(f'{city}: PM2.5 SHAP={city_shap[0]:.1f}')

# 可视化: 城市间 SHAP 对比热力图 (Fig. spatial_city_shap)
# 解读: "城市A的PM2.5贡献>城市B" ≠ "城市名导致了高AQI"
#       正确解读: "城市A的PM2.5浓度更高且处于AQI公式的敏感区间"
```

### 分层变量的本质

- "城市 A 的 AQI 比城市 B 高" ≠ "城市名导致了高 AQI"
- 实际机制: A 的排放源更多 + 气象条件更不利 + 地形不利于扩散
- 分层分析的目的是**分组比较**，不是因果推断

## 7.0a 激活条件

仅当数据包含以下列时激活：
- 空间列：lat/lon/longitude/latitude 或城市/区域名称
- 时间列：date/time/year/month/season

如果数据没有时空列，**跳过本 Phase**。

### 气候/环境领域三角验证注意事项 [??]

气候科学特征重要性研究发现: SHAP 与 Gain-based 重要性在 89% 的站点中方向一致，
约 11% 的站点存在方法间分歧，**气候信号的共线性是主要驱动因素**。
强烈共线性场景下建议同时检查两种重要性排序，分歧>20%时在报告中标注不确定性。

## 7.0b 分层SHAP可视化 [ML-SHAP NEW — 替代旧的特征编码方式]

```python
# 城市×特征 SHAP 热力图 (带单元格数值标注)
city_shap_df = pd.DataFrame(shap_values, columns=feature_names)
city_shap_df['City'] = df['City'].values
city_shap_mean = city_shap_df.groupby('City').mean()

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(city_shap_mean.values, cmap='RdBu_r', aspect='auto',
               vmin=-vmax, vmax=vmax)
ax.grid(False)
# 必须标注单元格数值
for i in range(len(city_shap_mean)):
    for j in range(n_features):
        v = city_shap_mean.values[i, j]
        if abs(v) > vmax * 0.1:
            ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=7,
                    color='white' if abs(v) > vmax*0.6 else '#222222')
ax.set_xticklabels(feature_names, rotation=45, ha='right')
ax.set_yticklabels(city_shap_mean.index)
ax.set_title('各城市SHAP污染物贡献均值 (分层分析, 非模型特征)')
```

---

## 7.1 空间自相关检验（Global Moran's I）

在生成空间图之前，先验证 SHAP 值是否有显著的空间聚集性：

```python
from scipy.spatial.distance import pdist, squareform

# 构建空间权重矩阵（KNN，k=5 或 8）
coords = df[['Longitude', 'Latitude']].values
dist_matrix = squareform(pdist(coords))
k = 5
weights = np.zeros_like(dist_matrix)
for i in range(len(coords)):
    knn_idx = np.argsort(dist_matrix[i])[1:k+1]
    weights[i, knn_idx] = 1.0 / dist_matrix[i, knn_idx]
    weights[knn_idx, i] = weights[i, knn_idx]  # 对称

def morans_i(values, weights):
    n = len(values)
    z = values - values.mean()
    z_sum_sq = (z ** 2).sum()
    numerator = n * (weights * np.outer(z, z)).sum()
    denominator = weights.sum() * 2 * z_sum_sq
    return numerator / denominator if denominator != 0 else 0

mi = morans_i(shap_mean_per_location, weights)
print(f"Moran's I = {mi:.4f} — {'显著空间聚集' if mi > 0.2 else '空间随机分布'}")
```

## 7.1.5 Local Moran's I 与 Getis-Ord Gi* 冷热点分析

Global Moran's I 只能判断全局聚集性，Local 指标可以识别具体热点/冷点位置。

```python
# --- Local Moran's I (LISA) ---
def local_morans_i(values, weights):
    """LISA - 识别高-高、低-低、高-低、低-高聚集区"""
    n = len(values)
    z = values - values.mean()
    z2_sum = (z ** 2).sum()
    I_local = np.zeros(n)
    for i in range(n):
        I_local[i] = z[i] * (weights[i] * z).sum() / (z2_sum / n)
    return I_local

# --- Getis-Ord Gi* 统计量 ---
def getis_ord_gi_star(values, coords, k=8):
    """识别统计显著的热点(Hot)和冷点(Cold)"""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k+1).fit(coords)
    _, indices = nn.kneighbors(coords)
    neighbor_indices = indices[:, 1:]  # 排除自身

    n = len(values)
    mean_all = values.mean(); std_all = values.std()
    gi_stars = np.zeros(n)
    for i in range(n):
        neighbors = neighbor_indices[i]
        w = 1.0 / (np.linalg.norm(coords[neighbors] - coords[i], axis=1) + 1e-10)
        local_mean = np.average(values[neighbors], weights=w)
        local_std = np.sqrt(((values[neighbors] - local_mean)**2).mean())
        gi_stars[i] = (local_mean - mean_all) / (std_all * np.sqrt(w.sum()/n - w.sum()**2/n) + 1e-10)

    hotspot_labels = np.where(gi_stars > 1.96, 'Hot spot (99% CI)',
                     np.where(gi_stars > 1.65, 'Hot spot (95% CI)',
                     np.where(gi_stars < -1.96, 'Cold spot (99% CI)',
                     np.where(gi_stars < -1.65, 'Cold spot (95% CI)',
                              'Not significant'))))
    return gi_stars, hotspot_labels
```

## 7.1.6 核密度估计（KDE）时空分布

```python
from scipy.stats import gaussian_kde

def kde_2d_temporal_comparison(df, x_col, y_col, time_periods):
    """多时期 2D KDE 对比图 —— 展示空间分布的动态演变"""
    fig, axes = plt.subplots(1, len(time_periods), figsize=(4*len(time_periods), 3.5))
    for ax, (t_start, t_end, label) in zip(axes, time_periods):
        sub = df[(df['year'] >= t_start) & (df['year'] <= t_end)]
        x, y = sub[x_col].values, sub[y_col].values
        xy = np.vstack([x, y]); kde = gaussian_kde(xy)
        xi = np.linspace(x.min(), x.max(), 100); yi = np.linspace(y.min(), y.max(), 100)
        Xi, Yi = np.meshgrid(xi, yi); Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
        ax.contourf(Xi, Yi, Zi, levels=10, cmap='YlOrRd', alpha=0.85)
        ax.set_title(f'{label}', fontsize=10, fontweight='bold')
    return fig
```

## 7.1.7 时空马尔可夫链

```python
def spatial_markov_chain(values_by_time, spatial_lags_by_time, n_classes=5):
    """传统 vs 空间马尔可夫链转移概率矩阵"""
    def discretize(arr, k):
        quantiles = np.percentile(arr, np.linspace(0, 100, k+1))
        return np.digitize(arr, quantiles[:-1]) - 1

    trans_traditional = np.zeros((n_classes, n_classes))
    for t in range(len(values_by_time) - 1):
        current = discretize(values_by_time[t], n_classes)
        next_c = discretize(values_by_time[t+1], n_classes)
        for i in range(len(current)):
            trans_traditional[current[i], next_c[i]] += 1
    trans_traditional /= trans_traditional.sum(axis=1, keepdims=True)

    trans_spatial = np.zeros((n_classes, n_classes, n_classes))
    for t in range(len(values_by_time) - 1):
        current = discretize(values_by_time[t], n_classes)
        neighbor = discretize(spatial_lags_by_time[t], n_classes)
        next_c = discretize(values_by_time[t+1], n_classes)
        for i in range(len(current)):
            trans_spatial[current[i], neighbor[i], next_c[i]] += 1
    for i in range(n_classes):
        for j in range(n_classes):
            if trans_spatial[i, j].sum() > 0:
                trans_spatial[i, j] /= trans_spatial[i, j].sum()
    return trans_traditional, trans_spatial
```

## 7.2 空间 SHAP 气泡地图

```python
from matplotlib.colors import TwoSlopeNorm

city_shap = df.groupby(['City', 'Longitude', 'Latitude']).agg({
    f'SHAP_{feat}': 'mean' for feat in top_features
}).reset_index()

norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
sc = ax.scatter(city_shap['Longitude'], city_shap['Latitude'],
                c=city_shap[f'SHAP_{top_feat}'], s=bubble_sizes,
                cmap='RdBu_r', norm=norm, edgecolor='black', linewidth=0.3, alpha=0.9)
```

## 7.2b Geo-SHAP 地图叠加 [ML-SHAP NEW — 条件激活]

仅当检测到 lat/lon 列时激活。需要 `geopandas` + `contextily` 依赖。

```python
# 条件导入 (非强制依赖)
try:
    import geopandas as gpd
    import contextily as ctx
    GEO_OK = True
except ImportError:
    GEO_OK = False
    print('[INFO] Geo-SHAP skipped: geopandas/contextily not available')

if GEO_OK and 'lat' in df.columns and 'lon' in df.columns:
    # 转换为 GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df['lon'], df['lat']),
        crs='EPSG:4326'
    ).to_crs(epsg=3857)  # Web Mercator

    # SHAP 值映射到地图颜色
    fig, ax = plt.subplots(figsize=(12, 8))
    gdf.plot(column='shap_sum', cmap='RdBu_r', ax=ax,
             markersize=df['shap_sum'].abs() / df['shap_sum'].abs().max() * 80 + 5,
             alpha=0.7, edgecolor='black', linewidth=0.3,
             legend=True, legend_kwds={'label': 'Total |SHAP|', 'shrink': 0.6})
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=8)
    ax.set_axis_off()
    ax.set_title('Spatial distribution of SHAP contributions')
    save_chart(fig, charts_dir / 'fig_spatial_geo_shap')
```

**输出**: `Charts/fig_spatial_geo_shap.{svg,png}` (仅当 lat/lon 存在 + geopandas 可用)

## 7.3 主导特征空间分布图

```python
def dominant_feature_map(city_shap, features):
    shap_cols = [f'SHAP_{f}' for f in features]
    abs_vals = np.abs(city_shap[shap_cols].values)
    dom_idx = np.argmax(abs_vals, axis=1)
    city_shap['dominant_feature'] = [features[i] for i in dom_idx]
    city_shap['dominant_abs_shap'] = abs_vals[np.arange(len(city_shap)), dom_idx]
    return city_shap
# 洞察: "城市A和B的主导特征都是PM2.5，沿海城市C的主导特征变为湿度"
```

## 7.4 区域×时间 SHAP 热力图

```python
region_month = df.groupby(['Region', 'Month']).agg({
    f'SHAP_{feat}': 'mean' for feat in top_features
}).reset_index()
pivot = region_month.pivot(index='Region', columns='Month', values=f'SHAP_{feat}')

im = ax.imshow(pivot.values, cmap='RdBu_r', norm=TwoSlopeNorm(vcenter=0), aspect='auto')
# 必须标注单元格数值
for r in range(pivot.shape[0]):
    for c in range(pivot.shape[1]):
        v = pivot.values[r, c]
        tcol = 'white' if abs(v) > 0.5*abs(pivot.values).max() else '#222222'
        ax.text(c, r, f'{v:.1f}', ha='center', va='center', fontsize=6, color=tcol)
```

## 7.5 风险象限散点图

```python
for region in regions:
    for season in seasons:
        sub = df[(df['Region']==region)&(df['Season']==season)]
        ax.scatter(sub['predicted'], sub[f'SHAP_{feat}'], s=sizes,
                   color=region_colors[region], alpha=0.78,
                   edgecolor='black', linewidth=0.45)
ax.axhline(y=0, color='#666', ls='--', lw=0.9)
ax.axvline(x=df['predicted'].median(), color='#666', ls='--', lw=0.9)
# 四象限: (高预测,高SHAP)/(高预测,低SHAP)/(低预测,高SHAP)/(低预测,低SHAP)
```

## 7.6 时间序列交叉验证

当数据有时间列时，**禁止随机 K-Fold CV**（会导致未来信息泄漏）：

```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
df_sorted = df.sort_values('Date')
for train_idx, test_idx in tscv.split(df_sorted):
    X_train = df_sorted.iloc[train_idx][features]
    y_train = df_sorted.iloc[train_idx][target]
    X_test = df_sorted.iloc[test_idx][features]
    y_test = df_sorted.iloc[test_idx][target]
```

## 7.7 月度 SHAP 演变追踪

```python
monthly_shap = df.groupby('Month').agg({
    f'SHAP_{feat}': 'mean' for feat in top_features
})
fig, ax = plt.subplots(figsize=(8, 4))
for feat in top_features[:5]:
    ax.plot(monthly_shap.index, monthly_shap[f'SHAP_{feat}'], 'o-', lw=1.5, markersize=5, label=feat)
ax.axhline(y=0, color='gray', ls='--', lw=0.8)
```

## 7.8 输出

保存到 `05_Results/<项目名>_v<N>/`：
- `Charts/fig_spatial_city_aqi.{svg,png}` — 城市分层AQI排名
- `Charts/fig_spatial_city_shap.{svg,png}` — 城市分层SHAP热力图 (含数值标注)
- `Charts/fig_spatial_geo_shap.{svg,png}` — Geo-SHAP地图叠加 [ML-SHAP, 条件激活]
- `Charts/fig_seasonal_aqi.{svg,png}` — 季节分层AQI分布
- `Charts/fig_temporal_monthly_aqi.{svg,png}` — 月度AQI趋势
- `Tables/spatial_autocorrelation_report.csv`
- `Tables/temporal_cv_results.csv`
- `Tables/city_spatial_shap_summary.csv`

## 7.9 GTWR 对比验证

```python
def compare_xgb_vs_gtwr(xgb_pred, gtwr_pred, y_true, coords):
    from sklearn.metrics import r2_score, mean_squared_error
    metrics = {
        'XGBoost_R2': r2_score(y_true, xgb_pred),
        'GTWR_R2': r2_score(y_true, gtwr_pred),
        'XGBoost_RMSE': np.sqrt(mean_squared_error(y_true, xgb_pred)),
        'GTWR_RMSE': np.sqrt(mean_squared_error(y_true, gtwr_pred)),
    }
    xgb_residuals = y_true - xgb_pred
    gtwr_residuals = y_true - gtwr_pred
    metrics['XGBoost_Residual_Moran_I'] = morans_i(xgb_residuals, build_weights(coords))
    metrics['GTWR_Residual_Moran_I'] = morans_i(gtwr_residuals, build_weights(coords))
    return metrics
```

## 7.10 空间情景模拟接口

```python
def spatial_scenario_interface(model, base_features, scenario_configs, coords):
    """空间情景模拟接口: BAU / Ecological / Development 等"""
    results = {}
    for scenario_name, params in scenario_configs.items():
        X_scenario = modify_spatial_features(base_features.copy(), params)
        pred = model.predict(X_scenario)
        shap_scenario = explainer.shap_values(X_scenario)
        results[scenario_name] = {'prediction': pred, 'shap_values': shap_scenario}
    return results
```
