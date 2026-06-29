---
name: shap-advanced
description: Phase 6 进阶 SHAP 分析 —— 修正交互效应分解 (逐样本分解) + 交互矩阵蓝橙双色 + Bootstrap CI + SHAP 聚类子群体分析
---

# Phase 6：进阶 SHAP 分析 [ML-SHAP]

## 6.0 文献驱动的进阶解释门槛

只有在模型泛化合格且 `method_decision_log.json` 没有高风险阻断项时才执行进阶解释：

- 强相关特征：交互矩阵只能作为模型行为描述，必须配合特征族解释或 ALE。
- 科学解释目标：执行 bootstrap 稳定性、跨模型排名一致性和解释 sanity check。
- 高风险任务：报告“后验解释不能替代透明模型”的限制，并保留可解释基线对照。
- 交互 SHAP 使用 tree-path-dependent explainer；不要把该模式的交互结果和 interventional 主效应混称为同一因果语义。

## 6.1 SHAP 交互值计算 [ML-SHAP — 完整重写]

### 解释器创建 (两个独立 explainer, 模型本体相同)

```python
# 交互分析必须用 tree_path_dependent 模式
# interventional 模式下 shap_interaction_values() 静默返回全零
explainer_inter = shap.TreeExplainer(best_model)  # 默认 = tree_path_dependent

# 抽样计算 (O(N·F²) 复杂度)
N_INTER = min(500, len(X_shap))
X_inter_sample = X_shap[:N_INTER]
shap_inter_values = explainer_inter.shap_interaction_values(X_inter_sample)
```

### 交互效应分解 [????]

**v1.0 的 `(off_diag - diag) / 2` 是错误的**。正确算法为逐样本分解:

```python
n_feat = len(feature_names)

# 1. 主效应 = 对角线元素 (interventional 模式下即 shap_values)
main_effects = np.abs(shap_inter_values[:, range(n_feat), range(n_feat)]).mean(axis=0)

# 2. 交互效应 (逐样本逐特征分解):
#    特征 i 在样本 n 上的交互贡献 = sum_j(inter[n,i,j]) - inter[n,i,i]
#    然后取 |交互贡献| 在所有样本上的均值
inter_effects = np.zeros(n_feat)
for i in range(n_feat):
    contribs = np.zeros(N_INTER)
    for n in range(N_INTER):
        total_contrib = shap_inter_values[n, i, :].sum()
        main_contrib = shap_inter_values[n, i, i]
        contribs[n] = total_contrib - main_contrib
    inter_effects[i] = np.abs(contribs).mean()

# 3. 交互矩阵 (蓝橙双色的源数据):
inter_mat = np.zeros((n_feat, n_feat))
for i in range(n_feat):
    for j in range(n_feat):
        inter_mat[i, j] = shap_inter_values[:, i, j].mean()

# 4. 验证: inter_mat 应对称, sum_j(inter_mat[i,:]) ≈ shap_values[:,i].mean()
```

**预期输出特征**:
- PM₂.₅ 交互比 < 0.2 (独立首要驱动)
- NO₂ 交互比可能 > 1.0 (主要通过交互贡献)
- 其他特征交互比在 0.3-0.9 之间

## 6.2 图表 E: Main vs Interaction [ML-SHAP 修正]

```python
total_effects = main_effects + inter_effects
sort_idx = np.argsort(total_effects)[::-1]

fig, ax = plt.subplots(figsize=(10, 5.5))
x_pos = np.arange(n_feat)
width = 0.35

bars_main = ax.bar(x_pos - width/2, main_effects[sort_idx], width,
                   label='Main Effect', color=AZURE_MAIN, alpha=0.85,
                   edgecolor='white', linewidth=0.5)
bars_inter = ax.bar(x_pos + width/2, inter_effects[sort_idx], width,
                    label='Interaction Effect', color='#c1724a', alpha=0.75,
                    edgecolor='white', linewidth=0.5)

ax.set_xticks(x_pos)
ax.set_xticklabels([DISPLAY_NAMES[i] for i in sort_idx], rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Mean |SHAP|', fontsize=10, color='#222222')
ax.set_title('Main vs Interaction Effects (SHAP Decomposition)', fontsize=13,
             fontweight='bold', color=AZURE_ACCENT, pad=12)
ax.legend(fontsize=9, loc='upper right')
ax.grid(axis='y', alpha=0.3, ls='--')
ax.set_axisbelow(True)

# ML-SHAP: 每个 bar 顶部标注数值
for bar in bars_main:
    h = bar.get_height()
    if h > 0.01:
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.002, f'{h:.3f}',
                ha='center', va='bottom', fontsize=6.5, color='#222222')
for bar in bars_inter:
    h = bar.get_height()
    if h > 0.01:
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.002, f'{h:.3f}',
                ha='center', va='bottom', fontsize=6.5, color='#222222')

plt.tight_layout()
save_chart(fig, 'fig_e_main_vs_interaction')
```

**报告 §5.1 需要包含完整表格**:

| 特征 | 主效应 | 交互效应 | 交互/主效应比 |
|------|--------|----------|---------------|
| PM₂.₅ | 0.369 | 0.064 | 0.17 |
| ... | ... | ... | ... |
| NO₂ | 0.005 | 0.008 | 1.51 ← 交互>主效应, 需重点分析 |

## 6.3 图表 F: 交互矩阵 (蓝橙双色) [ML-SHAP 强化]

```python
from matplotlib.colors import LinearSegmentedColormap

vmax = np.abs(inter_mat).max()
# ML-SHAP: 中间色 #d5d8d3 (Azure 浅灰绿), 与 axes 白底可区分
cmap_inter = LinearSegmentedColormap.from_list('inter', [
    '#2166ac', '#92c5de', '#d5d8d3', '#f4a582', '#d6604d'
])

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(inter_mat, cmap=cmap_inter, aspect='auto', vmin=-vmax, vmax=vmax)

# ML-SHAP: 全尺寸标注 (11×11=121 个单元格), 不是仅 Top-15
for i in range(n_feat):
    for j in range(n_feat):
        val = inter_mat[i, j]
        text_color = 'white' if abs(val) > vmax * 0.55 else '#222222'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=7,
                color=text_color,
                fontweight='bold' if abs(val) > vmax * 0.7 else 'normal')

ax.set_xticks(range(n_feat))
ax.set_xticklabels(DISPLAY_NAMES, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(n_feat))
ax.set_yticklabels(DISPLAY_NAMES, fontsize=8)
ax.set_title('SHAP Interaction Matrix\nBlue (Antagonistic) / Orange (Synergistic)',
             fontsize=13, fontweight='bold', color=AZURE_ACCENT, pad=15)
ax.grid(False)
cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('Mean SHAP Interaction', fontsize=9, color='#222222')
plt.tight_layout()
save_chart(fig, 'fig_f_interaction_matrix')
```

## 6.4 图表 G: SHAP Heatmap

```python
fig, ax = plt.subplots(figsize=(13, 8))
shap.summary_plot(shap_values, X_shap, feature_names=DISPLAY_NAMES,
                  show=False, plot_type='layered_violin', max_display=11)
ax = plt.gca()
ax.grid(False)
save_chart(fig, 'fig_g_heatmap')
```

## 6.5 Bootstrap SHAP 置信区间

```python
N_BOOT = 25  # 测试模式 (论文用 200)
bootstrap_importances = np.zeros((N_BOOT, n_feat))

for b in range(N_BOOT):
    boot_idx = np.random.choice(len(X_shap), len(X_shap), replace=True)
    boot_shap = shap_values[boot_idx]
    bootstrap_importances[b] = np.abs(boot_shap).mean(axis=0)

ci_low = np.percentile(bootstrap_importances, 2.5, axis=0)
ci_high = np.percentile(bootstrap_importances, 97.5, axis=0)
mean_imp = bootstrap_importances.mean(axis=0)
sort_idx_ci = np.argsort(mean_imp)[::-1]

fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(np.arange(n_feat), mean_imp[sort_idx_ci],
        xerr=[mean_imp[sort_idx_ci] - ci_low[sort_idx_ci],
              ci_high[sort_idx_ci] - mean_imp[sort_idx_ci]],
        color=AZURE_MAIN, alpha=0.85, capsize=3, error_kw={'linewidth': 1})
ax.set_yticks(np.arange(n_feat))
ax.set_yticklabels([DISPLAY_NAMES[i] for i in sort_idx_ci], fontsize=9)
ax.set_xlabel('Mean |SHAP|', fontsize=10, color='#222222')
ax.set_title(f'SHAP Importance with 95% Bootstrap CI (n={N_BOOT})', fontsize=13,
             fontweight='bold', color=AZURE_ACCENT, pad=12)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3, ls='--')
ax.set_axisbelow(True)
plt.tight_layout()
save_chart(fig, 'fig_j_bootstrap_ci')
```

## 6.6 SHAP 稳定性诊断 [ML-SHAP 保留]

```python
def shap_stability_diagnosis(X_train, y_train, X_test, feature_names, n_models=5):
    """M=5 (测试) / M=10 (论文) 模型 → Rank CV > 0.3 → ⚠ 不稳定"""
    shap_importances = np.zeros((n_models, len(feature_names)))
    for i in range(n_models):
        m = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1,
                         colsample_bytree=0.6, subsample=0.8,
                         random_state=i, verbosity=0)
        m.fit(X_train, y_train)
        sv = shap.TreeExplainer(m).shap_values(X_test[:500])
        shap_importances[i] = np.abs(sv).mean(axis=0)

    rank_matrix = np.argsort(np.argsort(-shap_importances, axis=1), axis=1)
    rank_cv = rank_matrix.std(axis=0) / (rank_matrix.mean(axis=0) + 1e-10)
    unstable = [feature_names[j] for j in range(len(feature_names))
                if rank_cv[j] > 0.3]
    return rank_cv, unstable
```

## 6.7 SHAP 聚类子群体分析

```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(shap_values)

pca = PCA(n_components=2)
shap_pca = pca.fit_transform(shap_values)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
cluster_colors = ['#5b8c85', '#c1724a', '#7b6b8e', '#d6604d']

# (a) PCA 散点
ax = axes[0, 0]
for c in range(n_clusters):
    mask = cluster_labels == c
    ax.scatter(shap_pca[mask, 0], shap_pca[mask, 1],
               c=cluster_colors[c], label=f'Cluster {c+1}', alpha=0.6, s=10)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=9)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=9)
ax.set_title('SHAP Value Clusters (PCA)', fontsize=11, fontweight='bold')
ax.legend(fontsize=7)

# (b)-(d) 每个聚类的平均 SHAP 模式
for c in range(min(3, n_clusters)):
    ax = axes[(c + 1) // 2, (c + 1) % 2]
    cluster_shap = shap_values[cluster_labels == c]
    cluster_mean = np.abs(cluster_shap).mean(axis=0)
    sort_idx_c = np.argsort(cluster_mean)[::-1]
    ax.barh(range(n_feat), cluster_mean[sort_idx_c], color=cluster_colors[c], alpha=0.85)
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels([DISPLAY_NAMES[i] for i in sort_idx_c], fontsize=7)
    ax.set_title(f'Cluster {c+1} (n={(cluster_labels == c).sum()})', fontsize=11)
    ax.invert_yaxis()

plt.tight_layout()
save_chart(fig, 'fig_k_shap_clusters')
```

## 6.8 输出

- `Charts/fig_e_main_vs_interaction.{svg,png}` — ML-SHAP 修正交互计算
- `Charts/fig_f_interaction_matrix.{svg,png}` — 蓝橙双色 + 全单元格标注
- `Charts/fig_g_heatmap.{svg,png}`
- `Charts/fig_j_bootstrap_ci.{svg,png}`
- `Charts/fig_k_shap_clusters.{svg,png}` — 2×2 面板
- `Tables/interaction_matrix.csv` — 特征名使用 Unicode 下标
- `Cache/interaction_values.npy` — tree_path_dependent explainer 计算
- `Cache/bootstrap_ci.npy` — stack([mean_imp, ci_low, ci_high], axis=0)

## ML-SHAP 关键变更清单

| Earlier pattern | ML-SHAP pattern |
|------|------|
| `(off_diag - diag) / 2` 交互计算 (错误) | 逐样本分解 `sum_j(inter[n,i,j]) - diag` |
| `fix_shap_text(ax)` 修复文本 | `save_chart()` 内 `sanitize_svg_u2212()` 防御性清除 |
| `feature_names=RAW_NAMES` (ASCII) | `feature_names=DISPLAY_NAMES` (Unicode 下标) |
| Bootstrap 200 次 (43min) | 25 次测试 / 200 次论文 |
| 交互矩阵 Top-15 截断 | 全 11×11 矩阵 |
