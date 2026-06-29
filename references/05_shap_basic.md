---
name: shap-basic
description: Phase 5 基础 SHAP 分析 —— Times New Roman 字体 + Unicode 化学下标 + 手动 barh 瀑布图 + 自动洞察提取 (identify key features, nonlinear patterns, interaction signals, anomaly samples)
---

# Phase 5：基础 SHAP 分析与自动洞察 [ML-SHAP]

## 5.0a 文献驱动解释策略

先读取 `method_decision_log.json`。SHAP 不是自动等于真因果解释：

- 记录 TreeExplainer 的 `feature_perturbation`、background 数据来源、样本数、模型输出尺度和随机种子。
- 强相关特征（如 `max_abs_correlation >= 0.7`）必须在报告中加入依赖 caveat；优先按特征族解释，并用 ALE 或分组解释辅助。
- 全局重要性至少与验证集 permutation importance 或跨模型排名稳定性做一次对照。
- 个体 waterfall 只能解释该样本的模型输出，不推广为总体规律。
- 分类风险解释应基于校准后的概率或明确说明未校准。

## 5.0 前置: 字体 + 下标 [ML-SHAP 强制]

```python
# 参见references/style_constants.md and scripts/xai_style_helpers.py
# 必须: Times New Roman + svg.fonttype='path' + chem_sub() 下标
DISPLAY_NAMES = [chem_sub(n) for n in RAW_NAMES]
# 所有 SHAP 绘图传入 feature_names=DISPLAY_NAMES
# print() 禁止使用 DISPLAY_NAMES (GBK crash), 用 RAW_NAMES
```

## 5.1 SHAP 值计算

```python
import shap

# ML-SHAP: interventional 模式 (主) — 关联 SHAP 值 + Beeswarm + Bar + Dependence
X_bg = X_train[np.random.choice(len(X_train), min(200, len(X_train)), replace=False)]
explainer = shap.TreeExplainer(best_model, feature_perturbation='interventional', data=X_bg)
shap_values = explainer.shap_values(X_shap, check_additivity=False)

mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

# 抽样策略: N > 3000 抽样到 3000
```

## 5.2 图表 A: 出版级蜂群图 [ML-SHAP 更新]

```python
fig, ax = plt.subplots(figsize=(7, 5))

shap.summary_plot(shap_values, X_shap, feature_names=DISPLAY_NAMES,
                  show=False, plot_type='dot')
ax = plt.gca()
ax.set_title('SHAP Beeswarm: AQI Driver Analysis', fontsize=13, fontweight='bold',
             color=AZURE_ACCENT, pad=12)
save_chart(fig, 'fig_a_beeswarm')
# save_chart() 自动执行 sanitize_svg_u2212()
```

**V1.1 变更**: 去除 `fix_shap_text()` 调用 (v1.0 遗留)。Times New Roman + svg.fonttype='path' 确保字体正确。U+2212 由 `save_chart()` 内部 `sanitize_svg_u2212()` 防御性清除。

## 5.3 图表 B: 特征重要性条形图

```python
fig, ax = plt.subplots(figsize=(7, 5))
shap.summary_plot(shap_values, X_shap, feature_names=DISPLAY_NAMES,
                  show=False, plot_type='bar')
ax = plt.gca()
ax.set_title('SHAP Feature Importance (mean |SHAP|)', fontsize=13,
             fontweight='bold', color=AZURE_ACCENT, pad=12)
save_chart(fig, 'fig_b_importance')
```

## 5.4 图表 C: 瀑布图 — 手动 barh [ML-SHAP CRITICAL — 替代 SHAP 内置]

**SHAP 内置 `waterfall_plot()` 被彻底弃用** (U+2212 tofu 无法修复)。
ML-SHAP 使用纯 matplotlib barh 手动绘制:

```python
def manual_waterfall(shap_row, base_val, x_data, feature_names, title, filename):
    """纯 matplotlib barh 瀑布图 — 0 个 U+2212, 完全字体控制"""
    n = len(feature_names)
    idx_sorted = np.argsort(np.abs(shap_row))[::-1]
    sorted_names = [feature_names[i] for i in idx_sorted]
    sorted_shap = shap_row[idx_sorted]

    # 累积和
    cumsum = np.zeros(n + 1)
    cumsum[0] = base_val
    for i in range(n):
        cumsum[i + 1] = cumsum[i] + sorted_shap[i]

    fig, ax = plt.subplots(figsize=(7, n * 0.35 + 1.5))
    colors = [AZURE_MAIN if v >= 0 else '#c1724a' for v in sorted_shap]

    for i in range(n):
        ax.barh(i, sorted_shap[i], 0.6, left=cumsum[i],
                color=colors[i], alpha=0.85, edgecolor='white', linewidth=0.5)

    ax.axvline(x=base_val, color='#444444', linewidth=1, linestyle='--', alpha=0.6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.set_xlabel('SHAP Value (log1p scale)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold', color=AZURE_ACCENT, pad=12)

    # 数值标注 (ASCII hyphen, 永不产生 U+2212)
    for i in range(n):
        val = sorted_shap[i]
        val_str = f'{val:+.3f}'  # + 或 ASCII -
        x_pos = cumsum[i] + val
        if val >= 0:
            ax.text(x_pos + 0.003, i, val_str, va='center', fontsize=7.5, color='#222222')
        else:
            ax.text(x_pos - 0.003, i, val_str, va='center', ha='right', fontsize=7.5, color='#222222')

    # E[f(x)] 和 f(x) 标注
    final_val = cumsum[-1]
    ax.text(base_val, -0.8, f'E[f(x)] = {base_val:.3f}', ha='center', fontsize=8,
            color='#444444', style='italic')
    ax.text(final_val, n - 0.2, f'f(x) = {final_val:.3f}', ha='center', fontsize=8,
            color='#444444', style='italic', fontweight='bold')

    ax.set_ylim(-1.2, n)
    ax.grid(axis='x', alpha=0.3, ls='--')
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_chart(fig, filename)

# 边界样本选择
y_pred_shap = best_model.predict(X_shap)
idx_highest = np.argmax(y_pred_shap)
idx_lowest = np.argmin(y_pred_shap)
idx_boundary = np.argmin(np.abs(y_pred_shap - np.median(y_pred_shap)))

base_val = float(explainer.expected_value) if not isinstance(explainer.expected_value, (list, np.ndarray)) else float(np.mean(explainer.expected_value))

samples = [
    (idx_highest, 'Highest AQI', 'fig_c_waterfall_highest'),
    (idx_lowest, 'Lowest AQI', 'fig_c_waterfall_lowest'),
    (idx_boundary, 'Median AQI (Decision Boundary)', 'fig_c_waterfall_boundary'),
]
for idx, label, fname in samples:
    pred_val = y_pred_shap[idx]
    pred_orig = np.expm1(pred_val) if TRANSFORM == 'log1p' else pred_val
    manual_waterfall(shap_values[idx], base_val, X_shap[idx],
                     DISPLAY_NAMES,
                     f'Waterfall: {label}  |  Predicted AQI = {pred_orig:.0f}',
                     fname)
```

## 5.5 图表 D: 依赖图 (Top 4 特征) [ML-SHAP 简化]

```python
shap_importance = np.abs(shap_values).mean(axis=0)
top4_idx = np.argsort(shap_importance)[::-1][:4]

for feat_idx in top4_idx:
    feat_disp = DISPLAY_NAMES[feat_idx]
    safe_name = RAW_NAMES[feat_idx].replace('.', '_').replace(' ', '_')

    fig, ax = plt.subplots(figsize=(7, 5))
    shap.dependence_plot(feat_idx, shap_values, X_shap,
                         feature_names=DISPLAY_NAMES, show=False, ax=ax)
    ax = plt.gca()
    ax.set_title(f'SHAP Dependence: {feat_disp}', fontsize=13,
                 fontweight='bold', color=AZURE_ACCENT, pad=12)
    ax.grid(axis='y', alpha=0.3, ls='--')
    ax.set_axisbelow(True)
    save_chart(fig, f'fig_d_dependence_{safe_name}')
```

## 5.6 自动洞察 JSON

```json
{
  "top_features": [
    {"rank": 1, "feature": "PM2.5", "mean_abs_shap": 0.2647, "direction": "positive"}
  ],
  "feature_ranking": ["PM2.5", "PM10", ...],
  "top_feature": "PM2.5",
  "top_shap_importance": 0.2647,
  "unstable_features": ["PM10"],
  "rank_cv": [0.0, 0.333, ...]
}
```

## 5.7 输出

- `Charts/fig_a_beeswarm.{svg,png}` — 蜂群图 (Times New Roman + 下标)
- `Charts/fig_b_importance.{svg,png}` — 重要性条形图
- `Charts/fig_c_waterfall_highest.{svg,png}` — 最高预测瀑布图 (手动 barh)
- `Charts/fig_c_waterfall_lowest.{svg,png}` — 最低预测瀑布图 (手动 barh)
- `Charts/fig_c_waterfall_boundary.{svg,png}` — 决策边界瀑布图 (手动 barh)
- `Charts/fig_d_dependence_{feature}.{svg,png}` — 依赖图 × 4
- `Tables/shap_values.csv` — 含 Raw_Name + Display_Name 两列
- `Cache/auto_insights.json`

**ML-SHAP 移除**: fix_shap_text() 调用 / SHAP 内置 waterfall_plot() / best-worst-typical 命名
