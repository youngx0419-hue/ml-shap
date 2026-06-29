---
name: evaluation
description: 全面模型评估模块 —— K-fold CV、学习曲线、残差诊断、分类指标、校准曲线。产出完整的模型评估报告。
---

# Phase 4：全面模型评估

## 4.0 评估设计必须匹配数据依赖结构

先读取 `method_decision_log.json`：

- 时间预测任务使用 forward-chaining 或 rolling-origin 验证，不随机打乱。
- 空间数据使用 spatial/environmental block CV，随机 CV 只能作为乐观敏感性分析。
- 同一主体、城市、站点、实验批次或文献来源内有多条样本时使用 GroupKFold 或 grouped holdout。
- 分类概率用于风险分层、预警或决策阈值时，必须报告 calibration curve、Brier score，并说明是否进行了 Platt/isotonic 校准。
- 任何目标变换都要分开报告变换尺度和原始尺度指标。

## 4.1 回归任务评估

### K-Fold 交叉验证
```python
from sklearn.model_selection import KFold, cross_validate

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(best_model, X_train, y_train, cv=kf,
                        scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'],
                        return_train_score=True)

print(f"Train R²:  {scores['train_r2'].mean():.4f} ± {scores['train_r2'].std():.4f}")
print(f"Test  R²:  {scores['test_r2'].mean():.4f} ± {scores['test_r2'].std():.4f}")
print(f"Test  RMSE: {np.sqrt(-scores['test_neg_mean_squared_error']).mean():.2f}")
print(f"Test  MAE:  {-scores['test_neg_mean_absolute_error'].mean():.2f}")
```

### 学习曲线（诊断过拟合/欠拟合）
```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    best_model, X_train, y_train, cv=5,
    scoring='neg_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 10),
    random_state=42, n_jobs=-1
)

# 绘制带 ±1σ 区间的学习曲线
fig, ax = plt.subplots(figsize=(4.5, 3.5))
ax.plot(train_sizes, -train_scores.mean(axis=1), 'o-', color='#2255a4', label='Training')
ax.fill_between(train_sizes, -train_scores.mean(axis=1) - train_scores.std(axis=1),
                -train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.15, color='#2255a4')
ax.plot(train_sizes, -test_scores.mean(axis=1), 'o-', color='#d62828', label='CV Test')
ax.fill_between(train_sizes, -test_scores.mean(axis=1) - test_scores.std(axis=1),
                -test_scores.mean(axis=1) + test_scores.std(axis=1), alpha=0.15, color='#d62828')
ax.set_xlabel('Training set size'); ax.set_ylabel('RMSE')
ax.legend()
```

**诊断逻辑：**
- 训练误差 ≪ 测试误差且差距不随样本增大而缩小 → **过拟合**
- 训练误差和测试误差都大且接近 → **欠拟合**
- 两条曲线收敛且接近 → **良好拟合**

### 残差诊断（四面板图）
```python
fig, axes = plt.subplots(2, 2, figsize=(8, 7))

# (a) 残差 vs 拟合值 —— 检查异方差
residuals = y_test - y_pred
axes[0, 0].scatter(y_pred, residuals, alpha=0.4, s=8, rasterized=True)
axes[0, 0].axhline(y=0, color='#d62828', lw=1.2, ls='--')
axes[0, 0].set_xlabel('Fitted values'); axes[0, 0].set_ylabel('Residuals')

# (b) Q-Q 图 —— 检查正态性
from scipy.stats import probplot
probplot(residuals, dist='norm', plot=axes[0, 1])

# (c) 实测 vs 预测散点图 + 1:1 线 + 拟合线
axes[1, 0].scatter(y_test, y_pred, alpha=0.4, s=8, rasterized=True)
axes[1, 0].plot([y_min, y_max], [y_min, y_max], 'k--', lw=1.2)

# (d) 残差直方图 + KDE
axes[1, 1].hist(residuals, bins=40, density=True, alpha=0.6, color='#2255a4')
from scipy.stats import gaussian_kde
kde = gaussian_kde(residuals)
x_kde = np.linspace(residuals.min(), residuals.max(), 200)
axes[1, 1].plot(x_kde, kde(x_kde), 'r-', lw=1.5)
```

### 回归指标汇总
```python
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                              explained_variance_score, mean_absolute_percentage_error)

metrics = {
    'R²': r2_score(y_test, y_pred),
    'Explained Variance': explained_variance_score(y_test, y_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
    'MAE': mean_absolute_error(y_test, y_pred),
    'MAPE': mean_absolute_percentage_error(y_test, y_pred) * 100,
}
```

## 4.2 分类任务评估

### 混淆矩阵
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

fig, axes = plt.subplots(1, 2, figsize=(9, 4))

# 原始计数
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[0], cmap='Blues')
axes[0].set_title('Confusion Matrix (Counts)')

# 行归一化
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[1],
    cmap='Blues', normalize='true')
axes[1].set_title('Confusion Matrix (Row-normalized)')
```

### ROC 曲线（多分类用 OvR）
```python
from sklearn.metrics import roc_curve, auc, RocCurveDisplay

# 二分类
RocCurveDisplay.from_predictions(y_test, y_pred_proba[:, 1], ax=ax)

# 多分类 OvR
from sklearn.preprocessing import label_binarize
y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=1.5, label=f'{class_names[i]} (AUC={roc_auc:.3f})')
```

### 校准曲线（可靠性图）
```python
from sklearn.calibration import calibration_curve

fig, ax = plt.subplots(figsize=(4, 3.5))
for name, proba in [('XGBoost', y_proba_xgb), ('LightGBM', y_proba_lgb)]:
    prob_true, prob_pred = calibration_curve(y_test, proba[:, 1], n_bins=10)
    ax.plot(prob_pred, prob_true, 'o-', lw=1.5, label=name)
ax.plot([0, 1], [0, 1], 'k--', lw=0.8, label='Perfectly calibrated')
ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Fraction of positives')
```

### 分类指标汇总
```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
```

## 4.3 模型对比总结表

生成一个综合对比表：

| 指标 | XGBoost | LightGBM | CatBoost | RF |
|------|---------|----------|----------|-----|
| CV R² (mean±std) | | | | |
| Test R² | | | | |
| Test RMSE | | | | |
| Test MAE | | | | |
| Train Time (s) | | | | |
| Overfit Gap (Train-Test R²) | | | | |

## 4.4 输出

评估图表统一保存到 `05_Results/<项目名>_v<N>/Charts/` (按 ML-SHAP 统一规范)：
- `fig_i_diagnostics.{svg,png}` — 残差四面板图 (Phase 4 核心产出)
- `fig_learning_curve.{svg,png}` — 学习曲线 (可选)
- `05_Results/<项目名>_v<N>/Tables/`:
  - `model_benchmark.csv` — 模型基准对比表
  - `predictions.csv` — 预测值与残差
