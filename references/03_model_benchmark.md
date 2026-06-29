---
name: model-benchmark
description: 多模型基准对比与超参数调优模块 —— 自动训练 XGBoost/LightGBM/CatBoost/Random Forest 并进行公平对比，对最佳模型执行 Optuna 超参数调优。
---

# Phase 3：多模型基准对比与调优

## 3.0 文献驱动的模型候选集选择

在训练前先读取 `method_decision_log.json`。候选模型不是固定清单，而是由数据结构和使用目标决定：

- 一般结构化表格：XGBoost + LightGBM + CatBoost + Random Forest + 简单基线。
- 大样本或高维数据：必须加入 LightGBM，记录运行时间和性能差异。
- 高基数类别变量：CatBoost 优先；XGBoost/LightGBM 只能使用 fold 内安全编码。
- 类别不平衡：优先 class weight、阈值调优、PR-AUC；SMOTE 只能放在训练 fold 内。
- 高风险决策：加入可解释基线（线性/广义加性/规则模型），并报告黑箱模型的性能增益是否足以抵消解释成本。
- 科学解释目标：优先稳定性和跨模型一致性，不以单次最优分数作为唯一选择标准。

## 3.1 四模型基准对比

```python
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score

MODELS = {
    'XGBoost': XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=42, verbosity=0, n_jobs=-1),
    'LightGBM': LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               random_state=42, verbose=-1, n_jobs=-1),
    'CatBoost': CatBoostRegressor(n_estimators=300, depth=6, learning_rate=0.05,
                                   subsample=0.8, random_seed=42,
                                   verbose=0, thread_count=-1),
    'RandomForest': RandomForestRegressor(n_estimators=300, max_depth=12,
                                           random_state=42, n_jobs=-1),
}

# 5-fold CV 对比
results = []
for name, model in MODELS.items():
    scores = cross_val_score(model, X_train, y_train, cv=5,
                             scoring='neg_mean_squared_error')
    results.append({
        'Model': name,
        'CV_RMSE_mean': np.sqrt(-scores.mean()),
        'CV_RMSE_std': np.sqrt(-scores).std(),
        'CV_R2_mean': cross_val_score(model, X_train, y_train, cv=5, scoring='r2').mean(),
    })
```

**关键**：所有模型必须使用相同的 CV 折叠（固定 `random_state`），确保对比公平。

## 3.2 特征重要性一致性分析

比较各模型 Top-20 特征的一致性：

```python
from scipy.stats import kendalltau

# 提取各模型的特征重要性
importance_dict = {}
for name, model in trained_models.items():
    imp = model.feature_importances_
    importance_dict[name] = pd.Series(imp, index=feature_names).sort_values(ascending=False)

# 计算 Kendall τ 排序相关性
for m1, m2 in combinations(importance_dict.keys(), 2):
    tau, p = kendalltau(importance_dict[m1].index[:20], importance_dict[m2].index[:20])
```

如果 Kendall τ > 0.7，说明模型间特征重要性排序高度一致，SHAP 分析的结论具有模型鲁棒性。

## 3.3 超参数调优（Optuna）

选择 CV 表现最好的 2 个模型进行调优：

```python
import optuna

def objective(trial, model_name, X, y):
    if model_name == 'XGBoost':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
        }
        model = XGBRegressor(**params, random_state=42, verbosity=0, n_jobs=-1,
                             early_stopping_rounds=50)
    # ... LightGBM, CatBoost 类似

    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    return -scores.mean()

# 50+ trials
study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, 'XGBoost', X_train, y_train),
               n_trials=50, show_progress_bar=True)
```

**分类任务**同样逻辑，将 scoring 改为 `'neg_log_loss'` 或 `'roc_auc_ovr'`。

## 3.4 Stacking 集成（ML-SHAP 新增）

当用户要求最高预测精度时，在 4 个基准模型训练完成后构建 Stacking 集成：

```python
from sklearn.ensemble import StackingRegressor, StackingClassifier
from sklearn.linear_model import Ridge, LogisticRegression

def build_stacking_ensemble(trained_models, X, y, task_type='regression'):
    """使用已训练的 4 个模型作为基学习器，构建 Stacking 集成"""
    estimators = [(name.lower(), model) for name, model in trained_models.items()]

    if task_type == 'regression':
        stacking = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=5, n_jobs=-1
        )
    else:
        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(C=1.0, max_iter=1000),
            cv=5, stack_method='predict_proba', n_jobs=-1
        )

    stacking.fit(X, y)
    return stacking

# 对比 Stacking vs 最佳单模型
stacking_scores = cross_val_score(stacking, X, y, cv=5, scoring='r2')
print(f"Stacking CV R²: {stacking_scores.mean():.4f} ± {stacking_scores.std():.4f}")
print(f"Best Single CV R²: {best_single_score:.4f}")
print(f"Improvement: {(stacking_scores.mean() - best_single_score) * 100:.2f}%")
```

## 3.5 贝叶斯优化（ML-SHAP 新增，作为 Optuna 的替代方案）

对于小样本场景（N < 500），贝叶斯优化通常比 Optuna 更高效：

```python
from bayes_opt import BayesianOptimization

def bo_xgboost_cv(n_estimators, max_depth, learning_rate, subsample,
                  colsample_bytree, min_child_weight, gamma,
                  reg_alpha, reg_lambda):
    """BO 目标函数"""
    params = {
        'n_estimators': int(n_estimators), 'max_depth': int(max_depth),
        'learning_rate': learning_rate, 'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'min_child_weight': int(min_child_weight),
        'gamma': gamma, 'reg_alpha': reg_alpha, 'reg_lambda': reg_lambda,
        'random_state': 42, 'verbosity': 0, 'n_jobs': -1
    }
    model = XGBRegressor(**params)
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    return -np.sqrt(-scores.mean())

# 仅在样本量 < 500 时推荐
if len(X) < 500:
    print("样本量较小(<500)，推荐使用贝叶斯优化替代 Optuna")
```

## 3.6 最终模型选择 [ML-SHAP 更新]

### 3.6.1 SHAP 兼容性约束

**ML-SHAP 强制**: interventional SHAP 模式不支持 RandomForest。
若 RF 综合得分最高, **必须从梯度提升模型中选最优**:

```python
# ML-SHAP: RandomForest + interventional SHAP → additivity check 失败
tree_models = ['XGBoost', 'LightGBM', 'CatBoost']
best_gb = benchmark_df[benchmark_df['Model'].isin(tree_models)]
best_gb = best_gb.sort_values('Test_R2', ascending=False).iloc[0]

rf_r2 = benchmark_df[benchmark_df['Model']=='RandomForest']['Test_R2'].values[0]
if rf_r2 > best_gb['Test_R2']:
    print(f'Note: RandomForest (R2={rf_r2:.4f}) cannot use interventional SHAP')
    print(f'Falling back to {best_gb["Model"]} (R2={best_gb["Test_R2"]:.4f})')
```

### 3.6.2 综合评分标准

基于以下标准推荐最佳模型：

| 标准 | 权重 | 说明 |
|------|------|------|
| CV 性能（均值） | 50% | 主要指标 |
| CV 稳定性（标准差） | 20% | 越小越好 |
| 训练时间 | 15% | 实用考量 |
| 特征重要性一致性 | 15% | 与其他模型的一致性 |

## 3.7 输出

```python
# 保存
save_model(best_model, dirs, "best_model.pkl")
results_df.to_csv(os.path.join(table_dir, "model_benchmark.csv"), index=False)

# 打印对比表 (ML-SHAP: 使用 ASCII 避免 GBK crash)
print("Model Benchmark (5-fold CV):")
print(results_df.to_string(index=False))
print("\nBest (GB for SHAP): %s (Test R2=%.4f)" % (best_name, best_r2))
```

## 3.6 早停策略

所有梯度提升模型使用 `early_stopping_rounds=50`，配合 `eval_set`：
```python
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          early_stopping_rounds=50,
          verbose=False)
```
