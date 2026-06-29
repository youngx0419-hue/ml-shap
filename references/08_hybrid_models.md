---
name: hybrid-models
description: 混合模型增强模块 —— LSTM-XGBoost 时序预测、BO-XGBoost 贝叶斯优化、SMOTE-XGBoost 不平衡处理、VMD-XGBoost 信号分解、Stacking 多模型集成。来源于 90+ 篇论文中高频使用的混合策略。
---

# Phase 8：混合模型增强策略

## 8.0 策略选择决策树

```
数据有时间列 + ≥24 时间点？
├── 是 → 非平稳性检验（ADF）
│   ├── 非平稳 → VMD-XGBoost（对原始序列做 VMD 分解）
│   └── 平稳 → LSTM-XGBoost（LSTM 提取时序特征 + XGBoost 回归）
└── 否 ↓

目标变量类别不平衡 > 10:1？
├── 是 → SMOTE-XGBoost
└── 否 ↓

特征维度 > 50 且样本 < 1000？
├── 是 → BO-XGBoost（贝叶斯优化在少样本下比 Optuna 更高效）
└── 否 → 标准 XGBoost + Optuna 调优

用户要求提升预测精度 + 有预算？
└── → Stacking 集成（XGBoost + LightGBM + CatBoost → Ridge 元学习器）
```

---

## 8.1 LSTM-XGBoost 混合模型

**适用场景**：碳排放时间序列预测、水质参数时序预测

**代表论文**：龚笑雨(长三角碳排放)、谢云杰(江西省碳排放)、林子琪(臭氧浓度)

### 8.1.1 架构设计

```
原始时序数据
    │
    ├──→ LSTM 分支：捕获长期时序依赖
    │        Input → LSTM(64) → Dropout(0.2) → LSTM(32) → Dense(16)
    │                                                      │
    ├──→ 特征工程分支：时间特征 + 滞后特征 + 统计特征          │
    │        [lag1, lag2, lag3, rolling_mean_7d, ...]        │
    │                                                      │
    └──→ XGBoost：融合 LSTM 隐藏状态 + 手工特征 + 外生变量
             XGBoost(n_estimators=200, max_depth=6) → 最终预测
```

### 8.1.2 实现代码

```python
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate
from xgboost import XGBRegressor

def build_lstm_xgboost(X_seq, X_static, y, seq_len=12):
    """
    X_seq:   时序特征 (N, seq_len, n_seq_features)
    X_static: 静态特征 (N, n_static_features)
    """
    # Step 1: LSTM 时序编码
    seq_input = Input(shape=(seq_len, X_seq.shape[2]), name='sequence_input')
    lstm1 = LSTM(64, return_sequences=True)(seq_input)
    drop1 = Dropout(0.2)(lstm1)
    lstm2 = LSTM(32, return_sequences=False)(drop1)
    lstm_features = Dense(16, activation='relu', name='lstm_encoding')(lstm2)

    lstm_encoder = Model(inputs=seq_input, outputs=lstm_features)
    lstm_encoded = lstm_encoder.predict(X_seq, verbose=0)

    # Step 2: 融合特征
    X_fused = np.hstack([lstm_encoded, X_static])

    # Step 3: XGBoost 最终预测
    xgb = XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0
    )
    xgb.fit(X_fused, y)
    return lstm_encoder, xgb

def predict_lstm_xgboost(lstm_encoder, xgb, X_seq, X_static):
    lstm_encoded = lstm_encoder.predict(X_seq, verbose=0)
    X_fused = np.hstack([lstm_encoded, X_static])
    return xgb.predict(X_fused)
```

### 8.1.3 SHAP 解释策略

LSTM-XGBoost 的 SHAP 解释通过 XGBoost 部分实现（LSTM 隐藏状态作为输入特征之一），需额外分析：

```python
# SHAP 解释（XGBoost 部分）
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_fused)

# LSTM 特征重要性分析
lstm_feature_names = [f'lstm_enc_{i}' for i in range(16)]
static_feature_names = X_static.columns.tolist()
all_feature_names = lstm_feature_names + static_feature_names

# 分析 LSTM 编码特征的 SHAP 贡献总和
lstm_total_shap = np.abs(shap_values[:, :16]).sum(axis=1).mean()
```

---

## 8.2 BO-XGBoost 贝叶斯优化

**适用场景**：高维小样本、调参预算有限

**代表论文**：Pan & Wu (成都住宅碳排放)、赵敬皓 (煤电机组)、SCI 多篇

### 8.2.1 贝叶斯优化 vs Optuna

| 对比项 | Optuna (TPE) | Bayesian Optimization |
|--------|-------------|----------------------|
| 小样本（<500） | 中等 | **优** |
| 大样本（>5000） | **优** | 慢 |
| 高维搜索（>20 参数） | **优** | 中等 |
| 离散+连续混合空间 | **优** | 仅连续 |
| 可解释的采集函数 | 无 | 有（EI/UCB） |

结论：小样本、低维搜索空间用 BO；大样本、高维搜索用 Optuna。论文中两者常并存。

### 8.2.2 实现代码

```python
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score

def bo_optimize_xgboost(X, y, init_points=10, n_iter=30):
    """贝叶斯优化 XGBoost 超参数"""

    def xgb_cv(n_estimators, max_depth, learning_rate, subsample,
               colsample_bytree, min_child_weight, gamma, reg_alpha, reg_lambda):
        params = {
            'n_estimators': int(n_estimators),
            'max_depth': int(max_depth),
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'min_child_weight': int(min_child_weight),
            'gamma': gamma,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': 42, 'verbosity': 0, 'n_jobs': -1
        }
        model = XGBRegressor(**params)
        scores = cross_val_score(model, X, y, cv=5,
                                 scoring='neg_mean_squared_error')
        return -np.sqrt(-scores.mean())  # 返回负 RMSE（最大化=最小化 RMSE）

    pbounds = {
        'n_estimators': (50, 500),
        'max_depth': (3, 12),
        'learning_rate': (0.01, 0.3),
        'subsample': (0.5, 1.0),
        'colsample_bytree': (0.5, 1.0),
        'min_child_weight': (1, 20),
        'gamma': (0, 5),
        'reg_alpha': (0, 10),
        'reg_lambda': (0, 10),
    }

    optimizer = BayesianOptimization(
        f=xgb_cv, pbounds=pbounds, random_state=42
    )
    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    # 最优参数
    best_params = optimizer.max['params']
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['min_child_weight'] = int(best_params['min_child_weight'])

    return best_params, optimizer
```

---

## 8.3 SMOTE-XGBoost 不平衡处理

**适用场景**：灾害/风险数据、稀有事件预测

**代表论文**：SCI 多篇（火灾风险、洪水风险、滑坡易发性）

### 8.3.1 完整流程

```python
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline

def smote_xgboost_pipeline(X, y, smote_method='borderline'):
    """SMOTE 变体 + XGBoost 完整流程"""

    # 选择 SMOTE 变体
    if smote_method == 'standard':
        sampler = SMOTE(random_state=42, k_neighbors=5)
    elif smote_method == 'borderline':
        sampler = BorderlineSMOTE(random_state=42, k_neighbors=5)
    elif smote_method == 'adasyn':
        sampler = ADASYN(random_state=42, n_neighbors=5)

    # 流水线
    pipeline = ImbPipeline([
        ('smote', sampler),
        ('xgb', XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, random_state=42,
            use_label_encoder=False, eval_metric='logloss', verbosity=0
        ))
    ])

    # 注意：SMOTE 只能在训练集上操作，用交叉验证包裹
    from sklearn.model_selection import StratifiedKFold, cross_validate
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 手动实现防泄漏的 SMOTE CV
    scores = []
    for train_idx, test_idx in skf.split(X, y):
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

        X_resampled, y_resampled = sampler.fit_resample(X_train_fold, y_train_fold)
        model = XGBClassifier(...)
        model.fit(X_resampled, y_resampled)
        scores.append(model.score(X_test_fold, y_test_fold))

    return model, np.mean(scores), np.std(scores)
```

### 8.3.2 样本权重替代方案

当 SMOTE 效果不佳时，使用 XGBoost 内置的 `scale_pos_weight`：

```python
# 自动计算
neg_count = (y == 0).sum()
pos_count = (y == 1).sum()
scale_pos_weight = neg_count / pos_count

model = XGBClassifier(scale_pos_weight=scale_pos_weight, ...)
```

---

## 8.4 VMD-XGBoost 信号分解

**适用场景**：非平稳时间序列（碳排放、能源消耗、水文序列）

**代表论文**：崔光耀 (矩形顶管隧道碳排放)

### 8.4.1 VMD 分解原理

变分模态分解（VMD）将原始序列分解为 K 个不同频率的 IMF（本征模态函数）：

```
原始序列 = IMF_1 (高频噪声) + IMF_2 (周期波动) + ... + IMF_K (长期趋势)
```

### 8.4.2 实现代码

```python
from vmdpy import VMD

def vmd_decompose(series, K=5, alpha=2000, tau=0, DC=0, init=1, tol=1e-7):
    """
    VMD 分解
    K:      模态数（建议用中心频率法确定，避免模态混叠）
    alpha:  带宽约束（越大 = 越窄带）
    """
    u, u_hat, omega = VMD(series, alpha, tau, K, DC, init, tol)
    # u: (K, N) 各模态分量
    imfs = pd.DataFrame(u.T, columns=[f'IMF_{i+1}' for i in range(K)])
    return imfs

def vmd_xgboost_pipeline(df, target_col, time_col, K=5):
    """
    VMD-XGBoost 完整流程：
    1. VMD 分解目标序列为 K 个 IMF
    2. 对每个 IMF 独立训练 XGBoost
    3. 最终预测 = Σ IMF 预测值
    """
    series = df[target_col].values
    imfs = vmd_decompose(series, K=K)

    models = {}
    preds_imf = {}
    for i, imf_col in enumerate(imfs.columns):
        # 对每个 IMF 训练独立模型
        X_features = df.drop(columns=[target_col, time_col]).values
        y_imf = imfs[imf_col].values

        model = XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_features, y_imf)
        models[imf_col] = model
        preds_imf[imf_col] = model.predict(X_features)

    # 重构预测 = Σ IMF 预测
    final_pred = np.sum([preds_imf[c] for c in imfs.columns], axis=0)
    return models, imfs, final_pred
```

### 8.4.3 K 值选择

```python
# 中心频率法：检查各 IMF 的中心频率是否充分分离
def determine_optimal_K(series, K_max=8):
    """当相邻 IMF 中心频率之比 < 1.1 时，说明模态混叠，K 过大"""
    from vmdpy import VMD
    for K in range(2, K_max + 1):
        u, u_hat, omega = VMD(series, 2000, 0, K, 0, 1, 1e-7)
        ratios = omega[1:] / omega[:-1]
        if any(r < 1.1 for r in ratios):
            return K - 1
    return K_max
```

---

## 8.5 Stacking 多模型集成

**适用场景**：追求最高预测精度，预算充足

**代表论文**：王城业 (辽宁省碳排放 Stacking)

### 8.5.1 实现代码

```python
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge, LogisticRegression

def build_stacking_ensemble(X, y, task_type='regression'):
    """构建双层 Stacking 集成"""

    # 第一层：基础学习器（多样化）
    base_estimators = [
        ('xgb', XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                             subsample=0.8, random_state=42, verbosity=0)),
        ('lgb', LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                              subsample=0.8, random_state=42, verbose=-1)),
        ('cat', CatBoostRegressor(n_estimators=200, depth=6, learning_rate=0.1,
                                  subsample=0.8, random_seed=42, verbose=0)),
        ('rf',  RandomForestRegressor(n_estimators=200, max_depth=15,
                                      random_state=42)),
    ]

    # 第二层：元学习器（简单模型，防过拟合）
    meta_learner = Ridge(alpha=1.0) if task_type == 'regression' \
                   else LogisticRegression(C=1.0)

    stacking = StackingRegressor(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=5,  # 内部 CV 防止过拟合
        n_jobs=-1
    )

    stacking.fit(X, y)
    return stacking

# 注意：Stacking 模型的 SHAP 解释需要单独处理
# 对每个基础学习器分别解释，然后加权平均贡献
def stacking_shap_explanation(stacking_model, X):
    """对 Stacking 集成做 SHAP 解释（分别解释每个基础模型）"""
    shap_dict = {}
    for name, estimator in stacking_model.named_estimators_.items():
        if name != 'final_estimator_':
            explainer = shap.TreeExplainer(estimator)
            shap_dict[name] = explainer.shap_values(X)
    # 堆叠各模型的 SHAP 值（加权平均）
    avg_shap = np.mean(list(shap_dict.values()), axis=0)
    return shap_dict, avg_shap
```

---

## 8.6 策略对比总结

| 策略 | 精度提升 | 计算成本 | 可解释性 | 最佳场景 |
|------|---------|---------|---------|---------|
| LSTM-XGBoost | ★★★★ | ★★★ | ★★★ | 长期时序依赖 |
| BO-XGBoost | ★★ | ★★ | ★★★★★ | 小样本高维 |
| SMOTE-XGBoost | ★★★ | ★ | ★★★★★ | 类别不均衡 |
| VMD-XGBoost | ★★★★★ | ★★★★ | ★★★ | 非平稳时序 |
| Stacking | ★★★ | ★★★★★ | ★★ | 精度优先 |

---

## 8.7 输出

按本 skill 统一规范保存到 `05_Results/`：
- `Models/best_model.pkl` — 最终使用的模型 (可能为混合模型)
- `Models/hybrid_models/`:
  - `lstm_encoder.h5` — LSTM 编码器 (如有)
  - `stacking_model.pkl` — Stacking 集成模型 (如有)
  - `vmd_models.pkl` — VMD 分解模型组 (如有)
- `Tables/hybrid_model_comparison.csv` — 各策略最终对比表
- `Tables/bo_optimization_log.csv` — BO 优化过程日志
