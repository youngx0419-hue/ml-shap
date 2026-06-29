---
name: preprocessing
description: 自适应数据预处理模块 —— ML-SHAP 新增特征域分析(因果vs代理变量分离/化学冗余检测/目标相关性筛选) + 缺失值策略选择 + 异常值处理 + 类别编码 + VIF共线性筛查 + 特征筛选 + 标准化。
---

# Phase 2：特征域分析 + 数据预处理

## 2.0 特征域分析 [ML-SHAP NEW, 必须在建模前执行]

**这是 ML-SHAP 最关键的新增步骤。** 在缺失值填充和建模之前，必须回答：哪些特征与研究问题有因果关系？

### (a) 特征三分类

```python
# 1. Causal features (因果特征): 与研究问题有直接因果/机制关系
causal_candidates = ['PM2.5', 'PM10', 'CO', 'NO2', ...]  # 污染物 → AQI

# 2. Proxy features (代理变量): 与目标相关但非直接因果
proxy_features = ['City', 'Year', 'Month', 'DayOfYear']  # 实际是排放源+气象的代理

# 3. Metadata (元数据/标识符): 与目标无任何机制关系
metadata_features = ['ID', 'SampleNo', 'Timestamp']
```

**规则**: 只有 causal_candidates 进入模型。proxy_features 留给 Phase 7 分层分析。metadata_features 排除。

### (b) 化学/物理冗余检测 [??]

```python
# 检查是否存在由其他特征通过已知公式推导的特征
# 示例1: NOx = NO + NO2 (化学定义)
estimated_nox = df['NO'] + df['NO2']
r_nox = df['NOx'].corr(estimated_nox)
if r_nox > 0.95:
    drop_features.append('NOx')  # 冗余: NOx = NO + NO2

# 示例2: 总VOC ≈ Benzene + Toluene + Xylene (近似)
# 常见化学/物理冗余:
# - NOx ≈ NO + NO2
# - PM_coarse ≈ PM10 - PM2.5
# - Total_VOC ≈ Σ(individual_VOCs)
# - COD ≈ f(BOD, TOC) in water quality
```

### (c) 目标相关性筛选 [??]

```python
# 删除与目标无线性相关的特征
for col in causal_candidates:
    r = df[col].corr(df[target])
    if abs(r) < 0.05:
        drop_features.append(col)
        print(f'DROP {col}: |r|={abs(r):.4f} with target')
```

### (d) 最终特征集确认

输出特征选择决策表供用户确认：

```
Feature | Category    | VIF  | Corr(target) | Decision
--------|-------------|------|--------------|----------
PM2.5   | causal      | 1.7  | +0.73        | KEEP
NOx     | redundant   | 3.4  | +0.46        | DROP (NOx=NO+NO2)
Benzene | causal      | 2.3  | +0.04        | DROP (|r|<0.05)
City    | proxy       | N/A  | N/A          | STRATA (Phase 7)
Year    | proxy       | N/A  | N/A          | STRATA (Phase 7)
```

保存为 `feature_selection_log.json`。

---

## 2.1 缺失值处理决策树

```
缺失率 < 3%     → 数值型: 中位数填充 | 类别型: 众数填充
缺失率 3%-20%   → 数值型: KNNImputer(n_neighbors=5) | 类别型: 添加"Missing"类别
缺失率 20%-50%  → IterativeImputer (MICE) + 缺失指示列
缺失率 > 50%    → 删除该列（需用户确认）
```

对于城市/区域数据，优先使用**组内中位数**填充（同城市/同区域的数据更相似）：

```python
# 优先: 城市级中位数填充
for col in pollutant_cols:
    df[col] = df.groupby('City')[col].transform(lambda x: x.fillna(x.median()))
    df[col] = df[col].fillna(df[col].median())  # fallback: 全局中位数
```

## 2.2 异常值处理

**不直接删除**。首先检查是否为物理可能值（如 PM2.5 不能为负数）。对极端值使用 Winsorize（1-99 百分位截尾）：

```python
from scipy.stats.mstats import winsorize
for col in high_outlier_cols:
    df[col] = winsorize(df[col], limits=(0.01, 0.01))
```

对于多维异常值，添加 `is_outlier` 标记列而非删除。

## 2.3 类别特征编码

按 ML-SHAP 分层原则，空间/类别标识符不作为模型特征。仅当类别特征本身就是因果变量时才编码：

```
唯一值 ≤ 10     → OneHotEncoder (drop='first')
唯一值 11-50    → TargetEncoder（CV编码防泄漏）
有序类别        → OrdinalEncoder
```

## 2.4 多重共线性筛查（VIF）

```python
from sklearn.linear_model import LinearRegression

def check_vif(df, features):
    """VIF 诊断 — 检测特征间共线性 (单次计算)"""
    vif_results = {}
    for col in features:
        y_v = df[col]
        X_v = df[features].drop(columns=[col])
        r2 = LinearRegression().fit(X_v, y_v).score(X_v, y_v)
        vif = 1.0 / (1.0 - r2) if r2 < 0.999 else float('inf')
        vif_results[col] = vif
    return vif_results

# ML-SHAP: VIF > 10 的特征使用迭代删除 (每次删除最高 VIF, 重新计算)
def iterative_vif_removal(df, features, max_vif=10):
    """迭代删除最高 VIF 特征, 直到所有 VIF < max_vif"""
    remaining = list(features)
    dropped = []
    while True:
        vif_results = check_vif(df, remaining)
        max_vif_col = max(vif_results, key=vif_results.get)
        max_vif_val = vif_results[max_vif_col]
        if max_vif_val <= max_vif:
            break
        print(f'  DROP {max_vif_col}: VIF={max_vif_val:.1f} > {max_vif}')
        remaining.remove(max_vif_col)
        dropped.append((max_vif_col, max_vif_val))
    return remaining, dropped
```

## 2.5 数据划分

```python
from sklearn.model_selection import train_test_split

# 先划分再预处理！避免数据泄漏
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 对训练集 fit，对测试集 transform
imputer.fit(X_train)
X_train_imputed = imputer.transform(X_train)
X_test_imputed = imputer.transform(X_test)
```

## 2.6 输出

保存到 `05_Results/<项目名>_v<N>/Tables/`：
- `feature_selection_log.json` — ML-SHAP NEW: 特征筛选决策记录
- `vif_report.csv` — VIF 共线性诊断结果
- `preprocessing_report.json` — 所有转换参数

## 2.7 鲁棒性要求

- 所有转换器必须保存为 `.pkl`，确保可复现
- 对测试集只能用 `transform`，禁止 `fit_transform`
- 每一步加 try/except，失败时输出清晰错误信息并跳过
- **严禁** `warnings.filterwarnings('ignore')` — 仅过滤 sklearn 收敛警告
