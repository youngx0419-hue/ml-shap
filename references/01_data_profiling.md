---
name: data-profiling
description: 自动数据探查模块 —— 识别列角色、检测数据质量、推断任务类型、生成数据质量评分卡。适用于任何结构化数据集的第一步分析。
---

# Phase 1：自动数据探查

## 1.1 列角色自动识别

对每一列执行以下分类逻辑：

| 角色 | 检测规则 | 处理建议 |
|------|---------|---------|
| **数值型** | dtype 为 int/float，唯一值 > 20 | 用于特征工程和建模 |
| **类别型** | dtype 为 object/category，或数值型但唯一值 ≤ 15 | 需要编码 |
| **时间型** | 列名含 date/time/year/month/day，或可被 `pd.to_datetime` 解析 | 提取时序特征 |
| **空间型** | 列名含 lat/lon/longitude/latitude/x/y/coord/城市/区域 | 空间分析 |
| **ID 型** | 列名含 id/ID/No/code，或唯一值率 > 90% 且非数值 | 排除 |
| **文本型** | dtype 为 object，唯一值率 > 50%，平均字符串长度 > 20 | 排除或 NLP |
| **目标候选** | 列名含 target/label/class/result/outcome/emission/risk/quality，或位于最后一列 | 用户确认 |

## 1.1.5 领域自动识别（ML-SHAP 新增）

在列角色识别的基础上，进一步推断数据所属的应用领域，为后续 Phase 提供策略指导：

```python
def detect_application_domain(df, col_names=None):
    """基于列名关键词的领域自动识别（来源于 90+ 篇论文的元分析）"""
    if col_names is None:
        col_names = ' '.join(df.columns).lower()

    domains = {
        'carbon': {
            'keywords': ['carbon', 'co2', 'emission', '碳排放', '碳达峰', '碳中和',
                         '能耗', '能源', '碳强度', '碳储量', '碳汇', '温室气体',
                         '碳金融', '碳交易', '碳减排'],
            'typical_targets': ['carbon_emission', 'co2', 'carbon_intensity', '碳排放量'],
            'typical_features': ['gdp', 'population', 'energy', 'urbanization', 'industry',
                                '城镇化率', '产业结构', '能源结构', '人均GDP'],
        },
        'water': {
            'keywords': ['do', 'cod', 'nh3', 'tp', 'tn', 'bod', 'ph', '溶解氧',
                         '水质', '污染物', '浊度', 'tss', 'ec', '盐度', '氨氮',
                         '总磷', '总氮', '高锰酸盐', '叶绿素', '富营养化'],
            'typical_targets': ['do_concentration', 'water_quality_class', 'cod', '水质等级'],
            'typical_features': ['temperature', 'precipitation', 'flow', 'landuse',
                                '水温', '流量', 'pH', '电导率'],
        },
        'risk': {
            'keywords': ['risk', 'hazard', 'fire', 'flood', '火灾', '洪水', '滑坡',
                         '地震', '易发性', '风险', 'susceptibility', 'vulnerability',
                         '灾害', '危险性', '暴露度', '韧性'],
            'typical_targets': ['risk_level', 'fire_probability', 'flood_depth', '风险等级'],
            'typical_features': ['dem', 'slope', 'precipitation', 'population', 'landuse',
                                '高程', '坡度', '降水', '人口密度'],
        },
        'spatial_temporal': {
            'keywords': ['ndvi', 'landscape', '景观', '生态韧性', '格局', 'landuse',
                         '土地利用', '生物量', 'biomass', '城市化', 'urban',
                         'lulc', '蔓延', '扩张', '演变', '空间格局'],
            'typical_targets': ['ecological_resilience', 'biomass', 'carbon_storage', '生态韧性'],
            'typical_features': ['ndvi', 'nightlight', 'dem', 'distance_to_city',
                                '夜间灯光', '高程', '距城市距离'],
        },
        'materials': {
            'keywords': ['biochar', 'adsorption', 'pyrolysis', '吸附', '热解',
                         '生物炭', '去除率', 'sorption', '表面积', 'bet',
                         '孔径', '官能团', '原子比', '灰分'],
            'typical_targets': ['adsorption_capacity', 'removal_rate', '吸附容量', '去除率'],
            'typical_features': ['pyrolysis_temp', 'SSA', 'pH', 'ash_content', 'C_content',
                                '热解温度', '比表面积', '灰分含量'],
        },
    }

    scores = {}
    for domain, info in domains.items():
        score = sum(1 for kw in info['keywords'] if kw in col_names)
        if score > 0:
            scores[domain] = score

    if not scores:
        return 'general'

    return max(scores, key=scores.get)
```

领域识别的结果会影响后续 Phase 的默认策略。例如：
- `carbon` 域 → 自动启用 LMDI 分解 + 多情景碳达峰
- `risk` 域 → 自动启用 SMOTE + 概率校准
- `materials` 域 → CatBoost 优先于 XGBoost（类别特征多）
- `spatial_temporal` 域 → 自动启用空间自相关分析

## 1.2 数据质量检测

### 缺失值分析
```python
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).sort_values(ascending=False)
missing_report = pd.DataFrame({
    '列名': missing_pct.index,
    '缺失率(%)': missing_pct.values,
    '缺失类型推断': ['MCAR' if pct < 5 else 'MAR' if pct < 30 else '需关注' for pct in missing_pct.values]
})
```

缺失机制推断规则：
- **MCAR**（完全随机缺失）：缺失率 < 5%，与其他列无显著相关 → 中位数/众数填充
- **MAR**（随机缺失）：缺失率 5-30%，与其他列有相关 → KNN/MICE 填充
- **MNAR**（非随机缺失）：缺失率 > 30%，或业务逻辑暗示缺失有意义 → 添加缺失指示列 + 特殊填充

### 异常值检测
```python
# 方法一：IQR 法（适用于近似正态分布）
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
outlier_mask = (df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))
outlier_pct = outlier_mask.mean() * 100

# 方法二：Isolation Forest（适用于多维异常检测，鲁棒性更好）
from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.05, random_state=42)
outlier_labels = iso.fit_predict(df[numeric_cols].fillna(0))
```

### 常数/近常数特征检测
```python
# 唯一值占比 < 1% 的特征 → 建议删除
near_constant_cols = [c for c in df.columns if df[c].nunique() / len(df) < 0.01]
```

### 重复列检测
```python
# 两两相关系数 > 0.99 的数值列 → 建议删除其一
```

## 1.3 任务类型自动推断

```python
def infer_task_type(y):
    n_unique = y.nunique()
    if n_unique <= 2:
        return 'binary_classification'
    elif 3 <= n_unique <= 15:
        return 'multi_class_classification'
    elif n_unique > 15 and y.dtype in ['int64', 'float64']:
        # 进一步判断：唯一值占比 > 50% 可能是连续回归
        if n_unique / len(y) > 0.5:
            return 'regression'
        else:
            return 'multi_class_classification'  # 可能是整数编码的分类
    else:
        return 'regression'
```

## 1.4 目标变量分析

### 回归目标
- 分布直方图 + KDE
- 偏度（|skewness| > 1 建议 log 变换）
- 是否有零膨胀（零值占比 > 30%）

### 分类目标
- 类别分布条形图（检查类别不平衡）
- 不平衡度 = 最大类样本数 / 最小类样本数
- 如果不平衡度 > 10 → 告警并建议使用 `scale_pos_weight` 或 SMOTE

## 1.5 输出：数据质量评分卡

```
┌─────────────────────────────────────────────┐
│              数据质量评分卡                    │
├─────────────────────────────────────────────┤
│ 总样本数:      XX,XXX                        │
│ 总特征数:      XX                            │
│ 缺失率 > 5%:   X 列  [警告]                   │
│ 异常值率 > 5%: X 列  [警告]                   │
│ 常数/近常数:   X 列  [建议删除]                │
│ 高共线性对:    X 对  [需关注]                  │
│ 类别不平衡度:  X:1   [严重/中等/正常]          │
│ 综合质量分:    XX/100                        │
└─────────────────────────────────────────────┘
```

## 1.6 缓存

探查结果保存到 `05_Results/00_数据探查/data_profile.csv` 和 `data_profile.json`，后续 Phase 可直接读取。
