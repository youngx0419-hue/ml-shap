---
name: report-assembly
description: Phase 10 报告聚合 —— ML-SHAP 审稿级报告、证据摘要、复现清单、输出审计
---

# Phase 10：报告聚合

## 10.0 文献证据与方法决策摘要

报告开头必须加入“方法决策摘要”：

- 领域识别：检测领域、置信度、证据列、是否用户确认。
- 分层证据：列出三篇最接近的近期论文、方法基础文献、XAI 局限性文献、报告/偏倚标准，说明它们如何影响方法流程和图形风格；若未能检索，说明原因。
- 数据依赖结构：随机、时间、空间、分组、外部验证中的哪一种，为什么。
- 模型候选集：为何选择 XGBoost/LightGBM/CatBoost/RF/可解释基线。
- 解释假设：TreeExplainer 模式、background 数据、是否存在强相关特征、是否使用 ALE/PDP。
- 风险控制：泄漏防控、校准、类别不平衡、不确定性、高风险使用、因果限制。
- 文献依据：从 `evidence_bank.json` 和 `references/literature_evidence.md` 选取与本任务相关的 5-10 条证据，不要机械列满全部文献。

## 10.1 报告结构 [ML-SHAP 更新]

```
┌──────────────────────────────────────────────────────┐
│       ML-SHAP 可解释机器学习分析报告                  │
│       ======================================          │
│       字体: Times New Roman + Unicode 化学下标         │
│       图表: 手动 Waterfall + 正确交互计算              │
├──────────────────────────────────────────────────────┤
│  §0 Causal Limitations [ML-SHAP MANDATORY — 不可跳过]   │
│     ⚠ SHAP != 因果                                    │
│     ⚠ 代理特征列表                                    │
│     ⚠ 公式泄漏声明                                    │
│     ⚠ 因果推断补充建议                                │
├──────────────────────────────────────────────────────┤
│  Executive Summary (执行摘要)                         │
├──────────────────────────────────────────────────────┤
│  1. Data Overview + 特征域分析 (Phase 0)              │
│  2. Model Development + 四模型对比 (Phase 3)          │
│  3. SHAP Global Interpretation (Phase 5)              │
│  4. SHAP Stability Diagnosis (Phase 5 稳定性诊断)      │
│  5. Feature Interaction Analysis (Phase 6 — 修正后)   │
│  6. Bootstrap Confidence Intervals (Phase 6)          │
│  7. Spatial-Temporal Stratified Analysis (Phase 7)    │
│  8. Limitations                                      │
│  9. Recommendations                                  │
│  Appendix: File Structure + Parameters + Audit        │
└──────────────────────────────────────────────────────┘
```

## 10.1a 附加审稿级产物

完整分析必须同时输出：

- `model_card.md`: 模型用途、非用途、训练数据、目标、候选模型、最终模型、性能、校准、解释范围、局限性、适用性边界。
- `dataset_datasheet.md`: 由 `scripts/profile_dataset.py` 生成后人工补充目标定义、样本单位、单位、采样机制、预测时点和泄漏审查。
- `risk_of_bias_checklist.md`: 高风险、临床、金融、灾害、风险预警、政策决策任务必须输出；普通科研任务可输出简版。
- `reproducibility_manifest.json`: 由 `scripts/reproducibility_manifest.py` 生成，记录 dataset hash、skill hash、包版本、随机种子、输出文件 hash。
- `output_audit.json`: 由 `scripts/validate_outputs.py` 生成，错误级 findings 必须修复或在最终回复中说明。

## 10.2 ML-SHAP 表格生成防错位规则 [CRITICAL]

**所有报告表格必须逐列显式赋值，严禁按位置索引**:

```python
# ✅ ML-SHAP 正确: 逐列显式赋值
strata = pd.read_csv(f'{RES_DIR}/Tables/strata_performance.csv')
city_perf = strata[strata['Type'] == 'City']
for _, row in city_perf.iterrows():
    report += (f"| {row['Strata']} | {row['N']:.0f} | "
               f"{row['R2']:.3f} | {row['MAE']:.0f} |\n")

# ❌ v1.0 BUG: 按位置索引 — 列序一变就错位
# | {row.iloc[0]} | {row.iloc[1]} | {row.iloc[2]} | {row.iloc[3]} |

# ❌ v1.0 BUG: 从 CSV 取列名顺序依赖
# 当 CSV 列为 [Strata,Type,N,R2,MAE] 而代码假设 [City,MeanAQI,MedianAQI,N,R2]
# 导致样本数出现在均值列
```

### §7.1 城市表格 (正确模板)

```python
# 从 strata_performance.csv 读取成绩数据
# 从 strata_context.json 读取均值/中位 AQI 数据
with open(f'{RES_DIR}/Cache/strata_context.json', 'r') as f:
    sctx = json.load(f)

city_stats_mean = sctx['city_stats']['Mean_AQI']
city_stats_median = sctx['city_stats']['Median_AQI']
city_stats_count = sctx['city_stats']['Count']

strata = pd.read_csv(f'{RES_DIR}/Tables/strata_performance.csv')
city_perf = strata[strata['Type'] == 'City']

report += "| 城市 | 均值 AQI | 中位 AQI | 样本数 | 模型 R2 | MAE |\n"
report += "|------|---------|----------|--------|---------|-----|\n"

top5_cities = ['Ahmedabad', 'Delhi', 'Patna', 'Lucknow', 'Gurugram']
for city in top5_cities:
    mean_aqi = city_stats_mean.get(city, 0)
    median_aqi = city_stats_median.get(city, 0)
    count = city_stats_count.get(city, 0)
    cdata = city_perf[city_perf['Strata'] == city]
    if len(cdata) > 0:
        r2 = cdata.iloc[0]['R2']
        mae = cdata.iloc[0]['MAE']
    else:
        r2, mae = 0, 0
    report += f"| {city} | {mean_aqi:.0f} | {median_aqi:.0f} | {count:.0f} | {r2:.3f} | {mae:.0f} |\n"
```

### §7.2 季节表格 (正确模板)

```python
season_stats_mean = sctx['season_stats']['Mean_AQI']
season_stats_median = sctx['season_stats']['Median_AQI']
season_stats_count = sctx['season_stats']['Count']
season_order = ['Spring', 'Summer', 'Autumn', 'Winter']

season_perf = strata[strata['Type'] == 'Season']

report += "| 季节 | 均值 AQI | 中位 AQI | 样本数 | 模型 R2 | MAE |\n"
report += "|------|---------|----------|--------|---------|-----|\n"

for s in season_order:
    mean_aqi = season_stats_mean.get(s, 0)
    median_aqi = season_stats_median.get(s, 0)
    count = season_stats_count.get(s, 0)
    sdata = season_perf[season_perf['Strata'] == s]
    if len(sdata) > 0:
        r2 = sdata.iloc[0]['R2']
        mae = sdata.iloc[0]['MAE']
    else:
        r2, mae = 0, 0
    report += f"| {s} | {mean_aqi:.0f} | {median_aqi:.0f} | {count:.0f} | {r2:.3f} | {mae:.0f} |\n"
```

### §3.1 SHAP 重要性表格

```python
shap_csv = pd.read_csv(f'{RES_DIR}/Tables/shap_values.csv')
total = shap_csv['SHAP_Importance'].sum()

report += "| 排名 | 特征 | Mean |SHAP| | 占比 | XGBoost Gain |\n"
report += "|------|------|------------|------|-------------|\n"

feat_imp = pd.read_csv(f'{RES_DIR}/Tables/feature_importance.csv')
for rank, (_, row) in enumerate(shap_csv.iterrows(), 1):
    pct = row['SHAP_Importance'] / total * 100
    gain_row = feat_imp[feat_imp['Feature'] == row['Raw_Name']]
    gain_val = gain_row['Importance'].values[0] if len(gain_row) > 0 else 0
    report += f"| {rank} | {row['Display_Name'] if 'Display_Name' in row else row['Feature']} "
    report += f"| {row['SHAP_Importance']:.4f} | {pct:.1f}% | {gain_val:.4f} |\n"
```

## 10.3 ML-SHAP 交叉验证 (报告生成后必须执行)

```python
# 验证 1: 城市表格数值与 strata_context.json 一致
with open(f'{RES_DIR}/Cache/strata_context.json', 'r') as f:
    sctx = json.load(f)
assert abs(sctx['city_stats']['Mean_AQI']['Ahmedabad'] - 446) < 1, "City mean mismatch"
assert abs(sctx['season_stats']['Mean_AQI']['Winter'] - 216) < 1, "Season mean mismatch"

# 验证 2: SHAP CSV 与 auto_insights.json 一致
with open(f'{RES_DIR}/Cache/auto_insights.json', 'r') as f:
    insights = json.load(f)
shap_csv = pd.read_csv(f'{RES_DIR}/Tables/shap_values.csv')
for i, (_, row) in enumerate(shap_csv.iterrows()):
    diff = abs(row['SHAP_Importance'] - insights['shap_importance'][i])
    assert diff < 1e-10, f"SHAP mismatch: {row['Feature']}"

# 验证 3: 报告 §0 因果声明存在
assert "SHAP !=" in report_text or "SHAP !=" in report_text, "Missing causal disclaimer"

# 验证 4: R2 标注正确 (两个尺度)
assert "R2=0.923" in report_text or "R2 = 0.923" in report_text, "Missing log1p R2"
assert "0.890" in report_text, "Missing original-scale R2"

# 验证 5: 图表数量正确
assert "20 " in report_text or "20 " in report_text, "Wrong chart count"
```

## 10.4 报告 §0 因果局限性强制模板

```markdown
## ⚠️ 因果局限性声明 [ML-SHAP 强制]

1. SHAP != 因果: SHAP 值量化模型内部的相关性归因, 不等同于因果效应。
2. 代理特征警告: City/Date 不是直接因果特征, 其影响通过 Phase 7 分层分析体现。
3. 公式泄漏声明: DecisionTree(depth=6) CV R2 = {cv_r2:.4f} (< 0.85 / 0.85-0.95 / > 0.95),
   {leakage_statement}
4. 因果推断补充建议: IV / DID / 格兰杰因果 / RCT / CTM
5. 模型预测能力: log1p Test R2 = {r2_log1p:.4f}, 原始 Test R2 = {r2_orig:.4f},
   Test RMSE = {rmse_orig:.1f} AQI
```

## 10.5 输出检查清单 [ML-SHAP — 分析完成后逐项确认]

- [ ] report.md §0 因果声明完整
- [ ] dataset_datasheet.md 记录目标定义、样本单位、预测时点、单位、分组/时间/空间结构
- [ ] evidence_bank.json 区分领域近邻文献、方法基础文献、XAI 局限性文献、报告标准
- [ ] report.md §2.2 两个尺度 R² 均正确标注
- [ ] report.md §7.1 城市表格列对齐 (均值/中位/样本数/R²/MAE)
- [ ] report.md §7.2 季节表格列对齐
- [ ] report.md §3.1 SHAP 表格占比与 §3.2 文字一致
- [ ] report.md §5.1 交互数据与 §6 源数据一致
- [ ] report.md §A 图表数量 = 实际文件数
- [ ] 所有 CSV 编码 utf-8-sig
- [ ] auto_insights.json 与 shap_values.csv 数值一致
- [ ] strata_context.json 与 strata_performance.csv 数值一致
- [ ] checkpoint.json phases 列表完整
- [ ] reproducibility_manifest.json 完整
- [ ] model_card.md 完整
- [ ] output_audit.json 无 error 级 finding
- [ ] 无临时脚本残留
