# Domain-First Literature And Visualization Workflow

Read this file immediately after the dataset profile is available. This workflow makes the skill domain-aware before model training.

## Required Output

Create `domain_context.json`, `literature_search_plan.json`, and `evidence_bank.json` before `method_decision_log.json`.

```json
{
  "domain": "carbon | water | risk | spatial_temporal | biochar_materials | health_environment | agriculture_ecology | energy_systems | finance_business | general",
  "confidence": 0.0,
  "evidence_columns": [],
  "user_override": null,
  "literature_search_queries": [],
  "nearest_recent_papers": [],
  "literature_search_status": "pending | completed | not_available",
  "evidence_strategy": "three nearest domain papers + method foundation + XAI caution + reporting standard",
  "domain_method_adjustments": [],
  "domain_visual_style": {},
  "open_questions": []
}
```

## Step 1: Identify The Main Research Domain

Use the dataset columns, target variable, user prompt, units, file names, and any metadata. If confidence is below 0.55, present the top two candidates and ask the user to choose.

Use `scripts/domain_context_engine.py` when a quick deterministic pass is helpful.

Common domain signals:

| Domain | Signals |
|---|---|
| `carbon` | CO2, carbon, emission, energy intensity, GDP, population, industrial structure, carbon peak, carbon neutrality |
| `water` | DO, COD, BOD, NH3-N, TN, TP, pH, turbidity, conductivity, river, lake, basin, water quality |
| `risk` | hazard, susceptibility, vulnerability, flood, landslide, wildfire, earthquake, disaster, accident, risk probability |
| `spatial_temporal` | NDVI, land use, landscape, urbanization, biomass, grid, region, coordinates, time series, spatial autocorrelation |
| `biochar_materials` | biochar, adsorption, pyrolysis, residence time, BET, SSA, pore volume, feedstock, activation, removal rate |
| `health_environment` | exposure, disease, clinical, hospital, patient, epidemiology, PM2.5-health, mortality, morbidity |
| `agriculture_ecology` | crop, yield, soil, fertilizer, irrigation, vegetation, biodiversity, ecosystem, remote sensing |
| `energy_systems` | power load, renewable, PV, wind, battery, grid, electricity, energy demand |
| `finance_business` | credit, default, churn, sales, customer, transaction, fraud, revenue |

## Step 2: Create A Layered Evidence Plan

Use `scripts/evidence_plan.py` to create `literature_search_plan.json`. The evidence plan must include:

- Three closest recent domain papers.
- Method-foundation evidence for gradient boosting, SHAP/TreeSHAP, leakage-safe validation, and feature-dependence caveats.
- One XAI caution or causal-limitation source.
- One reporting or risk-of-bias standard when the task is a prediction, risk, clinical, environmental decision, or other high-stakes analysis.

Do not treat the three nearest papers as sufficient on their own. Recent domain papers show field conventions; method-foundation papers guard against methodological mistakes.

## Step 3: Search Three Nearest Recent Papers

After domain detection, search for the three most recent and closest papers that combine the domain with XGBoost/SHAP/explainable machine learning. "Closest" means:

1. Same domain and target concept.
2. Uses XGBoost, gradient boosting, SHAP, interpretable ML, or explainable AI.
3. Similar data structure: tabular, spatial, temporal, remote sensing, experimental materials, classification risk, etc.
4. Published in 2024-2026 when possible. If too few relevant papers exist, broaden to 2021-2026 and state that fallback.

Recommended query template:

```text
("{domain keywords}" OR "{target variable}") AND (XGBoost OR "gradient boosting" OR SHAP OR "explainable machine learning" OR XAI) AND ({data structure keywords})
```

Use source routing:

- Cross-disciplinary environmental/materials/business: CrossRef or web search, then Semantic Scholar/Google Scholar if available.
- Biomedical/health: PubMed first, then Semantic Scholar/web.
- CS/methodology: arXiv plus CrossRef.
- Chinese-only domain context: web search for English nearest papers first; mention CNKI/Wanfang is manual if needed.

For each selected paper, store:

```json
{
  "title": "",
  "year": 0,
  "venue": "",
  "doi_or_url": "",
  "abstract_or_key_claim": "",
  "why_close": "",
  "similarity_score_0_100": 0,
  "method_pattern": "",
  "visual_pattern": "",
  "transferable_to_this_dataset": "",
  "caution": ""
}
```

If no internet/search tools are available, set `nearest_recent_papers=[]`, mark `literature_search_status="not_available"`, and fall back to `references/literature_evidence.md` plus domain templates. Do not fabricate paper titles.

## Step 4: Build `evidence_bank.json`

Create a machine-readable evidence bank:

```json
{
  "papers": [],
  "method_influence_table": [],
  "visual_influence_table": [],
  "claims_supported": [],
  "claims_not_supported": []
}
```

For every method or figure decision influenced by a paper, add one row to `method_influence_table` or `visual_influence_table`. Each row must include: decision, supporting source, transfer rationale, and limitation.

## Step 5: Convert Papers Into Domain Method Adjustments

Do not merely cite papers. Extract operational changes:

- Domain-specific feature families and leakage risks.
- Expected validation design: random, grouped, spatial block, time-series, external.
- Model family preferences: XGBoost, LightGBM, CatBoost, calibrated classifiers, interpretable baselines.
- Explanation plots expected by the field.
- Reporting phrases and caveats common to the field.
- Any domain-specific metrics, thresholds, or units.

Write these into `domain_method_adjustments`.

## Step 6: Choose Domain Visual Style

Visual style should fit the field, not personal taste. Use `references/domain_visual_style.md` and the selected recent papers. The style decision must specify:

- Primary palette.
- Figure background and axes background.
- Map/heatmap colormap.
- Preferred plot types.
- Whether uncertainty, threshold bands, or spatial overlays are required.
- Label conventions and units.

Write the result to `domain_context.json.domain_visual_style`, then pass it to plotting code.

## Step 7: Continue The Main Workflow

Use `domain_context.json` to enrich:

- `method_decision_log.json`: add domain and selected paper influence.
- `feature_decision_table.csv`: use domain feature families and leakage cautions.
- Model benchmark: include domain-preferred candidates.
- SHAP plots: use domain-specific grouping, coloring, and caveats.
- Report: include the three nearest recent papers in the method rationale.
