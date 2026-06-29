---
name: ml-shap
description: Run or review complete ML-SHAP explainable AI workflows for structured tabular data using XGBoost, LightGBM, CatBoost, Random Forest baselines, SHAP, evidence-layered literature search, data datasheets, reproducibility manifests, and output audits. Use for Chinese or English requests involving ML-SHAP, XAI, SHAP, XGBoost-SHAP, feature importance, feature interaction, model explanation, interventional TreeExplainer, SHAP beeswarm, dependence, waterfall, PDP/ALE validation, time-space stratified SHAP, carbon emission, water quality, wastewater or sludge prediction, risk assessment, biochar materials, publication-quality SVG/PNG figures, model cards, dataset datasheets, or full reproducible machine-learning reports.
---

# ML-SHAP Workflow

Use this skill to build, audit, or repair a reproducible ML-SHAP explainable AI analysis for structured tabular data. Treat it as a literature-backed workflow controller: keep this file in context, then load only the reference files needed for the current task.

## Runtime Compatibility

- Codex: invoke as `$ml-shap`; read this `SKILL.md`, then load references and run scripts as needed.
- Claude Code: install this folder as `~/.claude/skills/ml-shap` or `.claude/skills/ml-shap`; invoke with `/ml-shap` when slash-skill invocation is available, or ask Claude Code to use the installed ML-SHAP skill. Use this same `SKILL.md`, references, assets, and scripts. Do not rely on Codex-only `agents/openai.yaml`.
- Keep all core operations runnable through plain Python CLI scripts so the workflow remains portable across Codex, Claude Code, and ordinary terminal use.

## Operating Rules

- Choose the lightest run tier that satisfies the user request:
  - `quick`: dataset triage, leakage review, one suitable tree model, core metrics, SHAP beeswarm/bar/dependence, concise notes. Do not require literature search, model cards, or paper-grade audits unless requested.
  - `standard`: reproducible analysis with profiling, feature decision table, benchmarked tree models, evaluation diagnostics, core SHAP outputs, reproducibility manifest, and phase-aware audit.
  - `research`: full paper/report workflow with domain literature evidence, method decision log, advanced SHAP, uncertainty/stability checks, model card, risk-of-bias checklist when relevant, and final audit.
- Confirm the target variable, task type, prediction timing, final feature set, stratification variables, and output goal before modeling. If these cannot be inferred safely from the data or user request, ask one concise question before training.
- Profile the dataset and create `data_profile.json` plus `dataset_datasheet.md` before domain detection. Use `scripts/profile_dataset.py` when a dataset file is available.
- After profiling user data, identify the main research domain and create `domain_context.json`. Use `references/domain_literature_workflow.md` and `scripts/domain_context_engine.py`.
- Build a layered evidence plan with `scripts/evidence_plan.py`. Search three closest recent domain papers plus method-foundation, XAI-caution, and reporting-standard sources whenever internet or academic-search tools are available. Extract method and visual conventions into `domain_context.json` and `evidence_bank.json`; do not fabricate papers when search is unavailable.
- Before modeling, create a method decision log from the dataset profile. Use `references/method_decision_engine.md` and, when useful, `scripts/method_decision_engine.py`. This log must justify the split strategy, model family, explanation method, and reporting gates.
- Separate causal or mechanistic features, proxy features, and metadata before preprocessing. Do not encode row IDs, city names, timestamps, coordinates, or group labels as model features unless the user explicitly frames them as causal predictors.
- Prevent leakage. Split train/test before fitting imputers, encoders, scalers, target transforms, feature selectors, SMOTE, or any supervised transformation.
- Use one selected model instance for all downstream SHAP outputs. Do not mix models between beeswarm, dependence, interaction, waterfall, and report tables.
- Do not claim causality from SHAP alone. Every report must include a causal limitation statement unless the user provides an experimental or quasi-experimental design.
- Avoid `shap.force_plot()` and `shap.plots.force()`. Use beeswarm, bar, dependence, heatmap, interaction matrix, and manual matplotlib waterfall plots instead.
- Save publication figures as both SVG and PNG. Remove U+2212 minus signs from SVG files, use ASCII hyphen-minus for numeric labels, and use Unicode chemical subscripts only for display labels.
- Keep console output GBK-safe on Windows: log raw feature names to terminal and reserve Unicode display names for figures and reports.
- Finish every full analysis with `reproducibility_manifest.json`, `model_card.md`, `risk_of_bias_checklist.md` when relevant, and `output_audit.json` from `scripts/validate_outputs.py`.

## Reference Map

Load references only when their phase is needed.

| Need | Read |
|---|---|
| Data profiling, column roles, task inference, quality scorecard | `references/01_data_profiling.md` |
| Domain detection, nearest recent papers, and domain-first workflow | `references/domain_literature_workflow.md` |
| Domain-specific visual style cards and chart routing | `references/domain_visual_style.md` |
| Literature-backed method choice, validation design, SHAP/PDP/ALE/calibration decisions | `references/method_decision_engine.md` |
| Paper evidence and citation support for reports or method justification | `references/literature_evidence.md` |
| Feature domain analysis, leakage screening, imputation, encoding, VIF, train/test split | `references/02_preprocessing.md` |
| Fair model benchmarking, Optuna tuning, stacking, final model choice | `references/03_model_benchmark.md` |
| Regression/classification evaluation, learning curves, residuals, calibration | `references/04_evaluation.md` |
| Core SHAP values, beeswarm, bar importance, manual waterfall, dependence plots | `references/05_shap_basic.md` |
| SHAP interactions, interaction matrix, bootstrap CI, stability, clustering | `references/06_shap_advanced.md` |
| Spatial or temporal heterogeneity, strata-based SHAP, Moran/Gi/KDE, time-series CV | `references/07_spatial_temporal.md` |
| LSTM-XGBoost, BO-XGBoost, SMOTE-XGBoost, VMD-XGBoost, stacking variants | `references/08_hybrid_models.md` |
| Domain-specific workflows for carbon, water, risk, spatial patterns, biochar materials | `references/09_domain_templates.md` |
| Report structure, explicit table mapping, report validation checklist | `references/10_report_assembly.md` |
| Plot style constants, colors, fonts, SVG sanitation, chemical labels | `references/style_constants.md` |
| HTML report shell | `assets/report_template.html` |
| Optional local visual references for XGBoost, SHAP, interaction, heatmap, and PDP figures | `assets/reference-gallery/README.md` |
| Reusable plotting helpers | `scripts/xai_style_helpers.py` |
| Standard run directory bootstrap | `scripts/bootstrap_run.py` |
| Dataset profile and datasheet generator | `scripts/profile_dataset.py` |
| Domain context and literature query generator | `scripts/domain_context_engine.py` |
| Evidence-layer search plan generator | `scripts/evidence_plan.py` |
| Method decision log generator | `scripts/method_decision_engine.py` |
| Semi-automatic feature leakage and role reviewer | `scripts/review_features.py` |
| Minimal leakage-aware modeling runner with auto time/group/stratified split selection | `scripts/run_modeling.py` |
| Literature evidence bank importer from CSV/JSON notes | `scripts/update_evidence_bank.py` |
| Markdown report and model-card assembler | `scripts/assemble_report.py` |
| Reproducibility manifest generator | `scripts/reproducibility_manifest.py` |
| Output completeness and integrity audit | `scripts/validate_outputs.py` |

## Quick Start With A Dataset

When the user provides a dataset file, create the run directory and early handoff artifacts with the smallest suitable tier:

```bash
python scripts/bootstrap_run.py <dataset-path> --run-dir <output-run-dir> --target <target-column-if-known> --tier standard --user-text "<project description>"
```

For `research` tier, perform live literature search, fill `domain_context.json.nearest_recent_papers` and `evidence_bank.json`, rerun or update `method_decision_log.json` if the evidence changes validation/model/explanation choices, and continue modeling. For `quick` tier, keep literature fields explicit but do not block modeling on them.

Before modeling, review the drafted feature table. Use `--apply-safe-defaults` only when the auto-drafted role decisions are acceptable and no domain-specific leakage concerns remain:

```bash
python scripts/review_features.py --run-dir <output-run-dir> --target <target-column> --apply-safe-defaults
```

After confirming `feature_decision_table.csv`, run the minimal standard modeling pass when a full custom analysis script is not needed. The default split strategy is `auto`, which prefers time holdout, then group/spatial holdout, then classification stratification, then random holdout:

```bash
python scripts/run_modeling.py <dataset-path> --run-dir <output-run-dir> --target <target-column> --split-strategy auto
python scripts/validate_outputs.py <output-run-dir> --phase modeling
```

Treat `scripts/run_modeling.py` as a conservative baseline runner. For publication or high-stakes work, inspect and extend its outputs instead of blindly accepting the selected model.

For report-grade outputs, optionally import literature evidence notes, assemble the report, then run the final audit:

```bash
python scripts/update_evidence_bank.py --run-dir <output-run-dir> --source <evidence.csv>
python scripts/assemble_report.py --run-dir <output-run-dir> --title "<analysis title>"
python scripts/validate_outputs.py <output-run-dir> --phase final
```

## Workflow

1. Create a run directory with stable subfolders:
   `01_Data/`, `02_Processing/`, `03_Models/`, `04_SHAP/`, `05_Results/Charts/`, `05_Results/Tables/`, `05_Results/Reports/`, and `logs/`.
2. Profile the dataset. Identify target candidates, ID columns, date/time columns, spatial/group variables, categorical columns, missingness, duplicate rows, duplicate columns, constants, outliers, and likely task type. Create `data_profile.json` and `dataset_datasheet.md`.
3. Create `domain_context.json`. Detect the main research domain, record confidence and evidence columns, and choose an initial domain visual style.
4. Create `literature_search_plan.json`. Search for the three closest recent domain papers that use XGBoost, SHAP, gradient boosting, interpretable ML, or XAI, then add method-foundation, XAI-caution, and reporting-standard evidence. Update `domain_context.json` and create `evidence_bank.json` with paper metadata, similarity scores, method patterns, visual patterns, transferable decisions, and cautions.
5. Create `method_decision_log.json`. Decide whether random CV, grouped CV, spatial block CV, time-series split, external validation, CatBoost-first modeling, class-imbalance handling, calibration, conformal/prediction intervals, ALE, or high-stakes interpretable baselines are required. Include influence from the evidence bank.
6. Build and review a feature decision table. Mark each candidate as keep, drop, transform, or strata-only, with a reason. Run `scripts/review_features.py` to flag formula leakage, identifier predictors, target-name leakage, high-correlation interpretation issues, and time/group/spatial columns that should be used as split strata instead of predictors.
7. Preprocess after the split. Fit transformations on training data only. Preserve raw names and display names separately.
8. Benchmark at least XGBoost plus reasonable alternatives when available: LightGBM, CatBoost, Random Forest, and a simple baseline. Use the same splits, scoring rules, and random seeds selected in the method decision log. Prefer time, group, or spatial holdout over random holdout whenever the data profile indicates dependence.
9. Select the final explainable model. Prefer a gradient-boosted tree model for SHAP reliability. If a non-GBDT model wins, report the tradeoff and either choose the best compatible GBDT for SHAP or ask the user to approve the alternative.
10. Evaluate the final model on the appropriate scale. When target transforms such as `log1p` are used, report transformed-scale and original-scale metrics separately. For probability/risk outputs, include calibration evidence.
11. Compute SHAP:
   - Main effects: use `TreeExplainer(..., feature_perturbation="interventional", data=X_background)` when supported.
   - Interactions: use a separate tree-path-dependent explainer for `shap_interaction_values`; do not expect interventional mode to produce valid interaction values.
   - Large data: sample deterministically for SHAP plots and record the sample size and seed.
12. Generate figures using the domain visual style, style rules, and helper script. Do not call force plots or SHAP's built-in waterfall plot. Use ALE rather than PDP when strong feature dependence exists. When the user asks for figure styling, visual polishing, or examples, inspect `assets/reference-gallery/README.md` and any local user-provided images as visual references. Use only self-created, generated, licensed, or explicitly authorized images; treat them as style guidance only and do not let them override leakage, validation, SHAP, PDP/ALE, or audit rules.
13. Write the report plus model card, risk-of-bias/applicability checklist when relevant, and reproducibility manifest.
14. Run phase-aware validation with `scripts/validate_outputs.py <run-dir> --phase bootstrap|modeling|final`. Cross-check report tables against source CSV/JSON values by column name, not by position.

## Tiered Outputs

- `quick`: `data_profile.json`, `dataset_datasheet.md`, `feature_decision_table.csv`, `modeling_plan.md`, basic model metrics, core SHAP figures/tables, and a short report or notes file.
- `standard`: all `quick` outputs plus `domain_context.json`, `literature_search_plan.json`, `method_decision_log.json`, preprocessing metadata, model benchmark, saved final model/config, reproducibility manifest, and `output_audit.json`.
- `research`: all `standard` outputs plus `evidence_bank.json`, `citation_support_table.csv` when literature tools are available, model card, risk-of-bias/applicability checklist when relevant, uncertainty/stability evidence, advanced SHAP outputs when justified, and a Markdown or HTML report with causal limitation and evidence summary.

## Mandatory Outputs For A Full Analysis

- `data_profile.json`, `dataset_datasheet.md`, and a human-readable data quality scorecard.
- `domain_context.json` with detected domain, three closest recent papers when search is available, domain method adjustments, and visual style.
- `literature_search_plan.json`, `evidence_bank.json`, and `citation_support_table.csv` when literature tools are available.
- `method_decision_log.json` with literature-backed justification for validation, modeling, and explanation choices.
- `feature_decision_table.csv` with keep/drop/transform/strata-only decisions.
- `preprocessing_pipeline.pkl` or equivalent reproducible preprocessing metadata.
- `model_benchmark.csv`, `best_model.pkl`, and `model_config.json`.
- Evaluation metrics and diagnostic figures appropriate to the task.
- SHAP values or sampled SHAP arrays, global importance table, manual waterfall sample table, interaction table when interactions are requested, and auto-insights JSON.
- SVG and PNG copies of every final figure.
- `reproducibility_manifest.json`, `model_card.md`, and `risk_of_bias_checklist.md` for high-stakes or prediction-model reporting tasks.
- `output_audit.json` from the final validation script.
- A Markdown or HTML report with a causal limitation section, data/method summary, evidence summary, model evaluation, SHAP interpretation, domain-specific conclusions, and a quality checklist.

## Quality Gates

Before finishing, verify:

- No feature engineering or preprocessing was fit on test data.
- `dataset_datasheet.md` exists and records target definition, sample unit, prediction timing, grouping/time/space structure, units, and leakage review status.
- `domain_context.json` exists and either contains three nearest recent papers or explicitly states why live search was unavailable.
- `evidence_bank.json` separates nearest recent domain evidence from method-foundation and reporting-standard evidence.
- The validation split matches the data dependence structure: time, spatial, group, external, or random only when justified.
- Feature names in plots use display labels; logs and machine-readable files retain raw column names.
- All SVG files are free of U+2212 minus signs.
- Waterfall plots are manual matplotlib barh plots, not SHAP built-ins.
- Force plots are absent.
- Interaction calculations use the corrected per-sample decomposition from `references/06_shap_advanced.md`.
- Strongly correlated features use grouped interpretation and/or ALE support rather than unqualified single-feature claims.
- Probability/risk reports include calibration evidence.
- Uncertainty is reported where relevant: metric confidence intervals, SHAP rank stability, prediction intervals, or explicit reason unavailable.
- High-stakes tasks include an interpretable baseline or a written justification for using a black-box model.
- Time/space/group variables are treated as strata unless explicitly approved as predictors.
- Report tables use explicit column names and pass source cross-checks.
- Figure palette, plot types, and grouping variables match the detected domain and nearest-paper conventions where practical.
- Any causal language is framed as association unless the study design justifies stronger claims.
- `reproducibility_manifest.json` records dataset hash, skill hash, package versions, random seed, and output file hashes.
- `output_audit.json` has no error-level findings, or remaining findings are explicitly reported to the user.
