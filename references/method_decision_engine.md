# Literature-Backed Method Decision Engine

Read this file after profiling the dataset and before model training. Its purpose is to prevent the workflow from blindly applying XGBoost-SHAP when the data design, task type, or user goal requires a different validation strategy or explanation method.

## Inputs To Collect

Build or infer a `method_context` object with these fields when possible:

```json
{
  "task_type": "regression | binary_classification | multiclass_classification | forecasting | unknown",
  "n_rows": 0,
  "n_features": 0,
  "has_time_order": false,
  "has_group_structure": false,
  "has_spatial_structure": false,
  "has_high_cardinality_categoricals": false,
  "max_abs_correlation": 0.0,
  "minority_class_ratio": null,
  "uses_target_transform": false,
  "needs_probability": false,
  "high_stakes": false,
  "external_validation_available": false,
  "domain_goal": "prediction | scientific_explanation | decision_support | reporting"
}
```

## Split And Validation Rules

| Condition | Validation Choice | Required Action |
|---|---|---|
| `has_time_order=true` and task predicts future records | Forward-chaining holdout plus `TimeSeriesSplit` or rolling-origin CV | Do not shuffle. Keep all preprocessing inside each fold. Report time horizon and gap. |
| `has_spatial_structure=true` | Spatial or environmental block CV | Report random CV only as optimistic sensitivity analysis. Keep coordinates as strata unless approved predictors. |
| `has_group_structure=true` | `GroupKFold` or grouped holdout | No group may appear in both train and test. Use `StratifiedGroupKFold` for imbalanced classification if feasible. |
| Small data with no dependency structure | Repeated K-fold or nested CV | Report variability and avoid over-tuning. |
| External validation available | Train/internal tune, then lock and evaluate externally | Do not use external data for feature selection or hyperparameter tuning. |

## Model Selection Rules

| Condition | Preferred Models | Rationale |
|---|---|---|
| General tabular data | XGBoost, LightGBM, CatBoost, Random Forest, simple baseline | Gradient-boosted trees are strong tabular baselines; RF and simple baseline reveal overfitting or unnecessary complexity. |
| Large rows/features or runtime pressure | LightGBM + XGBoost | LightGBM is designed for efficient GBDT training; verify accuracy parity under same folds. |
| High-cardinality categorical predictors | CatBoost first, then XGBoost/LightGBM with leakage-safe encoding | CatBoost ordered categorical handling reduces target-statistic leakage risk. |
| Severe class imbalance | Class weights, threshold tuning, PR-AUC, optional fold-local SMOTE | Never resample before splitting. Report minority-class recall/precision and calibration. |
| High-stakes decision support | Add inherently interpretable baseline | If black-box performance gain is small, prefer the interpretable model or justify the tradeoff. |
| Scientific explanation goal | Favor stability, sensitivity analysis, and simpler baselines | Explanation is not credible if model generalization or feature ranking is unstable. |

## SHAP And Explanation Rules

| Condition | Explanation Plan |
|---|---|
| Tree ensemble selected | Use `TreeExplainer`. Record mode, background sample, model output scale, sample size, and random seed. |
| Main effects requested | Prefer interventional TreeExplainer with a representative background sample. |
| Interaction effects requested | Use separate tree-path-dependent explainer for `shap_interaction_values`; limit to sampled rows and top features. |
| Strong feature dependence (`max_abs_correlation >= 0.7`) | Group correlated features, add dependence caveat, compare ALE with PDP, and avoid single-feature causal language. |
| Many near-duplicate formula variables | Create feature families, remove target-derived variables, and interpret family-level effects. |
| Probability/risk communication | Explain calibrated probabilities after calibration, not raw uncalibrated scores. |
| Local individual explanation | Provide waterfall plus counterfactual/domain caveat; do not generalize one sample to global behavior. |
| Global explanation | Combine SHAP bar/beeswarm with permutation importance and feature-ranking stability across models. |

## Plot Choice Rules

| Question | Preferred Plot |
|---|---|
| Which features matter globally? | SHAP beeswarm + SHAP bar + validation-set permutation importance |
| How does a feature affect prediction? | ALE if features are correlated; PDP/ICE only with dependence caveat |
| Are interactions important? | SHAP interaction matrix and dependence plots colored by top interacting feature |
| Why this sample? | Manual matplotlib waterfall; avoid SHAP force plot |
| Does explanation vary by city/season/group? | Strata heatmap and grouped SHAP summaries; do not treat group label as causal feature |
| Is the model extrapolating spatially? | Area-of-applicability or predictor-space dissimilarity map/table |

## Minimum Advanced Quality Gates

- Build `method_decision_log.json` before modeling and include it in the report appendix.
- State why the chosen split is appropriate for the data dependence structure.
- State why the chosen model family is appropriate for the user goal.
- For SHAP, state the feature perturbation mode and background data source.
- For correlated features, add ALE or grouped feature interpretation.
- For imbalanced classification, include PR-AUC, confusion matrix at chosen threshold, and calibration evidence.
- For high-stakes tasks, include an inherently interpretable baseline or explicit reason it is insufficient.
- For scientific explanation, include stability checks and avoid causal language unless causal assumptions are explicit.
