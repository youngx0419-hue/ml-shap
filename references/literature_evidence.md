# Literature Evidence Map

Use this file when a task needs stronger methodological justification, paper-backed reporting, or a more advanced method choice than the default workflow. Do not load this file for quick operational fixes unless the user asks for literature support.

## Core Gradient Boosting And Tree Models

| Source | What It Supports | Skill Rule |
|---|---|---|
| Chen and Guestrin, 2016, XGBoost: A Scalable Tree Boosting System, https://arxiv.org/abs/1603.02754 | XGBoost is a regularized scalable gradient tree boosting system with sparse-aware learning and efficient approximate split finding. | Use XGBoost as the default strong tabular baseline, especially for mixed-quality structured data. Tune regularization, subsampling, column sampling, learning rate, depth, and early stopping. |
| Friedman, 2001, Greedy Function Approximation: A Gradient Boosting Machine, https://www.jstor.org/stable/2699986 | Gradient boosting frames additive tree modeling as stagewise function optimization and introduced partial dependence as a model inspection tool. | Treat boosted trees as powerful predictive models, not inherently causal models. Use PDP only after checking feature dependence. |
| Ke et al., 2017, LightGBM, https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree | LightGBM uses GOSS and EFB for efficient GBDT training at scale. | Add LightGBM when data are large, high-dimensional, or runtime-constrained; compare against XGBoost under the same folds. |
| Prokhorenkova et al., 2018, CatBoost, https://arxiv.org/abs/1706.09516 | CatBoost introduces ordered boosting and categorical feature handling to reduce target leakage and prediction shift. | Prefer CatBoost when high-cardinality categorical predictors are central and one-hot encoding would be unstable or sparse. |
| Breiman, 2001, Random Forests, https://link.springer.com/article/10.1023/A:1010933404324 | Random forests are strong nonparametric ensembles and introduced common permutation importance practice. | Keep Random Forest as a benchmark and robustness comparator, but do not assume its impurity importance is reliable for interpretation. |

## SHAP, TreeSHAP, And Feature Dependence

| Source | What It Supports | Skill Rule |
|---|---|---|
| Lundberg and Lee, 2017, SHAP, https://arxiv.org/abs/1705.07874 | SHAP unifies additive feature attribution methods and gives local additive attributions under clear assumptions. | Use SHAP for local and global attribution summaries, but always document the background distribution and feature dependence assumption. |
| Lundberg, Erion, and Lee, 2018, TreeSHAP, https://arxiv.org/abs/1802.03888 | TreeSHAP gives fast exact SHAP values for tree ensembles and defines SHAP interaction values. | Use TreeExplainer for tree ensembles; use SHAP interaction values only when runtime and sample size allow. |
| SHAP TreeExplainer documentation, https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html | TreeExplainer supports different assumptions about feature dependence, including interventional and tree-path-dependent modes. | For main effects, prefer interventional TreeExplainer with an explicit background sample when causal/interventional semantics are intended. For interaction values, use tree-path-dependent mode and document this switch. |
| Janzing, Minorics, and Bloebaum, 2020, https://proceedings.mlr.press/v108/janzing20a.html | Feature relevance with Shapley values depends on the causal/interventional versus observational distribution used for missing features. | Never call SHAP "causal" unless the causal graph, intervention target, and assumptions are explicit. Add a causal limitation statement by default. |
| Aas, Jullum, and Loland, 2021, https://arxiv.org/abs/1903.10464 | Kernel SHAP explanations can be wrong under feature dependence; dependence-aware approximations can improve fidelity. | When features are strongly correlated, group correlated features, compare conditional/dependence-aware explanations if available, and avoid over-interpreting single-feature SHAP ranks. |
| Fisher, Rudin, and Dominici, 2019, https://jmlr.org/papers/v20/18-760.html | Variable importance can vary across equally good models; model class reliance characterizes that uncertainty. | Add model-robustness checks: compare feature rankings across XGBoost, LightGBM, CatBoost, Random Forest, and permutation importance on holdout data. |

## PDP, ALE, Calibration, And Imbalance

| Source | What It Supports | Skill Rule |
|---|---|---|
| Apley and Zhu, 2020, ALE, https://arxiv.org/abs/1612.08468 | PDP can extrapolate outside the training distribution when predictors are dependent; ALE avoids this extrapolation problem. | If maximum absolute correlation or domain coupling is high, prefer ALE over PDP; if PDP is shown, add dependence/extrapolation caveats and compare with SHAP dependence. |
| scikit-learn permutation importance example, https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html | Impurity-based importance can inflate continuous or high-cardinality features; permutation importance is a useful external check. | Report permutation importance on validation/holdout data as a robustness check, not as a replacement for SHAP directionality. |
| Chawla et al., 2002, SMOTE, https://www.jair.org/index.php/jair/article/view/10302 | SMOTE creates synthetic minority examples and can improve ROC-space performance for imbalanced classification. | Use SMOTE only inside cross-validation folds or training pipelines; never resample before splitting. Also test class weights or threshold tuning. |
| Niculescu-Mizil and Caruana, 2005, calibration, https://dl.acm.org/doi/10.1145/1102351.1102430 | Some learners, including boosted methods, can produce distorted probability estimates; calibration methods improve probability quality. | For classification reports with probabilities or risk thresholds, include calibration curves, Brier score, and optional Platt/isotonic calibration fitted only on validation data. |

## Leakage, Validation Design, And Structured Data

| Source | What It Supports | Skill Rule |
|---|---|---|
| Kaufman et al., 2012, leakage, https://dl.acm.org/doi/10.1145/2382577.2382579 | Leakage is target information that would not be legitimately available at prediction time; learn-predict separation is core. | Build a leakage ledger before modeling. Drop target-derived columns, post-outcome variables, duplicate target encodings, and formula components that reveal the target. |
| scikit-learn common pitfalls, https://scikit-learn.org/stable/common_pitfalls.html | Inconsistent preprocessing and data leakage are common ML errors; transformations should be learned from training data only. | Use pipelines or explicit fit-on-train/transform-test discipline for imputation, scaling, encoding, feature selection, target transforms, and resampling. |
| scikit-learn GroupKFold, https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html | Groups must not appear in both train and test folds when observations are group-dependent. | Use GroupKFold or grouped holdout when samples share subjects, sites, cities, experiments, batches, or papers. |
| scikit-learn StratifiedGroupKFold, https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html | StratifiedGroupKFold preserves class balance while keeping groups non-overlapping. | Use StratifiedGroupKFold for imbalanced classification with grouped observations when enough groups exist. |
| scikit-learn TimeSeriesSplit, https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html | TimeSeriesSplit preserves temporal order and avoids training on future records. | Use forward-chaining or rolling/expanding validation for time-ordered prediction; random split is only acceptable after proving order is irrelevant. |
| Roberts et al., 2017, spatial/temporal/hierarchical CV, https://nsojournals.onlinelibrary.wiley.com/doi/10.1111/ecog.02881 | Block CV is recommended where temporal, spatial, hierarchical, or phylogenetic dependence exists. | Use block CV when data have spatial, temporal, or hierarchical dependence; report ordinary random CV only as optimistic sensitivity analysis. |
| Valavi et al., 2019, blockCV, https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.13107 | Spatial/environmental blocking creates separated folds for robust spatial evaluation. | For spatial environmental tasks, build spatial or environmental blocks and keep location variables as strata unless causal predictors are justified. |
| Meyer and Pebesma, 2021, area of applicability, https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.13650 | Model transferability depends on whether prediction points lie within the training predictor space. | For spatial transfer or extrapolation, report area-of-applicability or a predictor-space dissimilarity warning. |
| Bergmeir, Hyndman, and Koo, 2018, https://robjhyndman.com/publications/cv-time-series/ | Standard k-fold can be valid in limited autoregressive settings, but validation must respect forecasting assumptions. | Default to forward validation for forecasting; permit k-fold only when the task is not true future forecasting and residual dependence has been checked. |

## Reporting, Risk Of Bias, And XAI Cautions

| Source | What It Supports | Skill Rule |
|---|---|---|
| TRIPOD+AI, 2024, https://www.bmj.com/content/385/bmj-2023-078378 | TRIPOD+AI updates reporting guidance for prediction models using regression or machine learning. | Reports must include transparent data source, participants/samples, predictors, outcome, missing data handling, validation design, performance, limitations, and reproducibility details. |
| PROBAST+AI, 2025, https://www.bmj.com/content/388/bmj-2024-082505 | PROBAST+AI separates model development and evaluation and assesses quality, risk of bias, and applicability. | Add a risk-of-bias/applicability checklist for high-stakes prediction tasks. |
| Lipton, 2016, https://arxiv.org/abs/1606.03490 | Interpretability has multiple meanings and should not be asserted vaguely. | Define what "interpretable" means for the task: debugging, trust, scientific insight, compliance, stakeholder communication, or actionability. |
| Rudin, 2019, https://www.nature.com/articles/s42256-019-0048-x | High-stakes decisions may require inherently interpretable models rather than post-hoc explanations of black boxes. | For high-stakes use, benchmark an interpretable model and explicitly justify why a black-box-plus-SHAP model is acceptable. |
| Adebayo et al., 2018, https://papers.nips.cc/paper/8160-sanity-checks-for-saliency-maps | Visual explanations can be misleading unless sanity checks verify sensitivity to model and data. | Add explanation sanity checks where feasible: model randomization, label shuffle, stability bootstrap, or explanation invariance tests. |
| Molnar et al., 2020, https://arxiv.org/abs/2007.04131 | Interpretation methods can mislead under bad generalization, dependence, interactions, uncertainty, and unjustified causal claims. | Do not interpret a model that generalizes poorly. Attach uncertainty, dependence, and causal caveats to all global and local explanations. |
