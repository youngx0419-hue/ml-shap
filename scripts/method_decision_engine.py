"""Generate literature-backed method recommendations for XGBoost-SHAP workflows.

Input is a JSON object with dataset/task flags. The output is a JSON decision log
that can be saved as method_decision_log.json and cited in the report appendix.
This script uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def add_rule(log: dict[str, Any], category: str, trigger: str, recommendation: str, evidence: str) -> None:
    log.setdefault(category, []).append(
        {
            "trigger": trigger,
            "recommendation": recommendation,
            "evidence": evidence,
        }
    )


def recommend(context: dict[str, Any]) -> dict[str, Any]:
    task = str(context.get("task_type", "unknown")).lower()
    domain = str(context.get("domain", "general")).lower()
    max_corr = _number(context.get("max_abs_correlation"), 0.0) or 0.0
    minority_ratio = _number(context.get("minority_class_ratio"), None)
    n_rows = _number(context.get("n_rows"), 0) or 0
    n_features = _number(context.get("n_features"), 0) or 0

    log: dict[str, Any] = {
        "input_context": context,
        "domain_context": {
            "domain": domain,
            "nearest_recent_papers": context.get("nearest_recent_papers", []),
            "domain_method_adjustments": context.get("domain_method_adjustments", []),
        },
        "split_strategy": [],
        "model_strategy": [],
        "explanation_strategy": [],
        "reporting_strategy": [],
        "quality_gates": [],
    }

    has_time = _truthy(context.get("has_time_order"))
    has_group = _truthy(context.get("has_group_structure"))
    has_spatial = _truthy(context.get("has_spatial_structure"))
    high_card_cat = _truthy(context.get("has_high_cardinality_categoricals"))
    needs_probability = _truthy(context.get("needs_probability"))
    high_stakes = _truthy(context.get("high_stakes"))
    external_validation = _truthy(context.get("external_validation_available"))
    scientific = str(context.get("domain_goal", "")).lower() in {"scientific_explanation", "explanation", "science"}

    if domain in {"water", "risk", "spatial_temporal", "carbon"}:
        add_rule(
            log,
            "split_strategy",
            f"domain={domain}",
            "Check whether space, time, basin/region, station, or policy grouping requires blocked or grouped validation before allowing random CV.",
            "Domain-aware workflow; Roberts et al. 2017; Valavi et al. 2019",
        )
    if domain == "biochar_materials":
        add_rule(
            log,
            "model_strategy",
            "domain=biochar_materials",
            "Prioritize CatBoost or leakage-safe encoding because feedstock, activator, atmosphere, and modification type are often important categorical predictors.",
            "Domain-aware workflow; Prokhorenkova et al. 2018",
        )
    if domain == "risk":
        needs_probability = True
        add_rule(
            log,
            "reporting_strategy",
            "domain=risk",
            "Report calibrated risk probabilities, threshold choice, uncertainty, and risk-zone interpretation.",
            "Domain-aware workflow; Niculescu-Mizil and Caruana 2005",
        )

    if has_time or task == "forecasting":
        add_rule(
            log,
            "split_strategy",
            "time order or forecasting task",
            "Use forward-chaining holdout plus TimeSeriesSplit or rolling-origin validation; do not shuffle.",
            "scikit-learn TimeSeriesSplit; Roberts et al. 2017; Bergmeir et al. 2018",
        )
    if has_spatial:
        add_rule(
            log,
            "split_strategy",
            "spatial dependence",
            "Use spatial or environmental block cross-validation and report random CV only as optimistic sensitivity analysis.",
            "Roberts et al. 2017; Valavi et al. 2019; Meyer and Pebesma 2021",
        )
    if has_group:
        add_rule(
            log,
            "split_strategy",
            "grouped observations",
            "Use GroupKFold or grouped holdout; use StratifiedGroupKFold when grouped classification is imbalanced.",
            "scikit-learn GroupKFold and StratifiedGroupKFold documentation",
        )
    if external_validation:
        add_rule(
            log,
            "split_strategy",
            "external validation data available",
            "Use external data only once after model and hyperparameters are locked.",
            "TRIPOD+AI 2024; PROBAST+AI 2025",
        )
    if not any([has_time, has_spatial, has_group, external_validation]):
        add_rule(
            log,
            "split_strategy",
            "no detected dependency structure",
            "Use repeated K-fold or nested CV when tuning is heavy; keep a final holdout if enough data exist.",
            "scikit-learn common pitfalls and cross-validation guidance",
        )

    if high_card_cat:
        add_rule(
            log,
            "model_strategy",
            "high-cardinality categorical predictors",
            "Prioritize CatBoost and compare with XGBoost/LightGBM using leakage-safe encoding.",
            "Prokhorenkova et al. 2018",
        )
    if n_rows >= 100000 or n_features >= 500:
        add_rule(
            log,
            "model_strategy",
            "large dataset or high feature count",
            "Include LightGBM for efficient GBDT training and compare accuracy under identical folds.",
            "Ke et al. 2017",
        )
    add_rule(
        log,
        "model_strategy",
        "general tabular benchmark",
        "Benchmark XGBoost, LightGBM, CatBoost, Random Forest, and a simple baseline when dependencies are available.",
        "Chen and Guestrin 2016; Ke et al. 2017; Prokhorenkova et al. 2018; Breiman 2001",
    )
    if minority_ratio is not None and task in {"binary_classification", "multiclass_classification"} and minority_ratio < 0.2:
        add_rule(
            log,
            "model_strategy",
            "class imbalance",
            "Use class weights, threshold tuning, PR-AUC, and optional fold-local SMOTE; never resample before splitting.",
            "Chawla et al. 2002; imbalanced-learn and scikit-learn leakage guidance",
        )
    if high_stakes:
        add_rule(
            log,
            "model_strategy",
            "high-stakes use",
            "Add an inherently interpretable baseline and justify any black-box performance tradeoff.",
            "Rudin 2019; TRIPOD+AI 2024; PROBAST+AI 2025",
        )

    papers = context.get("nearest_recent_papers") or []
    if papers:
        add_rule(
            log,
            "reporting_strategy",
            "nearest recent domain papers found",
            "Summarize how the three closest recent papers influenced feature families, validation design, explanation plots, and visual style.",
            "domain_context.json nearest_recent_papers",
        )

    add_rule(
        log,
        "explanation_strategy",
        "tree ensemble explanation",
        "Use TreeExplainer; record feature perturbation mode, background sample, model output scale, row sample, and seed.",
        "Lundberg and Lee 2017; Lundberg et al. 2018; SHAP TreeExplainer documentation",
    )
    if max_corr >= 0.7:
        add_rule(
            log,
            "explanation_strategy",
            "strong feature dependence",
            "Group correlated features, prefer ALE over PDP, and add dependence caveats to SHAP ranks.",
            "Aas et al. 2021; Apley and Zhu 2020; Molnar et al. 2020",
        )
    if needs_probability or task in {"binary_classification", "multiclass_classification"}:
        add_rule(
            log,
            "explanation_strategy",
            "probability or risk communication",
            "Report calibration curve and Brier score; calibrate probabilities on validation data if needed.",
            "Niculescu-Mizil and Caruana 2005",
        )
    if scientific:
        add_rule(
            log,
            "explanation_strategy",
            "scientific explanation goal",
            "Add stability bootstrap, cross-model ranking comparison, and explicit non-causal wording.",
            "Janzing et al. 2020; Fisher et al. 2019; Molnar et al. 2020",
        )

    add_rule(
        log,
        "reporting_strategy",
        "all workflows",
        "Include data source, outcome, predictors, missing data, split design, model tuning, performance, limitations, and reproducibility details.",
        "TRIPOD+AI 2024",
    )
    add_rule(
        log,
        "quality_gates",
        "before final response",
        "Verify no leakage, locked validation, declared SHAP assumptions, calibrated probabilities when needed, and no causal overclaiming.",
        "Kaufman et al. 2012; scikit-learn common pitfalls; Janzing et al. 2020",
    )

    return log


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a literature-backed method decision log.")
    parser.add_argument("--profile", type=Path, help="JSON file containing method_context fields.")
    parser.add_argument("--domain-context", type=Path, help="Optional domain_context.json to merge into the method context.")
    parser.add_argument("--output", type=Path, help="Optional output JSON path.")
    parser.add_argument("--set", action="append", default=[], help="Override context key=value. Repeat as needed.")
    args = parser.parse_args()

    context: dict[str, Any] = {}
    if args.profile:
        context.update(json.loads(args.profile.read_text(encoding="utf-8")))
    if args.domain_context:
        domain_context = json.loads(args.domain_context.read_text(encoding="utf-8"))
        context.update(
            {
                "domain": domain_context.get("domain"),
                "nearest_recent_papers": domain_context.get("nearest_recent_papers", []),
                "domain_method_adjustments": domain_context.get("domain_method_adjustments", []),
                "domain_visual_style": domain_context.get("domain_visual_style", {}),
            }
        )
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got: {item}")
        key, value = item.split("=", 1)
        context[key.strip()] = value.strip()

    result = recommend(context)
    payload = json.dumps(result, ensure_ascii=True, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
