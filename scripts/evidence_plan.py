"""Build a structured literature evidence plan for ML-SHAP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FOUNDATION_SLOTS = [
    {"slot": "gradient_boosting_foundation", "purpose": "Justify XGBoost/LightGBM/CatBoost/RF benchmark design."},
    {"slot": "shap_foundation", "purpose": "Justify TreeExplainer assumptions, background data, and interaction limits."},
    {"slot": "validation_leakage_foundation", "purpose": "Justify leakage-safe preprocessing and split design."},
    {"slot": "xai_caution_foundation", "purpose": "Prevent causal overclaiming and unsupported black-box trust claims."},
    {"slot": "reporting_standard", "purpose": "Anchor transparent reporting, risk of bias, applicability, and reproducibility."},
]

DOMAIN_TERMS = {
    "carbon": "carbon emissions OR carbon neutrality OR energy intensity",
    "water": "water quality OR wastewater treatment OR sludge production OR process optimization",
    "risk": "hazard susceptibility OR disaster risk OR vulnerability",
    "spatial_temporal": "spatiotemporal pattern OR remote sensing OR land use",
    "biochar_materials": "biochar adsorption OR pyrolysis OR material performance",
    "health_environment": "environmental exposure OR health risk OR disease prediction",
    "agriculture_ecology": "crop yield OR soil property OR ecosystem prediction",
    "energy_systems": "energy load OR renewable power OR electricity demand",
    "finance_business": "credit risk OR customer churn OR fraud prediction",
    "general": "tabular prediction",
}


def read_json(path: Path | None) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path and path.exists() else {}


def build_plan(domain_context: dict[str, Any], data_profile: dict[str, Any], target: str = "") -> dict[str, Any]:
    domain = domain_context.get("domain") or "general"
    target = target or data_profile.get("target") or ""
    terms = DOMAIN_TERMS.get(domain, DOMAIN_TERMS["general"])
    data_bits = ["tabular"]
    if data_profile.get("has_time_order"):
        data_bits.append("temporal")
    if data_profile.get("has_spatial_structure"):
        data_bits.append("spatial")
    if data_profile.get("has_group_structure"):
        data_bits.append("grouped")
    structure = " OR ".join(data_bits)
    slots = []
    for i in range(1, 4):
        slots.append({
            "slot": f"nearest_recent_domain_paper_{i}",
            "query": f'("{terms}") AND (XGBoost OR SHAP OR "explainable machine learning" OR XAI OR "gradient boosting") AND ({structure})',
            "recency_rule": "Prefer 2024-2026; broaden to 2021-2026 if needed.",
            "fields_to_fill": ["title", "year", "venue", "doi_or_url", "abstract_or_key_claim", "why_close", "similarity_score_0_100", "method_pattern", "visual_pattern", "transferable_to_this_dataset", "caution"],
        })
    return {
        "domain": domain,
        "target": target,
        "data_structure": structure,
        "minimum_evidence_contract": {"nearest_recent_domain_papers": 3, "method_foundation_sources": 3, "xai_caution_sources": 1, "reporting_or_risk_standard_sources": 1},
        "domain_paper_slots": slots,
        "method_foundation_slots": FOUNDATION_SLOTS,
        "evidence_bank_schema": {"papers": [], "method_influence_table": [], "visual_influence_table": [], "claims_supported": [], "claims_not_supported": []},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an evidence search plan for ML-SHAP.")
    parser.add_argument("--domain-context", type=Path)
    parser.add_argument("--data-profile", type=Path)
    parser.add_argument("--target", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = build_plan(read_json(args.domain_context), read_json(args.data_profile), args.target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
