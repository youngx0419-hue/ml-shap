"""Infer domain context and generate literature search queries/style hints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "carbon": ["carbon", "co2", "emission", "energy intensity", "gdp", "population", "carbon peak", "carbon neutrality", "\u78b3", "\u4e8c\u6c27\u5316\u78b3", "\u6392\u653e", "\u80fd\u6e90", "\u78b3\u8fbe\u5cf0", "\u78b3\u4e2d\u548c"],
    "water": ["do", "cod", "bod", "nh3", "tn", "tp", "ph", "turbidity", "conductivity", "river", "lake", "basin", "water", "sludge", "wastewater", "hydrocyclone", "desanding", "\u6c61\u6ce5", "\u5e9f\u6c34", "\u6c61\u6c34", "\u6c34\u8d28", "\u6eb6\u89e3\u6c27", "\u6c28\u6c2e", "\u603b\u78f7", "\u603b\u6c2e", "\u65cb\u6d41", "\u9664\u7802"],
    "risk": ["risk", "hazard", "susceptibility", "vulnerability", "flood", "landslide", "wildfire", "earthquake", "disaster", "accident", "\u98ce\u9669", "\u707e\u5bb3", "\u6613\u53d1\u6027", "\u8106\u5f31\u6027", "\u4e8b\u6545"],
    "spatial_temporal": ["ndvi", "landuse", "land use", "landscape", "urban", "biomass", "remote sensing", "moran", "spatial", "temporal", "\u7ecf\u7eac\u5ea6", "\u7a7a\u95f4", "\u65f6\u7a7a", "\u9065\u611f", "\u571f\u5730\u5229\u7528"],
    "biochar_materials": ["biochar", "adsorption", "pyrolysis", "bet", "ssa", "pore", "feedstock", "activation", "removal", "\u751f\u7269\u70ad", "\u5438\u9644", "\u70ed\u89e3", "\u6bd4\u8868\u9762\u79ef", "\u53bb\u9664\u7387"],
    "health_environment": ["patient", "clinical", "disease", "mortality", "morbidity", "exposure", "hospital", "pm2.5", "health", "\u75be\u75c5", "\u6b7b\u4ea1\u7387", "\u66b4\u9732", "\u5065\u5eb7", "\u60a3\u8005", "\u4e34\u5e8a"],
    "agriculture_ecology": ["crop", "yield", "soil", "fertilizer", "irrigation", "vegetation", "biodiversity", "ecosystem", "\u519c\u4e1a", "\u4f5c\u7269", "\u4ea7\u91cf", "\u571f\u58e4", "\u751f\u6001"],
    "energy_systems": ["load", "power", "electricity", "renewable", "wind", "solar", "pv", "battery", "grid", "\u7535\u529b", "\u8d1f\u8377", "\u5149\u4f0f", "\u98ce\u7535"],
    "finance_business": ["credit", "default", "churn", "fraud", "sales", "customer", "transaction", "revenue", "\u91d1\u878d", "\u8fdd\u7ea6", "\u6b3a\u8bc8", "\u5ba2\u6237"],
}

STYLE_CARDS: dict[str, dict[str, Any]] = {
    "carbon": {"palette_name": "carbon", "figure_facecolor": "#fbf4ee", "primary": "#b5533c", "secondary": "#4c4c4c", "accent": "#d9a441", "map_cmap": "OrRd", "notes": ["Use scenario lines and policy threshold annotations."]},
    "water": {"palette_name": "water", "figure_facecolor": "#f2f7f7", "primary": "#287c8e", "secondary": "#6bb7a8", "accent": "#d9a441", "map_cmap": "YlGnBu", "notes": ["Use process-stage, station, basin, or hydrological-period grouping when present."]},
    "risk": {"palette_name": "risk", "figure_facecolor": "#f8f5f0", "primary": "#d95f0e", "secondary": "#525252", "accent": "#fec44f", "map_cmap": "YlOrRd", "notes": ["Show probability calibration, thresholds, uncertainty, and applicability limits."]},
    "spatial_temporal": {"palette_name": "spatial_temporal", "figure_facecolor": "#f5f7f6", "primary": "#3b7f7f", "secondary": "#6a6a6a", "accent": "#8c6bb1", "map_cmap": "viridis", "notes": ["Prefer faceted maps and space-time heatmaps."]},
    "biochar_materials": {"palette_name": "biochar_materials", "figure_facecolor": "#f7f5f8", "primary": "#705898", "secondary": "#4f8a5b", "accent": "#c49a3a", "map_cmap": "PuBuGn", "notes": ["Use process contour plots and material feature families."]},
    "health_environment": {"palette_name": "health_environment", "figure_facecolor": "#f4f7fb", "primary": "#2b6cb0", "secondary": "#9b2c2c", "accent": "#718096", "map_cmap": "Blues", "notes": ["Emphasize calibration, subgroups, fairness, and uncertainty."]},
    "agriculture_ecology": {"palette_name": "agriculture_ecology", "figure_facecolor": "#f5f8f1", "primary": "#4f7f3a", "secondary": "#8b6f47", "accent": "#4aa3c7", "map_cmap": "YlGn", "notes": ["Use seasonal panels and vegetation/soil gradients."]},
    "energy_systems": {"palette_name": "energy_systems", "figure_facecolor": "#f3f6fa", "primary": "#276fbf", "secondary": "#f0a202", "accent": "#4a5568", "map_cmap": "cividis", "notes": ["Use time profiles, scenario fan charts, and seasonal grouping."]},
    "finance_business": {"palette_name": "finance_business", "figure_facecolor": "#f6f7f9", "primary": "#2f5f8f", "secondary": "#2f855a", "accent": "#c53030", "map_cmap": "Blues", "notes": ["Use lift, calibration, and segment-level SHAP."]},
    "general": {"palette_name": "general", "figure_facecolor": "#f8f7f4", "primary": "#3f6f8f", "secondary": "#6a8f5a", "accent": "#d9a441", "map_cmap": "viridis", "notes": ["Use robust general XAI plots."]},
}


def read_columns(csv_path: Path, limit: int = 200) -> list[str]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader)[:limit]


def infer_domain(columns: list[str], target: str = "", user_text: str = "", user_override: str | None = None) -> dict[str, Any]:
    if user_override:
        domain = normalize_domain(user_override)
        return build_context(domain, 1.0, [], target, user_text, user_override)
    text = " ".join(columns + [target, user_text]).lower()
    scores: dict[str, int] = {}
    evidence: dict[str, list[str]] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        hits = [kw for kw in keywords if kw.lower() in text]
        scores[domain] = len(hits)
        evidence[domain] = hits[:12]
    best = max(scores, key=scores.get) if scores else "general"
    total = sum(scores.values())
    if scores.get(best, 0) == 0:
        best = "general"
        confidence = 0.35
    else:
        confidence = min(0.95, 0.45 + scores[best] / max(total, 1) * 0.5)
    return build_context(best, round(confidence, 3), evidence.get(best, []), target, user_text, None, scores)


def normalize_domain(value: str) -> str:
    value = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {"materials": "biochar_materials", "biochar": "biochar_materials", "water_quality": "water", "wastewater": "water", "sludge": "water", "spatiotemporal": "spatial_temporal", "spatial": "spatial_temporal", "health": "health_environment", "agriculture": "agriculture_ecology", "ecology": "agriculture_ecology", "energy": "energy_systems", "finance": "finance_business", "business": "finance_business"}
    return aliases.get(value, value if value in DOMAIN_KEYWORDS or value == "general" else "general")


def build_context(domain: str, confidence: float, evidence_columns: list[str], target: str, user_text: str, user_override: str | None, scores: dict[str, int] | None = None) -> dict[str, Any]:
    keywords = domain_query_keywords(domain)
    data_structure = infer_data_structure(user_text)
    queries = [
        f'("{keywords}") AND (XGBoost OR SHAP OR "explainable machine learning" OR XAI) AND ({data_structure})',
        f'"{keywords}" XGBoost SHAP recent study',
        f'"{keywords}" "gradient boosting" "feature importance" 2024 OR 2025 OR 2026',
    ]
    style = STYLE_CARDS.get(domain, STYLE_CARDS["general"]).copy()
    style.update({"axes_facecolor": "white", "diverging_cmap": "RdBu_r", "sequential_cmap": "viridis"})
    return {
        "domain": domain,
        "confidence": confidence,
        "evidence_columns": evidence_columns,
        "domain_scores": scores or {},
        "target_variable": target,
        "user_override": user_override,
        "literature_search_queries": queries,
        "nearest_recent_papers": [],
        "literature_search_status": "pending",
        "evidence_strategy": "Use three nearest recent domain papers plus method-foundation and reporting-standard sources.",
        "domain_method_adjustments": [],
        "domain_visual_style": style,
        "open_questions": [] if confidence >= 0.55 else ["Domain confidence is low; ask the user to confirm the research field."],
    }


def domain_query_keywords(domain: str, target: str = "") -> str:
    base = {
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
    }.get(domain, "tabular prediction")
    return base


def infer_data_structure(user_text: str) -> str:
    text = user_text.lower()
    bits = []
    if any(k in text for k in ["time", "temporal", "forecast", "year", "month", "\u65f6\u95f4", "\u9884\u6d4b"]):
        bits.append("temporal")
    if any(k in text for k in ["spatial", "map", "coordinate", "city", "region", "\u7a7a\u95f4", "\u5730\u56fe", "\u57ce\u5e02", "\u533a\u57df"]):
        bits.append("spatial")
    if any(k in text for k in ["classification", "risk", "binary", "\u5206\u7c7b", "\u98ce\u9669"]):
        bits.append("classification")
    return " OR ".join(bits) if bits else "tabular OR structured data"


def main() -> int:
    parser = argparse.ArgumentParser(description="Infer domain context for ML-SHAP workflows.")
    parser.add_argument("--csv", type=Path, help="CSV file whose header should be used for domain inference.")
    parser.add_argument("--columns", help="Comma-separated column names.")
    parser.add_argument("--target", default="", help="Target variable name.")
    parser.add_argument("--text", default="", help="User prompt or project description.")
    parser.add_argument("--domain", help="User-provided domain override.")
    parser.add_argument("--output", type=Path, help="Optional output JSON path.")
    args = parser.parse_args()
    columns: list[str] = []
    if args.csv:
        columns = read_columns(args.csv)
    if args.columns:
        columns.extend([c.strip() for c in args.columns.split(",") if c.strip()])
    context = infer_domain(columns, target=args.target, user_text=args.text, user_override=args.domain)
    payload = json.dumps(context, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
