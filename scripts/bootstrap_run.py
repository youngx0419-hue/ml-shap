"""Bootstrap an ML-SHAP run directory before modeling."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


SUBDIRS = ["01_Data", "02_Processing", "03_Models", "04_SHAP", "05_Results/Charts", "05_Results/Tables", "05_Results/Reports", "logs"]


def run(cmd: list[str]) -> None:
    print("RUN", " ".join(str(x) for x in cmd))
    subprocess.check_call(cmd)


def script(name: str) -> Path:
    return Path(__file__).resolve().parent / name


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def draft_feature_decision(role: str) -> str:
    if role == "target":
        return "target"
    if role == "id_candidate":
        return "drop"
    if role in {"time_candidate", "group_candidate", "spatial_or_group_candidate"}:
        return "strata-only"
    return "keep-review"


def write_feature_decision_template(run_dir: Path, profile: dict) -> None:
    path = run_dir / "feature_decision_table.csv"
    rows = []
    for col in profile.get("columns", []):
        role = col.get("role_guess", "")
        rows.append(
            {
                "raw_feature": col.get("name", ""),
                "role_guess": role,
                "draft_decision": draft_feature_decision(role),
                "final_decision": "",
                "reason": "Auto-drafted from dataset profile; confirm before modeling.",
                "prediction_time_available": "",
                "leakage_review": "needs_review",
                "display_name": col.get("name", ""),
            }
        )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["raw_feature", "role_guess", "draft_decision", "final_decision", "reason", "prediction_time_available", "leakage_review", "display_name"])
        writer.writeheader()
        writer.writerows(rows)


def write_modeling_plan(run_dir: Path, tier: str, profile: dict, method_log: dict) -> None:
    path = run_dir / "modeling_plan.md"
    target = profile.get("target") or ", ".join(c.get("name", "") for c in profile.get("target_candidates", [])) or "not confirmed"
    lines = [
        "# Modeling Plan",
        "",
        f"- Tier: `{tier}`",
        f"- Target: `{target}`",
        f"- Rows: {profile.get('n_rows')}",
        f"- Columns: {profile.get('n_columns')}",
        f"- Time structure detected: {profile.get('has_time_order')}",
        f"- Group/spatial structure detected: {profile.get('has_group_structure') or profile.get('has_spatial_structure')}",
        "",
        "## Before Modeling",
        "- Confirm `feature_decision_table.csv`, especially ID, time, group, spatial, and leakage-flagged columns.",
        "- Split data before fitting imputers, encoders, scalers, selectors, SMOTE, or target transforms.",
        "- Use the split strategy recommended below unless the project context justifies a stricter one.",
        "",
        "## Recommended Split Strategy",
    ]
    split_rules = method_log.get("split_strategy", [])
    lines.extend([f"- {item.get('recommendation', '')}" for item in split_rules] or ["- No split recommendation generated; review `method_decision_log.json`."])
    lines.extend(
        [
            "",
            "## Recommended Model Strategy",
        ]
    )
    model_rules = method_log.get("model_strategy", [])
    lines.extend([f"- {item.get('recommendation', '')}" for item in model_rules] or ["- Benchmark a simple baseline and a SHAP-compatible tree model."])
    lines.extend(
        [
            "",
            "## Recommended Explanation Strategy",
        ]
    )
    explanation_rules = method_log.get("explanation_strategy", [])
    lines.extend([f"- {item.get('recommendation', '')}" for item in explanation_rules] or ["- Use TreeExplainer and record background/sample/seed choices."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report_todo(run_dir: Path, tier: str) -> None:
    path = run_dir / "report_todo.md"
    lines = [
        "# Report TODO",
        "",
        "- Confirm target definition, prediction timing, row unit, units, and feature availability.",
        "- Summarize data quality issues from `data_profile.json` and `dataset_datasheet.md`.",
        "- Explain the split strategy and why it matches time/group/spatial structure.",
        "- Report model benchmark results and final model selection tradeoffs.",
        "- Report evaluation metrics on the correct scale; include calibration for probability or risk outputs.",
        "- Interpret SHAP as associations unless the study design supports causal claims.",
        "- Cross-check all report tables against source CSV/JSON by column name.",
    ]
    if tier == "research":
        lines.extend(
            [
                "- Fill three nearest recent papers in `domain_context.json` and `evidence_bank.json` when search is available.",
                "- Add evidence summary, uncertainty/stability results, model card, and risk-of-bias/applicability checklist when relevant.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def method_context(profile: dict, target: str | None, domain_context: dict) -> dict:
    target_name = target or profile.get("target")
    task = "unknown"
    for candidate in profile.get("target_candidates", []):
        if not target_name or candidate.get("name") == target_name:
            task = candidate.get("task_type_guess", "unknown")
            break
    if task == "classification_candidate":
        task = "classification"
    return {
        "task_type": task,
        "n_rows": profile.get("n_rows"),
        "n_features": max(0, int(profile.get("n_columns", 1)) - 1),
        "has_time_order": profile.get("has_time_order", False),
        "has_group_structure": profile.get("has_group_structure", False),
        "has_spatial_structure": profile.get("has_spatial_structure", False),
        "has_high_cardinality_categoricals": any(c.get("role_guess") == "categorical_feature_candidate" and c.get("unique_count", 0) > 30 for c in profile.get("columns", [])),
        "max_abs_correlation": profile.get("correlation_summary", {}).get("max_abs_correlation"),
        "minority_class_ratio": None,
        "uses_target_transform": False,
        "needs_probability": task in {"binary_classification", "multiclass_classification", "classification"},
        "high_stakes": domain_context.get("domain") in {"health_environment", "risk", "finance_business"},
        "external_validation_available": False,
        "domain_goal": "scientific_explanation",
        "domain": domain_context.get("domain"),
        "nearest_recent_papers": domain_context.get("nearest_recent_papers", []),
        "domain_method_adjustments": domain_context.get("domain_method_adjustments", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap an ML-SHAP analysis run.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--target")
    parser.add_argument("--sheet")
    parser.add_argument("--user-text", default="")
    parser.add_argument("--domain")
    parser.add_argument("--tier", choices=["quick", "standard", "research"], default="standard")
    parser.add_argument("--skill-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    for subdir in SUBDIRS:
        (args.run_dir / subdir).mkdir(parents=True, exist_ok=True)

    run([sys.executable, str(script("profile_dataset.py")), str(args.dataset), "--outdir", str(args.run_dir), *([] if not args.target else ["--target", args.target]), *([] if not args.sheet else ["--sheet", args.sheet])])
    profile = load_json(args.run_dir / "data_profile.json")
    columns = ",".join(c["name"] for c in profile.get("columns", []))

    domain_cmd = [sys.executable, str(script("domain_context_engine.py")), "--columns", columns, "--target", args.target or "", "--text", args.user_text, "--output", str(args.run_dir / "domain_context.json")]
    if args.domain:
        domain_cmd.extend(["--domain", args.domain])
    run(domain_cmd)

    run([sys.executable, str(script("evidence_plan.py")), "--domain-context", str(args.run_dir / "domain_context.json"), "--data-profile", str(args.run_dir / "data_profile.json"), "--target", args.target or "", "--output", str(args.run_dir / "literature_search_plan.json")])

    domain_context = load_json(args.run_dir / "domain_context.json")
    context_path = args.run_dir / "method_context.json"
    context_path.write_text(json.dumps(method_context(profile, args.target, domain_context), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    run([sys.executable, str(script("method_decision_engine.py")), "--profile", str(context_path), "--domain-context", str(args.run_dir / "domain_context.json"), "--output", str(args.run_dir / "method_decision_log.json")])
    method_log = load_json(args.run_dir / "method_decision_log.json")

    evidence_bank = args.run_dir / "evidence_bank.json"
    if not evidence_bank.exists():
        evidence_bank.write_text(json.dumps({"papers": [], "method_influence_table": [], "visual_influence_table": [], "claims_supported": [], "claims_not_supported": []}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    write_feature_decision_template(args.run_dir, profile)
    write_modeling_plan(args.run_dir, args.tier, profile, method_log)
    write_report_todo(args.run_dir, args.tier)
    run([sys.executable, str(script("reproducibility_manifest.py")), "--run-dir", str(args.run_dir), "--dataset", str(args.dataset), "--skill-dir", str(args.skill_dir), "--seed", str(args.seed)])
    print(f"Bootstrapped ML-SHAP run at {args.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
