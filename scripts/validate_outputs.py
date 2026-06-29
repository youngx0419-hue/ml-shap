"""Audit required ML-SHAP output artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


PHASE_REQUIRED = {
    "bootstrap": ["data_profile.json", "dataset_datasheet.md", "domain_context.json", "literature_search_plan.json", "evidence_bank.json", "method_decision_log.json", "feature_decision_table.csv", "modeling_plan.md", "report_todo.md", "reproducibility_manifest.json"],
    "modeling": ["data_profile.json", "dataset_datasheet.md", "domain_context.json", "method_decision_log.json", "feature_decision_table.csv", "preprocessing_pipeline.pkl|preprocessing_metadata.json", "model_benchmark.csv", "best_model.pkl", "model_config.json", "reproducibility_manifest.json"],
    "final": ["data_profile.json", "dataset_datasheet.md", "domain_context.json", "literature_search_plan.json", "evidence_bank.json", "method_decision_log.json", "feature_decision_table.csv", "preprocessing_pipeline.pkl|preprocessing_metadata.json", "model_benchmark.csv", "best_model.pkl", "model_config.json", "reproducibility_manifest.json", "model_card.md", "report.md|report.html"],
}

CODE_SUFFIXES = {".py", ".ipynb"}


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_error": str(exc)}


def read_csv(path: Path) -> list[dict]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except Exception:
        return []


def first_existing(run_dir: Path, name: str) -> Path | None:
    direct = run_dir / name
    if direct.exists():
        return direct
    matches = list(run_dir.rglob(name))
    return matches[0] if matches else None


def exists_any(run_dir: Path, spec: str) -> bool:
    names = spec.split("|")
    return any((run_dir / name).exists() or list(run_dir.rglob(name)) for name in names)


def add_finding(findings: list[dict], severity: str, item: str, message: str) -> None:
    findings.append({"severity": severity, "item": item, "message": message})


def audit_required_files(run_dir: Path, phase: str, findings: list[dict]) -> None:
    for spec in PHASE_REQUIRED[phase]:
        if not exists_any(run_dir, spec):
            add_finding(findings, "error", spec, "Required artifact not found for this validation phase.")


def audit_domain_context(run_dir: Path, phase: str, findings: list[dict]) -> None:
    domains = list(run_dir.rglob("domain_context.json"))
    if not domains:
        return
    ctx = load_json(domains[0])
    if ctx.get("_error"):
        add_finding(findings, "error", "domain_context.json", f"Could not parse JSON: {ctx['_error']}")
        return
    status = ctx.get("literature_search_status")
    if phase == "final" and status != "not_available" and len(ctx.get("nearest_recent_papers", [])) < 3:
        add_finding(findings, "warning", "domain_context.json", "Final context has fewer than three nearest recent papers recorded.")
    if phase in {"modeling", "final"} and status == "pending":
        add_finding(findings, "warning", "domain_context.json", "Literature search status is still pending; mark not_available or fill nearest papers before final reporting.")


def audit_feature_decisions(run_dir: Path, phase: str, findings: list[dict]) -> None:
    tables = list(run_dir.rglob("feature_decision_table.csv"))
    if not tables:
        return
    with tables[0].open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if phase in {"modeling", "final"} and any(row.get("leakage_review") == "needs_review" for row in rows):
        add_finding(findings, "warning", "feature_decision_table.csv", "Feature table still contains needs_review markers.")
    if phase in {"modeling", "final"} and rows and any(not row.get("final_decision") for row in rows if row.get("draft_decision") != "target"):
        add_finding(findings, "warning", "feature_decision_table.csv", "Final feature decisions are not fully filled.")
    review = first_existing(run_dir, "feature_review.json")
    if review:
        payload = load_json(review)
        for item in payload.get("findings", []):
            if item.get("severity") == "error":
                add_finding(findings, "error", "feature_review.json", f"Feature review error remains: {item.get('feature')} - {item.get('issue')}")


def audit_model_config(run_dir: Path, phase: str, findings: list[dict]) -> None:
    if phase not in {"modeling", "final"}:
        return
    configs = list(run_dir.rglob("model_config.json"))
    if not configs:
        return
    config = load_json(configs[0])
    if config.get("_error"):
        add_finding(findings, "error", "model_config.json", f"Could not parse JSON: {config['_error']}")
        return
    split = config.get("split", {})
    if split.get("preprocessing_fit_scope") != "train_only":
        add_finding(findings, "error", "model_config.json", "preprocessing_fit_scope must be train_only.")
    model_artifact = config.get("selected_model_artifact")
    if model_artifact and not exists_any(run_dir, str(model_artifact)):
        add_finding(findings, "error", "model_config.json", f"Selected model artifact not found: {model_artifact}")
    preprocessing_artifact = config.get("preprocessing_artifact")
    if preprocessing_artifact and not exists_any(run_dir, str(preprocessing_artifact)):
        add_finding(findings, "error", "model_config.json", f"Preprocessing artifact not found: {preprocessing_artifact}")
    shap = config.get("shap", {})
    if shap and shap.get("same_model_for_shap") is not True:
        add_finding(findings, "error", "model_config.json", "SHAP outputs must use the selected model instance.")
    if phase == "final" and shap.get("status") in {"failed", "not_available", "skipped"}:
        add_finding(findings, "warning", "model_config.json", f"SHAP outputs are not complete: {shap.get('status')}.")
    benchmark_path = first_existing(run_dir, "model_benchmark.csv")
    if benchmark_path:
        benchmark = read_csv(benchmark_path)
        models = {row.get("model") for row in benchmark}
        selected = config.get("selected_model")
        if selected and selected not in models:
            add_finding(findings, "error", "model_benchmark.csv", f"Selected model is absent from benchmark table: {selected}")
    profile = first_existing(run_dir, "data_profile.json")
    if profile:
        profile_payload = load_json(profile)
        strategy = split.get("strategy")
        if strategy == "random_holdout" and any(profile_payload.get(flag) for flag in ["has_time_order", "has_group_structure", "has_spatial_structure"]):
            add_finding(findings, "warning", "model_config.json", "Random holdout was used although time/group/spatial structure is detected; record justification if intentional.")


def audit_shap_consistency(run_dir: Path, phase: str, findings: list[dict]) -> None:
    if phase not in {"modeling", "final"}:
        return
    importance_path = first_existing(run_dir, "global_shap_importance.csv")
    insights_path = first_existing(run_dir, "auto_insights.json")
    if not importance_path or not insights_path:
        return
    importance = read_csv(importance_path)
    insights = load_json(insights_path)
    top_features = insights.get("top_features", [])
    if importance and top_features:
        top_table = importance[0].get("feature")
        top_json = top_features[0].get("feature")
        if top_table != top_json:
            add_finding(findings, "error", "auto_insights.json", f"Top SHAP feature does not match global_shap_importance.csv: {top_json} != {top_table}")


def audit_svg(run_dir: Path, findings: list[dict]) -> None:
    for svg in run_dir.rglob("*.svg"):
        text = svg.read_text(encoding="utf-8", errors="ignore")
        if "\u2212" in text:
            add_finding(findings, "error", str(svg), "SVG contains U+2212 minus sign.")
        if not svg.with_suffix(".png").exists():
            add_finding(findings, "warning", str(svg), "SVG figure does not have a PNG pair with the same basename.")
    for png in run_dir.rglob("*.png"):
        if not png.with_suffix(".svg").exists():
            add_finding(findings, "warning", str(png), "PNG figure does not have an SVG pair with the same basename.")


def audit_force_plots(run_dir: Path, findings: list[dict]) -> None:
    for path in run_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in CODE_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "shap.force_plot" in text or "shap.plots.force" in text:
            add_finding(findings, "error", str(path), "Force plots are disallowed; use beeswarm, bar, dependence, heatmap, interaction matrix, or manual waterfall.")


def audit_final_report(run_dir: Path, phase: str, findings: list[dict]) -> None:
    if phase != "final":
        return
    reports = [p for p in run_dir.rglob("*") if p.is_file() and p.name.lower() in {"report.md", "report.html"}]
    if not reports:
        return
    report_texts = {p: p.read_text(encoding="utf-8", errors="ignore") for p in reports}
    text = "\n".join(content.lower() for content in report_texts.values())
    if not any(term in text for term in ["causal limitation", "causal limitations", "\u56e0\u679c\u5c40\u9650", "\u56e0\u679c\u9650\u5236"]):
        add_finding(findings, "warning", "report", "Final report does not appear to include a causal limitation section.")
    if "force_plot" in text or "shap.plots.force" in text:
        add_finding(findings, "error", "report", "Final report references disallowed force plots.")
    config_path = first_existing(run_dir, "model_config.json")
    if config_path:
        config = load_json(config_path)
        selected = str(config.get("selected_model", "")).lower()
        target = str(config.get("target", "")).lower()
        if selected and selected not in text:
            add_finding(findings, "warning", "report", "Final report does not mention the selected model recorded in model_config.json.")
        if target and target not in text:
            add_finding(findings, "warning", "report", "Final report does not mention the target recorded in model_config.json.")
    for report, content in report_texts.items():
        for link in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", content):
            if re.match(r"^[a-z]+://", link) or link.startswith("#"):
                continue
            local = (report.parent / link).resolve()
            fallback = (run_dir / link).resolve()
            if not local.exists() and not fallback.exists():
                add_finding(findings, "error", str(report), f"Report image link not found: {link}")


def audit(run_dir: Path, phase: str) -> dict:
    findings = []
    audit_required_files(run_dir, phase, findings)
    audit_domain_context(run_dir, phase, findings)
    audit_feature_decisions(run_dir, phase, findings)
    audit_model_config(run_dir, phase, findings)
    audit_shap_consistency(run_dir, phase, findings)
    audit_svg(run_dir, findings)
    audit_force_plots(run_dir, findings)
    audit_final_report(run_dir, phase, findings)
    return {"run_dir": str(run_dir), "phase": phase, "status": "pass" if not any(f["severity"] == "error" for f in findings) else "fail", "finding_count": len(findings), "findings": findings}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate ML-SHAP output artifacts.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--phase", choices=sorted(PHASE_REQUIRED), default="final")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.run_dir, args.phase)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
