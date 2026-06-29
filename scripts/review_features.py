"""Review feature decisions before ML-SHAP modeling."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


LEAKAGE_TERMS = {"after", "post", "outcome", "label", "result", "score", "risk", "diagnosis", "death", "survival", "未来", "结果", "标签", "结局", "评分"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_feature_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_feature_table(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def column_index(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {col.get("name", ""): col for col in profile.get("columns", [])}


def add_finding(findings: list[dict[str, Any]], severity: str, feature: str, issue: str, recommendation: str) -> None:
    findings.append({"severity": severity, "feature": feature, "issue": issue, "recommendation": recommendation})


def review(profile: dict[str, Any], rows: list[dict[str, str]], target: str | None) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    columns = column_index(profile)
    target_name = target or profile.get("target")

    for row in rows:
        name = row.get("raw_feature", "")
        role = row.get("role_guess", "") or columns.get(name, {}).get("role_guess", "")
        decision = (row.get("final_decision") or row.get("draft_decision") or "").strip().lower()
        lower = name.lower()
        if (name == target_name or role == "target") and decision != "target":
            add_finding(findings, "error", name, "target column appears in feature table", "Set final_decision=target and exclude it from modeling.")
        if role == "id_candidate":
            add_finding(findings, "warning", name, "identifier-like column", "Set final_decision=drop unless the identifier has an approved causal meaning.")
        if role in {"time_candidate", "group_candidate", "spatial_or_group_candidate"}:
            add_finding(findings, "warning", name, "time/group/spatial column", "Use as strata-only for splitting or SHAP stratification unless explicitly approved as a predictor.")
        if target_name and target_name.lower() in lower and name != target_name:
            add_finding(findings, "error", name, "target-name leakage risk", "Review formula leakage; drop unless available before prediction and not derived from target.")
        if any(term in lower for term in LEAKAGE_TERMS):
            add_finding(findings, "warning", name, "post-outcome or label-like name", "Confirm prediction-time availability before keeping this feature.")

    for pair in profile.get("correlation_summary", {}).get("high_pairs", []):
        left = pair.get("left") or pair.get("feature_a")
        right = pair.get("right") or pair.get("feature_b")
        if left and right:
            add_finding(findings, "info", f"{left} | {right}", "high feature correlation", "Interpret SHAP jointly or consider grouped/ALE support.")
    return findings


def apply_safe_defaults(rows: list[dict[str, str]]) -> None:
    for row in rows:
        draft = row.get("draft_decision", "").strip()
        role = row.get("role_guess", "").strip()
        if not row.get("final_decision"):
            row["final_decision"] = draft
        if role in {"id_candidate", "time_candidate", "group_candidate", "spatial_or_group_candidate", "target"}:
            row["leakage_review"] = "reviewed"
        elif row.get("leakage_review") == "needs_review":
            row["leakage_review"] = "reviewed"


def main() -> int:
    parser = argparse.ArgumentParser(description="Review ML-SHAP feature decisions for leakage and split roles.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--target")
    parser.add_argument("--apply-safe-defaults", action="store_true")
    args = parser.parse_args()

    profile = load_json(args.run_dir / "data_profile.json")
    table = args.run_dir / "feature_decision_table.csv"
    fieldnames, rows = read_feature_table(table)
    findings = review(profile, rows, args.target)

    if args.apply_safe_defaults:
        apply_safe_defaults(rows)
        write_feature_table(table, fieldnames, rows)

    write_json(args.run_dir / "feature_review.json", {"finding_count": len(findings), "findings": findings})
    with (args.run_dir / "feature_review_findings.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["severity", "feature", "issue", "recommendation"])
        writer.writeheader()
        writer.writerows(findings)
    print(f"Wrote {args.run_dir / 'feature_review.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
