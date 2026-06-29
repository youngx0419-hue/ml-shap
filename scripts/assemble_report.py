"""Assemble a concise Markdown report and model card for an ML-SHAP run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return default or {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    if not rows:
        return ["Not available."]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble ML-SHAP report.md and model_card.md.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--title", default="ML-SHAP Analysis Report")
    args = parser.parse_args()

    run_dir = args.run_dir
    profile = load_json(run_dir / "data_profile.json")
    domain = load_json(run_dir / "domain_context.json")
    config = load_json(run_dir / "model_config.json")
    insights = load_json(run_dir / "auto_insights.json")
    evidence = load_json(run_dir / "evidence_bank.json", {"papers": []})
    benchmark = read_csv(run_dir / "model_benchmark.csv")
    features = read_csv(run_dir / "feature_decision_table.csv")
    shap_importance = read_csv(run_dir / "05_Results" / "Tables" / "global_shap_importance.csv")

    report = [
        f"# {args.title}",
        "",
        "## Data Summary",
        f"- Rows: {profile.get('n_rows', 'unknown')}",
        f"- Columns: {profile.get('n_columns', 'unknown')}",
        f"- Target: {config.get('target') or profile.get('target') or 'not confirmed'}",
        f"- Domain: {domain.get('domain', 'not classified')}",
        "",
        "## Feature Review",
        f"- Candidate features used by the model: {len(config.get('features', []))}",
        f"- Strata-only columns: {', '.join(config.get('strata_columns', [])) or 'none'}",
        f"- Features requiring final review: {sum(1 for row in features if not row.get('final_decision'))}",
        "",
        "## Split And Modeling",
        f"- Split strategy: {config.get('split', {}).get('strategy', 'not recorded')}",
        f"- Selected model: {config.get('selected_model', 'not recorded')}",
        f"- Preprocessing fit scope: {config.get('split', {}).get('preprocessing_fit_scope', 'not recorded')}",
        "",
        "## Benchmark",
        *markdown_table(benchmark, list(benchmark[0].keys()) if benchmark else ["model"]),
        "",
        "## SHAP Summary",
        f"- SHAP status: {config.get('shap', {}).get('status', 'not recorded')}",
        *markdown_table(shap_importance[:10], ["feature", "mean_abs_shap"]),
        "",
        "## Evidence Summary",
        f"- Literature search status: {domain.get('literature_search_status', 'not recorded')}",
        f"- Evidence records: {len(evidence.get('papers', []))}",
        "",
        "## Causal Limitation",
        "SHAP values describe model-attributed associations under the fitted data, split, and preprocessing pipeline. They should not be read as causal effects unless the study design supplies independent experimental or quasi-experimental identification.",
        "",
        "## Quality Notes",
        f"- Top SHAP features from auto insights: {', '.join(str(item.get('feature', '')) for item in insights.get('top_features', [])[:5]) or 'not available'}",
        "- Tables above are assembled from source CSV/JSON artifacts in the run directory.",
    ]

    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    model_card = [
        "# Model Card",
        "",
        f"- Target: {config.get('target', 'not recorded')}",
        f"- Task type: {config.get('task_type', 'not recorded')}",
        f"- Selected model: {config.get('selected_model', 'not recorded')}",
        f"- Split strategy: {config.get('split', {}).get('strategy', 'not recorded')}",
        f"- Training features: {', '.join(config.get('features', []))}",
        "- Intended use: exploratory structured-tabular prediction and SHAP-based model interpretation.",
        "- Limitations: no causal interpretation from SHAP alone; external validity depends on the dataset sampling frame and held-out validation design.",
    ]
    (run_dir / "model_card.md").write_text("\n".join(model_card) + "\n", encoding="utf-8")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
