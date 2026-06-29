"""Import literature evidence into an ML-SHAP run directory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "evidence_type",
    "title",
    "year",
    "venue",
    "doi_or_url",
    "abstract_or_key_claim",
    "why_close",
    "similarity_score_0_100",
    "method_pattern",
    "visual_pattern",
    "transferable_to_this_dataset",
    "caution",
]


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize(row: dict[str, Any]) -> dict[str, Any]:
    item = {field: str(row.get(field, "") or "") for field in FIELDS}
    if not item["doi_or_url"]:
        item["doi_or_url"] = str(row.get("url") or row.get("doi") or "")
    if not item["abstract_or_key_claim"]:
        item["abstract_or_key_claim"] = str(row.get("key_claim") or row.get("claim") or "")
    if not item["evidence_type"]:
        item["evidence_type"] = "nearest_domain"
    return item


def read_source(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = load_json(path, [])
        rows = payload.get("papers", payload) if isinstance(payload, dict) else payload
        return [normalize(row) for row in rows]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [normalize(row) for row in csv.DictReader(handle)]


def write_template(path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow({field: "" for field in FIELDS})


def main() -> int:
    parser = argparse.ArgumentParser(description="Update evidence_bank.json from CSV or JSON literature notes.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--mark-search-status", choices=["completed", "not_available", "pending"], default="completed")
    args = parser.parse_args()

    if args.template:
        write_template(args.template)
        print(f"Wrote template: {args.template}")
        return 0
    if not args.source:
        raise SystemExit("Pass --source or --template.")

    papers = read_source(args.source)
    bank_path = args.run_dir / "evidence_bank.json"
    bank = load_json(bank_path, {"papers": [], "method_influence_table": [], "visual_influence_table": [], "claims_supported": [], "claims_not_supported": []})
    bank["papers"] = papers
    bank["method_influence_table"] = [{"title": p["title"], "method_pattern": p["method_pattern"], "transferable": p["transferable_to_this_dataset"], "caution": p["caution"]} for p in papers if p.get("method_pattern")]
    bank["visual_influence_table"] = [{"title": p["title"], "visual_pattern": p["visual_pattern"]} for p in papers if p.get("visual_pattern")]
    write_json(bank_path, bank)

    ctx_path = args.run_dir / "domain_context.json"
    ctx = load_json(ctx_path, {})
    nearest = [p for p in papers if p.get("evidence_type") == "nearest_domain"][:3]
    ctx["nearest_recent_papers"] = nearest
    ctx["literature_search_status"] = args.mark_search_status
    write_json(ctx_path, ctx)

    with (args.run_dir / "citation_support_table.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["claim_area", "title", "year", "doi_or_url", "support", "caution"])
        writer.writeheader()
        for paper in papers:
            writer.writerow({"claim_area": paper["evidence_type"], "title": paper["title"], "year": paper["year"], "doi_or_url": paper["doi_or_url"], "support": paper["abstract_or_key_claim"], "caution": paper["caution"]})
    print(f"Imported {len(papers)} evidence records.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
