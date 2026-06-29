"""Create data_profile.json and dataset_datasheet.md for ML-SHAP."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ID_RE = re.compile(r"(^id$|_id$|\u7f16\u53f7|\u5e8f\u53f7|\u6837\u672c\u53f7|sample\s*id|index)", re.I)
TIME_RE = re.compile(r"(date|time|year|month|day|\u65e5\u671f|\u65f6\u95f4|\u5e74\u4efd|\u6708\u4efd)", re.I)
SPATIAL_RE = re.compile(r"(lat|lon|lng|longitude|latitude|x_coord|y_coord|\u7ecf\u5ea6|\u7eac\u5ea6|\u5750\u6807|\u57ce\u5e02|\u533a\u57df|\u7ad9\u70b9|\u6d41\u57df)", re.I)
GROUP_RE = re.compile(r"(group|site|station|batch|subject|patient|city|region|\u7ec4|\u6279\u6b21|\u7ad9\u70b9|\u57ce\u5e02|\u533a\u57df|\u6d41\u57df)", re.I)
TARGET_RE = re.compile(r"(target|label|outcome|response|yield|\u4ea7\u91cf|\u53bb\u9664\u7387|\u6d53\u5ea6|\u98ce\u9669|\u7b49\u7ea7|score|\u5f97\u5206)", re.I)


def load_table(path: Path, sheet: str | int | None = None) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path, encoding="utf-8-sig")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise SystemExit(f"Unsupported dataset format: {path.suffix}")


def safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) or np.isinf(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def role(name: str, series: pd.Series, target: str | None) -> str:
    non_null = int(series.notna().sum())
    unique = int(series.nunique(dropna=True))
    if target and name == target:
        return "target"
    if ID_RE.search(name) or (non_null > 20 and unique == non_null):
        return "id_candidate"
    if TIME_RE.search(name):
        return "time_candidate"
    if SPATIAL_RE.search(name):
        return "spatial_or_group_candidate"
    if GROUP_RE.search(name):
        return "group_candidate"
    return "numeric_feature_candidate" if pd.api.types.is_numeric_dtype(series) else "categorical_feature_candidate"


def task_guess(series: pd.Series) -> str:
    clean = series.dropna()
    if clean.empty:
        return "unknown"
    n_unique = int(clean.nunique())
    if pd.api.types.is_numeric_dtype(clean):
        return "classification_candidate" if n_unique <= min(20, max(2, int(len(clean) * 0.05))) else "regression"
    return "binary_classification" if n_unique == 2 else "multiclass_classification"


def column_profile(df: pd.DataFrame, target: str | None) -> list[dict[str, Any]]:
    out = []
    n = max(len(df), 1)
    for col in df.columns:
        s = df[col]
        item = {
            "name": str(col),
            "dtype": str(s.dtype),
            "role_guess": role(str(col), s, target),
            "missing_count": int(s.isna().sum()),
            "missing_rate": round(float(s.isna().sum() / n), 6),
            "unique_count": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s):
            item.update({"mean": safe(s.mean()), "std": safe(s.std()), "min": safe(s.min()), "median": safe(s.median()), "max": safe(s.max())})
        else:
            item["top_values"] = [{"value": str(k), "count": int(v)} for k, v in s.astype("string").value_counts(dropna=True).head(8).items()]
        out.append(item)
    return out


def correlation_summary(df: pd.DataFrame, target: str | None) -> dict[str, Any]:
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return {"numeric_feature_count": int(num.shape[1]), "max_abs_correlation": None, "high_pairs": []}
    corr = num.corr(numeric_only=True).abs()
    np.fill_diagonal(corr.values, np.nan)
    max_corr = None if np.isnan(corr.values).all() else float(np.nanmax(corr.values))
    pairs = []
    for i, left in enumerate(corr.columns):
        for right in corr.columns[i + 1 :]:
            val = corr.loc[left, right]
            if pd.notna(val) and val >= 0.85:
                pairs.append({"left": str(left), "right": str(right), "abs_corr": round(float(val), 6)})
    return {"numeric_feature_count": int(num.shape[1]), "max_abs_correlation": round(max_corr, 6) if max_corr is not None else None, "high_pairs": sorted(pairs, key=lambda x: x["abs_corr"], reverse=True)[:30]}


def leakage_flags(df: pd.DataFrame, target: str | None) -> list[dict[str, str]]:
    flags = []
    target_low = str(target).lower() if target else ""
    for col in df.columns:
        low = str(col).lower()
        if target and col == target:
            continue
        if target_low and target_low in low:
            flags.append({"column": str(col), "risk": "name_contains_target", "action": "review before modeling"})
        if any(k in low for k in ["after", "post", "result", "score", "grade", "label", "\u9884\u6d4b", "\u7ed3\u679c", "\u7b49\u7ea7"]):
            flags.append({"column": str(col), "risk": "possible_post_outcome_or_label_proxy", "action": "review prediction-time availability"})
    return flags


def profile(path: Path, target: str | None, sheet: str | int | None) -> dict[str, Any]:
    df = load_table(path, sheet)
    if target and target not in df.columns:
        raise SystemExit(f"Target not found: {target}")
    cols = column_profile(df, target)
    candidates = [target] if target else [c for c in df.columns if TARGET_RE.search(str(c))] or list(df.columns[-min(5, len(df.columns)) :])
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(path),
        "sheet": sheet,
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "target": target,
        "target_candidates": [{"name": str(c), "task_type_guess": task_guess(df[c]), "missing_rate": round(float(df[c].isna().mean()), 6), "unique_count": int(df[c].nunique(dropna=True))} for c in candidates if c in df.columns],
        "id_columns": [c["name"] for c in cols if c["role_guess"] == "id_candidate"],
        "time_columns": [c["name"] for c in cols if c["role_guess"] == "time_candidate"],
        "group_or_spatial_columns": [c["name"] for c in cols if c["role_guess"] in {"group_candidate", "spatial_or_group_candidate"}],
        "has_time_order": any(c["role_guess"] == "time_candidate" for c in cols),
        "has_group_structure": any(c["role_guess"] in {"group_candidate", "spatial_or_group_candidate"} for c in cols),
        "has_spatial_structure": any(c["role_guess"] == "spatial_or_group_candidate" for c in cols),
        "columns": cols,
        "correlation_summary": correlation_summary(df, target),
        "leakage_flags": leakage_flags(df, target),
    }


def datasheet(payload: dict[str, Any]) -> str:
    lines = [
        "# Dataset Datasheet",
        "",
        f"- Source file: `{payload['source_file']}`",
        f"- Rows: {payload['n_rows']}",
        f"- Columns: {payload['n_columns']}",
        f"- Duplicate rows: {payload['duplicate_rows']}",
        f"- Target candidates: {', '.join(c['name'] for c in payload['target_candidates']) or 'not inferred'}",
        f"- Time columns: {', '.join(payload['time_columns']) or 'none detected'}",
        f"- Group/spatial columns: {', '.join(payload['group_or_spatial_columns']) or 'none detected'}",
        "",
        "## Required Human Review",
        "- Confirm target definition, unit, and prediction timing.",
        "- Confirm row independence or grouping by site, batch, time, subject, station, city, or experiment.",
        "- Confirm whether any feature is measured after the outcome or derived from the target.",
        "",
        "## Leakage Flags",
    ]
    lines.extend([f"- `{x['column']}`: {x['risk']} ({x['action']})" for x in payload["leakage_flags"]] or ["- No obvious name-based leakage flags detected."])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile a tabular dataset for ML-SHAP.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--target")
    parser.add_argument("--sheet")
    parser.add_argument("--outdir", type=Path, default=Path("."))
    args = parser.parse_args()
    payload = profile(args.dataset, args.target, args.sheet)
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "data_profile.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=safe) + "\n", encoding="utf-8")
    (args.outdir / "dataset_datasheet.md").write_text(datasheet(payload), encoding="utf-8")
    print(f"Wrote {args.outdir / 'data_profile.json'}")
    print(f"Wrote {args.outdir / 'dataset_datasheet.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
