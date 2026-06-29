"""Create reproducibility_manifest.json for an ML-SHAP run."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def versions(names: list[str]) -> dict[str, str | None]:
    import importlib.metadata as md

    out = {}
    for name in names:
        try:
            out[name] = md.version(name)
        except Exception:
            out[name] = None
    return out


def git_info(path: Path) -> dict[str, Any]:
    try:
        root = subprocess.check_output(["git", "-C", str(path), "rev-parse", "--show-toplevel"], text=True, stderr=subprocess.DEVNULL).strip()
        commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL)
        return {"root": root, "commit": commit, "dirty": bool(dirty.strip())}
    except Exception:
        return {"root": None, "commit": None, "dirty": None}


def build(run_dir: Path, dataset: Path | None, skill_dir: Path | None, seed: int) -> dict[str, Any]:
    files = []
    for p in sorted(run_dir.rglob("*")):
        if p.is_file():
            files.append({"path": str(p.relative_to(run_dir)), "bytes": p.stat().st_size, "sha256": sha256_file(p)})
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {"python": sys.version, "platform": platform.platform(), "package_versions": versions(["numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "catboost", "shap", "matplotlib", "seaborn"])},
        "random_seed": seed,
        "dataset": {"path": str(dataset) if dataset else None, "sha256": sha256_file(dataset) if dataset and dataset.exists() else None},
        "skill": {"path": str(skill_dir) if skill_dir else None, "skill_md_sha256": sha256_file(skill_dir / "SKILL.md") if skill_dir and (skill_dir / "SKILL.md").exists() else None},
        "git": git_info(run_dir),
        "run_files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create reproducibility_manifest.json.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--skill-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = args.output or args.run_dir / "reproducibility_manifest.json"
    out.write_text(json.dumps(build(args.run_dir, args.dataset, args.skill_dir, args.seed), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
