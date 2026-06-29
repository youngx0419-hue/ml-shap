"""Run a minimal leakage-aware tabular modeling pass for ML-SHAP."""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from reproducibility_manifest import build as build_manifest


DROP_DECISIONS = {"drop", "strata-only", "target"}
KEEP_DRAFTS = {"keep", "keep-review", "transform"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_table(path: Path, sheet: str | int | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, encoding="utf-8-sig")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise SystemExit(f"Unsupported dataset format: {path.suffix}")


def task_type(profile: dict[str, Any], target: str) -> str:
    for item in profile.get("target_candidates", []):
        if item.get("name") == target:
            guess = item.get("task_type_guess", "unknown")
            if guess == "classification_candidate":
                return "classification"
            if "classification" in guess:
                return "classification"
            return "regression"
    return "regression"


def feature_decisions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def selected_features(rows: list[dict[str, str]], target: str) -> list[str]:
    features = []
    for row in rows:
        name = row.get("raw_feature", "")
        final = row.get("final_decision", "").strip().lower()
        draft = row.get("draft_decision", "").strip().lower()
        decision = final or draft
        if name and name != target and decision not in DROP_DECISIONS and (final or draft in KEEP_DRAFTS):
            features.append(name)
    return features


def strata_columns(rows: list[dict[str, str]], target: str) -> list[str]:
    columns = []
    for row in rows:
        name = row.get("raw_feature", "")
        final = row.get("final_decision", "").strip().lower()
        draft = row.get("draft_decision", "").strip().lower()
        role = row.get("role_guess", "").strip().lower()
        if name and name != target and (final == "strata-only" or (not final and draft == "strata-only") or "time" in role or "group" in role or "spatial" in role):
            columns.append(name)
    return columns


def first_role_column(rows: list[dict[str, str]], candidates: list[str], roles: set[str]) -> str | None:
    for row in rows:
        name = row.get("raw_feature", "")
        role = row.get("role_guess", "").strip().lower()
        if name in candidates and any(token in role for token in roles):
            return name
    return candidates[0] if candidates else None


def choose_split(
    data: pd.DataFrame,
    profile: dict[str, Any],
    decisions: list[dict[str, str]],
    task: str,
    target: str,
    features: list[str],
    strata: list[str],
    test_size: float,
    seed: int,
    requested: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict[str, Any]]:
    split_meta: dict[str, Any] = {"requested_strategy": requested, "test_size": test_size, "random_state": seed, "preprocessing_fit_scope": "train_only"}
    time_col = first_role_column(decisions, strata, {"time"})
    group_col = first_role_column(decisions, strata, {"group", "spatial"})

    if requested in {"auto", "time"} and time_col and profile.get("has_time_order", False):
        ordered = data.sort_values(time_col)
        test_n = max(1, int(math.ceil(len(ordered) * test_size)))
        if len(ordered) - test_n >= 2:
            train = ordered.iloc[:-test_n]
            test = ordered.iloc[-test_n:]
            split_meta.update({"strategy": "time_holdout", "time_column": time_col, "shuffle": False})
            return train[features], test[features], train[target], test[target], split_meta
        if requested == "time":
            raise SystemExit("Time holdout is not feasible: not enough rows after applying --test-size.")

    if requested in {"auto", "group"} and group_col and data[group_col].nunique(dropna=True) >= 3:
        groups = data[group_col].astype(str)
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(data[features], data[target], groups=groups))
        train = data.iloc[train_idx]
        test = data.iloc[test_idx]
        strategy = "spatial_group_shuffle_split" if "spatial" in group_col.lower() else "group_shuffle_split"
        split_meta.update({"strategy": strategy, "group_column": group_col, "shuffle": True})
        return train[features], test[features], train[target], test[target], split_meta
    if requested == "group":
        raise SystemExit("Group split is not feasible: no usable strata-only/group column with at least 3 groups.")

    stratify = None
    if requested in {"auto", "stratified"} and task == "classification" and data[target].nunique() > 1 and data[target].value_counts().min() >= 2:
        stratify = data[target]
        split_meta.update({"strategy": "stratified_random_holdout", "shuffle": True})
    else:
        if requested == "stratified":
            raise SystemExit("Stratified split is not feasible: target classes are too sparse or task is not classification.")
        split_meta.update({"strategy": "random_holdout", "shuffle": True})

    x_train, x_test, y_train, y_test = train_test_split(data[features], data[target], test_size=test_size, random_state=seed, shuffle=True, stratify=stratify)
    return x_train, x_test, y_train, y_test, split_meta


def make_preprocessor(df: pd.DataFrame, features: list[str]) -> ColumnTransformer:
    numeric = [col for col in features if pd.api.types.is_numeric_dtype(df[col])]
    categorical = [col for col in features if col not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric))
    if categorical:
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", encoder)]), categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def candidate_models(task: str, seed: int) -> dict[str, Any]:
    if task == "classification":
        models: dict[str, Any] = {
            "dummy": DummyClassifier(strategy="most_frequent"),
            "random_forest": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1, class_weight="balanced"),
        }
        try:
            from xgboost import XGBClassifier

            models["xgboost"] = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=seed, n_jobs=-1)
        except Exception:
            pass
        return models
    models = {
        "dummy": DummyRegressor(strategy="median"),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1),
    }
    try:
        from xgboost import XGBRegressor

        models["xgboost"] = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror", random_state=seed, n_jobs=-1)
    except Exception:
        pass
    return models


def metrics(task: str, y_true: pd.Series, pred: np.ndarray, proba: np.ndarray | None = None) -> dict[str, float | None]:
    if task == "classification":
        out: dict[str, float | None] = {
            "accuracy": float(accuracy_score(y_true, pred)),
            "f1_weighted": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
            "roc_auc": None,
        }
        if proba is not None:
            try:
                if proba.ndim == 2 and proba.shape[1] == 2:
                    out["roc_auc"] = float(roc_auc_score(y_true, proba[:, 1]))
                elif proba.ndim == 2 and proba.shape[1] > 2:
                    out["roc_auc"] = float(roc_auc_score(y_true, proba, multi_class="ovr"))
            except Exception:
                out["roc_auc"] = None
        return out
    rmse = math.sqrt(mean_squared_error(y_true, pred))
    return {"r2": float(r2_score(y_true, pred)), "mae": float(mean_absolute_error(y_true, pred)), "rmse": float(rmse)}


def score_key(task: str, row: dict[str, Any]) -> float:
    if task == "classification":
        return float(row.get("f1_weighted") or row.get("accuracy") or 0.0)
    return float(row.get("r2") or -1e9)


def transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return [str(x) for x in preprocessor.get_feature_names_out()]
    except Exception:
        return []


def save_shap_outputs(run_dir: Path, best_name: str, pipeline: Pipeline, x_test: pd.DataFrame, seed: int) -> dict[str, Any]:
    out: dict[str, Any] = {"status": "not_available", "reason": "", "same_model_for_shap": True}
    try:
        import matplotlib.pyplot as plt
        import shap

        from xai_style_helpers import save_chart, setup_plot_style
    except Exception as exc:
        out["reason"] = f"SHAP or plotting dependency unavailable: {exc}"
        return out

    try:
        preprocessor = pipeline.named_steps["preprocess"]
        model = pipeline.named_steps["model"]
        sample = x_test.sample(n=min(len(x_test), 300), random_state=seed) if len(x_test) > 300 else x_test
        x_matrix = preprocessor.transform(sample)
        names = transformed_feature_names(preprocessor)
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(x_matrix)
        shap_array = values[1] if isinstance(values, list) and len(values) > 1 else values
        shap_array = np.asarray(shap_array)
        if shap_array.ndim == 3:
            shap_array = shap_array[:, :, 0]

        shap_dir = run_dir / "04_SHAP"
        table_dir = run_dir / "05_Results" / "Tables"
        chart_dir = run_dir / "05_Results" / "Charts"
        shap_dir.mkdir(parents=True, exist_ok=True)
        table_dir.mkdir(parents=True, exist_ok=True)
        chart_dir.mkdir(parents=True, exist_ok=True)
        np.save(shap_dir / "shap_values_sample.npy", shap_array)

        importance = pd.DataFrame({"feature": names or [f"feature_{i}" for i in range(shap_array.shape[1])], "mean_abs_shap": np.abs(shap_array).mean(axis=0)})
        importance = importance.sort_values("mean_abs_shap", ascending=False)
        importance.to_csv(table_dir / "global_shap_importance.csv", index=False, encoding="utf-8-sig")
        write_json(run_dir / "auto_insights.json", {"top_features": importance.head(10).to_dict(orient="records"), "model": best_name})

        setup_plot_style()
        plt.figure(figsize=(7, 5))
        shap.summary_plot(shap_array, x_matrix, feature_names=names or None, show=False, plot_type="bar")
        save_chart(plt.gcf(), chart_dir / "shap_bar")
        plt.close()

        plt.figure(figsize=(7, 5))
        shap.summary_plot(shap_array, x_matrix, feature_names=names or None, show=False)
        save_chart(plt.gcf(), chart_dir / "shap_beeswarm")
        plt.close()
        out["status"] = "created"
        out["sample_size"] = int(len(sample))
        out["explainer"] = "TreeExplainer"
    except Exception as exc:
        out["status"] = "failed"
        out["reason"] = str(exc)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a minimal ML-SHAP modeling pass.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--target")
    parser.add_argument("--sheet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skill-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--split-strategy", choices=["auto", "random", "stratified", "group", "time"], default="auto")
    parser.add_argument("--skip-shap", action="store_true")
    args = parser.parse_args()

    profile = load_json(args.run_dir / "data_profile.json")
    target = args.target or profile.get("target")
    if not target:
        candidates = profile.get("target_candidates", [])
        if len(candidates) == 1:
            target = candidates[0]["name"]
    if not target:
        raise SystemExit("Target is not confirmed. Pass --target or rerun bootstrap with --target.")

    decisions = feature_decisions(args.run_dir / "feature_decision_table.csv")
    features = selected_features(decisions, target)
    strata = strata_columns(decisions, target)
    if not features:
        raise SystemExit("No usable features selected in feature_decision_table.csv.")

    df = load_table(args.dataset, args.sheet)
    split_columns = [col for col in strata if col in df.columns and col not in features and col != target]
    missing = [col for col in [target, *features, *split_columns] if col not in df.columns]
    if missing:
        raise SystemExit("Columns not found in dataset: " + ", ".join(missing))

    data = df[[target, *features, *split_columns]].dropna(subset=[target]).copy()
    task = task_type(profile, target)
    x_train, x_test, y_train, y_test, split_meta = choose_split(data, profile, decisions, task, target, features, split_columns, args.test_size, args.seed, args.split_strategy)

    rows = []
    fitted: dict[str, Pipeline] = {}
    for name, model in candidate_models(task, args.seed).items():
        pipe = Pipeline([("preprocess", make_preprocessor(data, features)), ("model", model)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(x_train, y_train)
        pred = pipe.predict(x_test)
        proba = pipe.predict_proba(x_test) if task == "classification" and hasattr(pipe, "predict_proba") else None
        row = {"model": name, **metrics(task, y_test, pred, proba)}
        rows.append(row)
        fitted[name] = pipe

    benchmark = pd.DataFrame(rows).sort_values(by="model")
    best_row = max(rows, key=lambda item: score_key(task, item))
    best_name = str(best_row["model"])
    best_pipe = fitted[best_name]

    (args.run_dir / "03_Models").mkdir(parents=True, exist_ok=True)
    (args.run_dir / "05_Results" / "Tables").mkdir(parents=True, exist_ok=True)
    benchmark.to_csv(args.run_dir / "model_benchmark.csv", index=False, encoding="utf-8-sig")
    joblib.dump(best_pipe, args.run_dir / "best_model.pkl")
    joblib.dump(best_pipe.named_steps["preprocess"], args.run_dir / "preprocessing_pipeline.pkl")

    shap_meta = {"status": "skipped", "reason": "--skip-shap was used", "same_model_for_shap": True}
    if not args.skip_shap and best_name != "dummy":
        shap_meta = save_shap_outputs(args.run_dir, best_name, best_pipe, x_test, args.seed)

    config = {
        "target": target,
        "task_type": task,
        "features": features,
        "excluded_features": [row.get("raw_feature") for row in decisions if row.get("raw_feature") not in features and row.get("raw_feature") != target],
        "strata_columns": split_columns,
        "split": split_meta,
        "models_tested": list(fitted),
        "selected_model": best_name,
        "selected_model_artifact": "best_model.pkl",
        "preprocessing_artifact": "preprocessing_pipeline.pkl",
        "metrics": best_row,
        "shap": shap_meta,
    }
    write_json(args.run_dir / "model_config.json", config)
    write_json(args.run_dir / "05_Results" / "Tables" / "evaluation_metrics.json", {"benchmark": rows, "selected_model": best_name})
    write_json(args.run_dir / "reproducibility_manifest.json", build_manifest(args.run_dir, args.dataset, args.skill_dir, args.seed))
    print(f"Selected model: {best_name}")
    print(f"Wrote {args.run_dir / 'model_config.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
