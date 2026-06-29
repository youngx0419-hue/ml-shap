# Examples

This folder contains small synthetic inputs for checking that the ML-SHAP skill can run end to end.

Run from the repository root:

```bash
python scripts/bootstrap_run.py examples/water_quality_sample.csv --run-dir runs/demo --target target --tier standard --user-text "water quality regression with temporal site groups"
python scripts/review_features.py --run-dir runs/demo --target target --apply-safe-defaults
python scripts/run_modeling.py examples/water_quality_sample.csv --run-dir runs/demo --target target --split-strategy auto
python scripts/update_evidence_bank.py --run-dir runs/demo --source examples/evidence_sample.csv
python scripts/assemble_report.py --run-dir runs/demo --title "Demo ML-SHAP Report"
python scripts/validate_outputs.py runs/demo --phase final
```

The sample data are synthetic and are only meant for smoke testing the workflow.
