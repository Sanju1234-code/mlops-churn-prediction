import json, pathlib, subprocess, sys

def test_training_produces_artifacts():
    # Run training as a module so imports work on CI
    cmd = [sys.executable, "-m", "src.train", "--data", "data/raw/sample_telco_churn.csv"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"train failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    models_dir = pathlib.Path("models")
    artifacts_dir = pathlib.Path("artifacts")

    # Artifacts exist
    assert (artifacts_dir / "metrics.json").exists()

    # Accept either legacy single-model artifact or new best-model artifact
    assert (models_dir / "best_model.pkl").exists() or (models_dir / "model.pkl").exists()

    # Handle both metrics formats:
    #  - legacy: {"roc_auc": ..., "accuracy": ..., "f1": ...}
    #  - new two-model: {"gradient_boosting": {...}, "xgboost": {...}}
    with open(artifacts_dir / "metrics.json") as f:
        metrics = json.load(f)

    def ok(m):
        return isinstance(m, dict) and 0.0 <= float(m.get("roc_auc", 0.0)) <= 1.0

    if "roc_auc" in metrics:
        # legacy single-model format
        assert ok(metrics)
    else:
        # multi-model format
        assert any(k in metrics for k in ("gradient_boosting", "xgboost"))
        for m in metrics.values():
            assert ok(m)
