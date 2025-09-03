import os, json, pathlib, subprocess, sys

def test_training_produces_artifacts():
    # run training
    cmd = [sys.executable, "src/train.py", "--data", "data/raw/sample_telco_churn.csv"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    # check files
    assert pathlib.Path("models/model.pkl").exists()
    assert pathlib.Path("artifacts/metrics.json").exists()

    # basic metrics sanity
    with open("artifacts/metrics.json") as f:
        m = json.load(f)
    assert 0.5 <= m["roc_auc"] <= 1.0
