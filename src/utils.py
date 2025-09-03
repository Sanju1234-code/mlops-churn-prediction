from pathlib import Path
ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = Path("models")
for d in [ARTIFACTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
