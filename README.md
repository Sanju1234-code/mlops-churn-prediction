![CI](https://github.com/Sanju1234-code/mlops-churn-prediction/actions/workflows/ci.yml/badge.svg)
# MLOps Churn Prediction (End‑to‑End)

An end‑to‑end **Customer Churn Prediction** project designed to showcase
**production‑minded ML** for a ~3‑year engineer. It includes clean repo
structure, data processing, training pipeline, tests, CI, Dockerized API, and
deploy‑ready instructions.

> Tech: Python, scikit‑learn, pandas, FastAPI, pytest, GitHub Actions, Docker

## Features
- Reproducible training pipeline (`src/train.py`) that:
  - loads CSV from `data/raw/`, cleans/encodes features, trains a model
  - logs metrics, saves artifacts to `models/`
- Simple FastAPI service (`api/app.py`) for real‑time predictions
- Unit tests (`tests/`) and lint checks (flake8) wired into GitHub Actions
- Dockerfile for serving; Makefile for common tasks
- Model/experiment metadata saved to `artifacts/metrics.json`

## Quickstart (Local)
```bash
# 1) Create & activate a venv (recommended)
python -m venv .venv && source .venv/bin/activate  # (Windows) .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Train
python src/train.py --data data/raw/sample_telco_churn.csv

# 4) Serve API
uvicorn api.app:app --reload
# Then POST to http://127.0.0.1:8000/predict (see docs at /docs)
```

## Docker (API)
```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## Makefile shortcuts
```bash
make setup        # install deps
make train        # run training
make test         # run unit tests
make api          # launch FastAPI (uvicorn)
```

## Repo layout
```
.
├── api/                 # FastAPI app
├── data/                # raw/processed data (small sample included)
├── notebooks/           # exploratory notebooks
├── src/                 # pipeline code
├── tests/               # unit tests
├── .github/workflows/   # CI
├── Dockerfile
├── Makefile
├── requirements.txt
└── README.md
```

## Next steps 
- Swap sample CSV with a public dataset & add a Data Card.
- Add MLflow tracking & experiment comparison in `notebooks/`.
- Create an Airflow or GitHub Actions workflow to retrain weekly.
- Push a Docker image to AWS ECR or Azure ACR and deploy to ECS/AKS.
- Add a small Streamlit front‑end for demo.

---
*Generated on 2025-09-03 14:21 UTC*
