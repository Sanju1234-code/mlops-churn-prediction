.PHONY: setup train test api lint

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt || ( .venv\Scripts\activate && pip install -r requirements.txt )

train:
	python src/train.py --data data/raw/sample_telco_churn.csv

test:
	pytest -q

api:
	uvicorn api.app:app --reload

lint:
	flake8 src api tests
