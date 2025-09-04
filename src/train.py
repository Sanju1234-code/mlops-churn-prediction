import argparse
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import ARTIFACTS_DIR, MODELS_DIR

TARGET = "Churn"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number", "bool", "float64", "int64"]).columns.tolist()
    preproc = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ]
    )
    return preproc


def evaluate(pipe: Pipeline, X_test, y_test):
    proba = pipe.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
    }


def main(data_path: str):
    df = load_data(data_path)
    y = df[TARGET].map({"Yes": 1, "No": 0}).values
    X = df.drop(columns=[TARGET, "customerID"], errors="ignore")

    preproc = build_preprocessor(X)

    # ---- Two models: GradientBoosting and XGBoost ----
    models = {}

    # 1) Gradient Boosting (baseline)
    gbm = GradientBoostingClassifier(random_state=7)
    models["gradient_boosting"] = Pipeline([("prep", preproc), ("clf", gbm)])

    # 2) XGBoost (if available)
    try:
        from xgboost import XGBClassifier

        pos_weight = float((y == 0).sum()) / max(1.0, float((y == 1).sum()))
        xgb = XGBClassifier(
            n_estimators=600,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            random_state=7,
            tree_method="hist",
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
        )
        models["xgboost"] = Pipeline([("prep", preproc), ("clf", xgb)])
    except Exception as e:
        print("Warning: XGBoost not available, skipping. Error:", e)

    # One split for fair comparison
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    results = {}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    best_name = None
    best_score = -1.0

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test)
        results[name] = metrics

        # Save each model
        joblib.dump(pipe, MODELS_DIR / f"{name}.pkl")

        if metrics["roc_auc"] > best_score:
            best_name = name
            best_score = metrics["roc_auc"]

    # Write metrics comparison
    with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save a canonical "best" model for the API
    if best_name is not None:
        best_model_path = MODELS_DIR / f"{best_name}.pkl"
        joblib.dump(joblib.load(best_model_path), MODELS_DIR / "best_model.pkl")

    print("Saved models in:", MODELS_DIR.as_posix())
    print("Metrics:", json.dumps(results, indent=2))
    print("Best model:", best_name, "ROC-AUC:", best_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/raw/sample_telco_churn.csv")
    args = parser.parse_args()
    main(args.data)
