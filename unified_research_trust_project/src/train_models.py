import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from config import (
    CATEGORICAL_FEATURES,
    DATA_DIR,
    FEATURE_COLUMNS,
    MODEL_DIR,
    NUMERIC_FEATURES,
    OUTPUT_DIR,
    RANDOM_SEED,
    TARGET,
)


def evaluate_model(name: str, model, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    prob = model.predict_proba(x_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "model": name,
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "auc": round(float(roc_auc_score(y_test, prob)), 4),
        "brier_score": round(float(brier_score_loss(y_test, prob)), 4),
        "log_loss": round(float(log_loss(y_test, prob)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_DIR / "applications.csv")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    try:
        from interpret.glassbox import ExplainableBoostingClassifier
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: interpret. Install project requirements before training: "
            "pip install -r requirements.txt"
        ) from exc

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    x = df[FEATURE_COLUMNS]
    y = df[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    ebm = ExplainableBoostingClassifier(
        random_state=args.seed,
        interactions=8,
        max_bins=128,
        learning_rate=0.03,
    )
    ebm.fit(x_train, y_train)

    blackbox_preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    linear_preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    logistic = Pipeline(
        steps=[
            ("preprocess", linear_preprocess),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=args.seed,
                ),
            ),
        ]
    )
    logistic.fit(x_train, y_train)

    blackbox = Pipeline(
        steps=[
            ("preprocess", blackbox_preprocess),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=5,
                    random_state=args.seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    blackbox.fit(x_train, y_train)

    metrics = {
        "data_rows": int(len(df)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "positive_rate": round(float(y.mean()), 4),
        "models": [
            evaluate_model("Logistic Regression transparent linear baseline", logistic, x_test, y_test),
            evaluate_model("Explainable Boosting Machine", ebm, x_test, y_test),
            evaluate_model("Random Forest black-box baseline", blackbox, x_test, y_test),
        ],
        "interpretation_note": (
            "Logistic Regression is used as a transparent linear baseline, Random Forest as a "
            "black-box predictive baseline, and EBM as the main model because it provides "
            "nonlinear global term importance, local additive contributions, and feature "
            "response curves for trust calibration."
        ),
    }

    joblib.dump(ebm, MODEL_DIR / "ebm_model.joblib")
    joblib.dump(logistic, MODEL_DIR / "logistic_regression_baseline.joblib")
    joblib.dump(blackbox, MODEL_DIR / "random_forest_baseline.joblib")
    x_test.assign(funded=y_test.values).to_csv(OUTPUT_DIR / "test_set.csv", index=False, encoding="utf-8-sig")
    with (OUTPUT_DIR / "model_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
