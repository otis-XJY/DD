import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_FILES = [
    "requirements.txt",
    "README.md",
    "docs/技术路线与复现实验说明.md",
    "src/config.py",
    "src/generate_data.py",
    "src/download_public_reference.py",
    "src/train_models.py",
    "src/explain_and_export.py",
    "app/streamlit_app.py",
    "data/applications.csv",
    "data/openalex_reference_sample.csv",
    "models/ebm_model.joblib",
    "models/logistic_regression_baseline.joblib",
    "models/random_forest_baseline.joblib",
    "outputs/model_metrics.json",
    "outputs/global_importance.json",
    "outputs/group_importance.json",
    "outputs/local_explanations.json",
    "outputs/response_curves.json",
    "outputs/case_routing.csv",
    "outputs/public_feature_mapping.csv",
    "outputs/public_data_summary.json",
]


def assert_file_exists(rel_path: str) -> None:
    path = ROOT / rel_path
    if not path.exists():
        raise AssertionError(f"Missing required file: {rel_path}")
    if path.is_file() and path.stat().st_size == 0:
        raise AssertionError(f"Empty required file: {rel_path}")


def load_json(rel_path: str):
    with (ROOT / rel_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    for rel_path in REQUIRED_FILES:
        assert_file_exists(rel_path)

    applications = pd.read_csv(ROOT / "data/applications.csv")
    if len(applications) < 1000:
        raise AssertionError("applications.csv should contain at least 1000 rows")
    if "funded" not in applications.columns:
        raise AssertionError("applications.csv is missing funded target")
    positive_rate = applications["funded"].mean()
    if not 0.15 <= positive_rate <= 0.45:
        raise AssertionError(f"Funding rate {positive_rate:.3f} is outside expected range [0.15, 0.45]")

    metrics = load_json("outputs/model_metrics.json")
    model_names = {row["model"] for row in metrics["models"]}
    expected_models = {
        "Logistic Regression transparent linear baseline",
        "Explainable Boosting Machine",
        "Random Forest black-box baseline",
    }
    missing_models = expected_models - model_names
    if missing_models:
        raise AssertionError(f"Missing model metrics: {sorted(missing_models)}")

    routing = pd.read_csv(ROOT / "outputs/case_routing.csv")
    expected_routing = {"fast_track", "regular_review", "expert_review"}
    if set(routing["routing"].unique()) != expected_routing:
        raise AssertionError("case_routing.csv should contain all three routing categories")

    mapping = pd.read_csv(ROOT / "outputs/public_feature_mapping.csv")
    if len(mapping) < 5:
        raise AssertionError("public_feature_mapping.csv has too few mapping rows")

    global_importance = load_json("outputs/global_importance.json")
    if not global_importance:
        raise AssertionError("global_importance.json is empty")

    response_curves = load_json("outputs/response_curves.json")
    if "novelty_score" not in response_curves:
        raise AssertionError("response_curves.json is missing novelty_score")

    print("Project validation passed.")
    print(f"Rows: {len(applications)}, funding rate: {positive_rate:.3f}")
    print("Models:", ", ".join(sorted(model_names)))
    print("Routing counts:")
    print(routing["routing"].value_counts().to_string())


if __name__ == "__main__":
    main()
