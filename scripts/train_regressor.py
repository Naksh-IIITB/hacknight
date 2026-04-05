from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from f1_ai.data import DATA_DIR, MODEL_DIR


CSV_PATH = DATA_DIR / "historical_laps_2025_2026.csv"
MODEL_PATH = MODEL_DIR / "lap_time_regressor.joblib"
META_PATH = MODEL_DIR / "lap_regressor_metadata.json"


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Training data missing: {CSV_PATH}. Run scripts/build_datasets.py first.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(CSV_PATH)
    target = frame.pop("lap_time_sec")

    categorical_features = ["session_code", "compound_key"]
    numeric_features = [column for column in frame.columns if column not in categorical_features]

    pipeline = Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        (
                            "categorical",
                            Pipeline(
                                [
                                    ("imputer", SimpleImputer(strategy="most_frequent")),
                                    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                                ]
                            ),
                            categorical_features,
                        ),
                        (
                            "numeric",
                            Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                            numeric_features,
                        ),
                    ]
                ),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=250,
                    random_state=42,
                    min_samples_leaf=2,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(frame, target, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metadata = {
        "rows": int(len(frame)),
        "feature_columns": list(frame.columns),
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "mae_sec": round(float(mean_absolute_error(y_test, predictions)), 4),
        "rmse_sec": round(float(mean_squared_error(y_test, predictions) ** 0.5), 4),
        "r2": round(float(r2_score(y_test, predictions)), 4),
    }

    joblib.dump(pipeline, MODEL_PATH)
    with META_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
