from pathlib import Path
import time

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import calculate_all_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bdg2_train.parquet"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bdg2_validation.parquet"
)

OUTPUT_TABLE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "bdg2_baseline_validation_metrics.csv"
)

LINEAR_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "bdg2_linear_baseline.joblib"
)

LASSO_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "bdg2_lasso_baseline.joblib"
)

TARGET_COLUMN = (
    "target_electricity_next_hour"
)

SEASONAL_NAIVE_COLUMN = (
    "seasonal_naive_24h"
)

FEATURE_COLUMNS = [
    "electricity_lag_1",
    "electricity_lag_24",
    "electricity_rolling_24h_mean",
    "electricity_rolling_168h_mean",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "airTemperature",
    "windSpeed",
    "heating_degree",
    "cooling_degree",
]

LASSO_ALPHA = 0.1


def build_linear_pipeline(
    model: object,
) -> Pipeline:

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median",
                            ),
                        ),
                        (
                            "scaler",
                            StandardScaler(),
                        ),
                    ]
                ),
                FEATURE_COLUMNS,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessing,
            ),
            (
                "model",
                model,
            ),
        ]
    )


def evaluate_model(
    name: str,
    actual: pd.Series,
    predicted,
    fit_seconds: float | None = None,
    predict_seconds: float | None = None,
) -> dict[str, object]:

    metrics = calculate_all_metrics(
        actual,
        predicted,
    )

    result: dict[str, object] = {
        "model": name,
        **metrics,
    }

    if fit_seconds is not None:
        result[
            "fit_seconds"
        ] = fit_seconds

    if predict_seconds is not None:
        result[
            "predict_seconds"
        ] = predict_seconds

    return result


def main() -> None:

    print(
        "Loading BDG2 training and validation data..."
    )

    required_columns = [
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
        SEASONAL_NAIVE_COLUMN,
    ]

    train = pd.read_parquet(
        TRAIN_PATH,
        columns=required_columns,
    )

    validation = pd.read_parquet(
        VALIDATION_PATH,
        columns=required_columns,
    )

    print(
        f"Training rows: "
        f"{len(train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation):,}"
    )

    missing_train = int(
        train[
            required_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    missing_validation = int(
        validation[
            required_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Missing required training values: "
        f"{missing_train:,}"
    )

    print(
        f"Missing required validation values: "
        f"{missing_validation:,}"
    )

    if missing_train > 0:
        raise ValueError(
            "Training data contain missing "
            "required values."
        )

    if missing_validation > 0:
        raise ValueError(
            "Validation data contain missing "
            "required values."
        )

    X_train = train[
        FEATURE_COLUMNS
    ].copy()

    y_train = train[
        TARGET_COLUMN
    ].copy()

    X_validation = validation[
        FEATURE_COLUMNS
    ].copy()

    y_validation = validation[
        TARGET_COLUMN
    ].copy()

    results: list[
        dict[str, object]
    ] = []

    print()
    print(
        "Evaluating Seasonal Naive..."
    )

    seasonal_result = evaluate_model(
        name="Seasonal Naive",
        actual=y_validation,
        predicted=validation[
            SEASONAL_NAIVE_COLUMN
        ],
    )

    results.append(
        seasonal_result
    )

    print(
        seasonal_result
    )

    print()
    print(
        "Training Linear Regression..."
    )

    linear = build_linear_pipeline(
        LinearRegression()
    )

    start = time.perf_counter()

    linear.fit(
        X_train,
        y_train,
    )

    linear_fit_seconds = (
        time.perf_counter()
        - start
    )

    start = time.perf_counter()

    linear_predictions = (
        linear.predict(
            X_validation
        )
    )

    linear_predict_seconds = (
        time.perf_counter()
        - start
    )

    linear_result = evaluate_model(
        name="Linear Regression",
        actual=y_validation,
        predicted=linear_predictions,
        fit_seconds=linear_fit_seconds,
        predict_seconds=(
            linear_predict_seconds
        ),
    )

    results.append(
        linear_result
    )

    print(
        linear_result
    )

    print()
    print(
        "Training Lasso..."
    )

    lasso = build_linear_pipeline(
        Lasso(
            alpha=LASSO_ALPHA,
            max_iter=10000,
        )
    )

    start = time.perf_counter()

    lasso.fit(
        X_train,
        y_train,
    )

    lasso_fit_seconds = (
        time.perf_counter()
        - start
    )

    start = time.perf_counter()

    lasso_predictions = (
        lasso.predict(
            X_validation
        )
    )

    lasso_predict_seconds = (
        time.perf_counter()
        - start
    )

    lasso_result = evaluate_model(
        name="Lasso alpha=0.1",
        actual=y_validation,
        predicted=lasso_predictions,
        fit_seconds=lasso_fit_seconds,
        predict_seconds=(
            lasso_predict_seconds
        ),
    )

    results.append(
        lasso_result
    )

    print(
        lasso_result
    )

    results_df = pd.DataFrame(
        results
    )

    OUTPUT_TABLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    LINEAR_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_TABLE_PATH,
        index=False,
    )

    joblib.dump(
        linear,
        LINEAR_MODEL_PATH,
    )

    joblib.dump(
        lasso,
        LASSO_MODEL_PATH,
    )

    print()
    print(
        "BDG2 baseline validation complete."
    )

    print()

    print(
        results_df.to_string(
            index=False
        )
    )

    print()

    print(
        f"Metrics: "
        f"{OUTPUT_TABLE_PATH}"
    )

    print(
        f"Linear model: "
        f"{LINEAR_MODEL_PATH}"
    )

    print(
        f"Lasso model: "
        f"{LASSO_MODEL_PATH}"
    )


if __name__ == "__main__":
    main()