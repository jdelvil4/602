from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comstock_ca_train.parquet"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comstock_ca_validation.parquet"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "comstock_baseline_validation_metrics.csv"
)

LINEAR_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "comstock_linear_baseline.joblib"
)

LASSO_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "comstock_lasso_baseline.joblib"
)


TARGET_COLUMN = "target_next_hour"
SEASONAL_NAIVE_COLUMN = "seasonal_naive"

LASSO_ALPHA = 0.1


NUMERIC_FEATURES = [
    "in.sqft",
    "in.number_of_stories",
    "stat.occupant_density_ppl_per_m_2",
    "Dry Bulb Temperature [°C]",
    "Relative Humidity [%]",
    "Wind Speed [m/s]",
    "Global Horizontal Radiation [W/m2]",
    "Direct Normal Radiation [W/m2]",
    "Diffuse Horizontal Radiation [W/m2]",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "heating_degree",
    "cooling_degree",
    "lag_1h",
    "lag_24h",
    "rolling_mean_24h",
    "rolling_mean_168h",
]

CATEGORICAL_FEATURES = [
    "in.building_type",
    "in.hvac_system_type",
    "in.heating_fuel",
    "in.service_water_heating_fuel",
    "in.climate_zone_ashrae_2004",
    "in.climate_zone_building_america",
]


def load_split(
    path: Path,
    split_name: str,
) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"{split_name} dataset not found: "
            f"{path}"
        )

    data = pd.read_parquet(
        path
    )

    required_columns = {
        TARGET_COLUMN,
        SEASONAL_NAIVE_COLUMN,
        *NUMERIC_FEATURES,
        *CATEGORICAL_FEATURES,
    }

    missing_columns = (
        required_columns
        - set(
            data.columns
        )
    )

    if missing_columns:
        raise ValueError(
            f"{split_name} dataset is missing "
            f"required columns: "
            f"{sorted(missing_columns)}"
        )

    return data


def validate_split(
    data: pd.DataFrame,
    split_name: str,
) -> None:

    required_columns = [
        TARGET_COLUMN,
        SEASONAL_NAIVE_COLUMN,
        *NUMERIC_FEATURES,
        *CATEGORICAL_FEATURES,
    ]

    missing_counts = (
        data[
            required_columns
        ]
        .isna()
        .sum()
    )

    total_missing = int(
        missing_counts.sum()
    )

    if total_missing != 0:
        print()
        print(
            f"{split_name} missing-value counts:"
        )

        print(
            missing_counts.loc[
                missing_counts > 0
            ].to_string()
        )

        raise ValueError(
            f"{split_name} contains "
            f"{total_missing:,} missing values "
            "in required baseline fields."
        )


def build_preprocessor() -> ColumnTransformer:

    numeric_pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    return preprocessor


def build_linear_pipeline() -> Pipeline:

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                LinearRegression(),
            ),
        ]
    )


def build_lasso_pipeline() -> Pipeline:

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                Lasso(
                    alpha=LASSO_ALPHA,
                    max_iter=5_000,
                ),
            ),
        ]
    )


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float | int]:

    mae = float(
        mean_absolute_error(
            actual,
            predicted,
        )
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                actual,
                predicted,
            )
        )
    )

    actual_mean = float(
        np.mean(
            actual
        )
    )

    if actual_mean == 0.0:
        cvrmse = float(
            "nan"
        )
    else:
        cvrmse = (
            rmse
            / actual_mean
            * 100.0
        )

    nonzero_mask = (
        actual != 0.0
    )

    mape_n = int(
        nonzero_mask.sum()
    )

    if mape_n == 0:
        mape = float(
            "nan"
        )
    else:
        mape = float(
            np.mean(
                np.abs(
                    (
                        actual[
                            nonzero_mask
                        ]
                        - predicted[
                            nonzero_mask
                        ]
                    )
                    / actual[
                        nonzero_mask
                    ]
                )
            )
            * 100.0
        )

    return {
        "mae": mae,
        "rmse": rmse,
        "cvrmse_percent": cvrmse,
        "mape_percent": mape,
        "mape_n": mape_n,
    }


def evaluate_predictions(
    model_name: str,
    actual: np.ndarray,
    predicted: np.ndarray,
    training_seconds: float,
    prediction_seconds: float,
) -> dict[str, object]:

    metrics = calculate_metrics(
        actual,
        predicted,
    )

    return {
        "model": model_name,
        "mae": metrics[
            "mae"
        ],
        "rmse": metrics[
            "rmse"
        ],
        "cvrmse_percent": metrics[
            "cvrmse_percent"
        ],
        "mape_percent": metrics[
            "mape_percent"
        ],
        "mape_n": metrics[
            "mape_n"
        ],
        "training_seconds": training_seconds,
        "prediction_seconds": prediction_seconds,
    }


def main() -> None:

    print(
        "Loading ComStock training split..."
    )

    train = load_split(
        TRAIN_PATH,
        "Training",
    )

    print(
        f"Training rows: "
        f"{len(train):,}"
    )

    print()

    print(
        "Loading ComStock validation split..."
    )

    validation = load_split(
        VALIDATION_PATH,
        "Validation",
    )

    print(
        f"Validation rows: "
        f"{len(validation):,}"
    )

    print()

    print(
        "Validating baseline modeling fields..."
    )

    validate_split(
        train,
        "Training",
    )

    validate_split(
        validation,
        "Validation",
    )

    predictor_columns = [
        *NUMERIC_FEATURES,
        *CATEGORICAL_FEATURES,
    ]

    x_train = train[
        predictor_columns
    ]

    y_train = train[
        TARGET_COLUMN
    ].to_numpy(
        dtype=float
    )

    x_validation = validation[
        predictor_columns
    ]

    y_validation = validation[
        TARGET_COLUMN
    ].to_numpy(
        dtype=float
    )

    results = []

    print()

    print(
        "Evaluating seasonal-naive baseline..."
    )

    seasonal_predictions = validation[
        SEASONAL_NAIVE_COLUMN
    ].to_numpy(
        dtype=float
    )

    seasonal_result = evaluate_predictions(
        model_name="Seasonal Naive",
        actual=y_validation,
        predicted=seasonal_predictions,
        training_seconds=0.0,
        prediction_seconds=0.0,
    )

    results.append(
        seasonal_result
    )

    print(
        f"  MAE: "
        f"{seasonal_result['mae']:.6f}"
    )

    print(
        f"  RMSE: "
        f"{seasonal_result['rmse']:.6f}"
    )

    print(
        f"  CVRMSE: "
        f"{seasonal_result['cvrmse_percent']:.3f}%"
    )

    print(
        f"  MAPE: "
        f"{seasonal_result['mape_percent']:.3f}%"
    )

    print(
        f"  MAPE n: "
        f"{seasonal_result['mape_n']:,}"
    )

    print()

    print(
        "Training Linear Regression baseline..."
    )

    linear_pipeline = build_linear_pipeline()

    start = perf_counter()

    linear_pipeline.fit(
        x_train,
        y_train,
    )

    linear_training_seconds = (
        perf_counter()
        - start
    )

    print(
        "Predicting validation set with "
        "Linear Regression..."
    )

    start = perf_counter()

    linear_predictions = np.array(
        linear_pipeline.predict(
            x_validation
        ),
        dtype=float,
    )

    linear_prediction_seconds = (
        perf_counter()
        - start
    )

    linear_result = evaluate_predictions(
        model_name="Linear Regression",
        actual=y_validation,
        predicted=linear_predictions,
        training_seconds=linear_training_seconds,
        prediction_seconds=linear_prediction_seconds,
    )

    results.append(
        linear_result
    )

    print(
        f"  MAE: "
        f"{linear_result['mae']:.6f}"
    )

    print(
        f"  RMSE: "
        f"{linear_result['rmse']:.6f}"
    )

    print(
        f"  CVRMSE: "
        f"{linear_result['cvrmse_percent']:.3f}%"
    )

    print(
        f"  MAPE: "
        f"{linear_result['mape_percent']:.3f}%"
    )

    print(
        f"  MAPE n: "
        f"{linear_result['mape_n']:,}"
    )

    print(
        f"  Training time: "
        f"{linear_training_seconds:,.2f} seconds"
    )

    print()

    print(
        f"Training Lasso baseline "
        f"(alpha={LASSO_ALPHA})..."
    )

    lasso_pipeline = build_lasso_pipeline()

    start = perf_counter()

    lasso_pipeline.fit(
        x_train,
        y_train,
    )

    lasso_training_seconds = (
        perf_counter()
        - start
    )

    print(
        "Predicting validation set with Lasso..."
    )

    start = perf_counter()

    lasso_predictions = np.asarray(
        lasso_pipeline.predict(
            x_validation
        ),
        dtype=float,
    )

    lasso_prediction_seconds = (
        perf_counter()
        - start
    )

    lasso_result = evaluate_predictions(
        model_name="Lasso",
        actual=y_validation,
        predicted=lasso_predictions,
        training_seconds=lasso_training_seconds,
        prediction_seconds=lasso_prediction_seconds,
    )

    results.append(
        lasso_result
    )

    print(
        f"  MAE: "
        f"{lasso_result['mae']:.6f}"
    )

    print(
        f"  RMSE: "
        f"{lasso_result['rmse']:.6f}"
    )

    print(
        f"  CVRMSE: "
        f"{lasso_result['cvrmse_percent']:.3f}%"
    )

    print(
        f"  MAPE: "
        f"{lasso_result['mape_percent']:.3f}%"
    )

    print(
        f"  MAPE n: "
        f"{lasso_result['mape_n']:,}"
    )

    print(
        f"  Training time: "
        f"{lasso_training_seconds:,.2f} seconds"
    )

    results_df = pd.DataFrame(
        results
    )

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    LINEAR_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    joblib.dump(
        linear_pipeline,
        LINEAR_MODEL_PATH,
    )

    joblib.dump(
        lasso_pipeline,
        LASSO_MODEL_PATH,
    )

    print()

    print(
        "ComStock baseline evaluation complete."
    )

    print()

    print(
        results_df[
            [
                "model",
                "mae",
                "rmse",
                "cvrmse_percent",
                "mape_percent",
                "mape_n",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        f"Saved validation metrics to: "
        f"{RESULTS_PATH}"
    )

    print(
        f"Saved Linear Regression model to: "
        f"{LINEAR_MODEL_PATH}"
    )

    print(
        f"Saved Lasso model to: "
        f"{LASSO_MODEL_PATH}"
    )


if __name__ == "__main__":
    main()