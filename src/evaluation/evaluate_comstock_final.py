from pathlib import Path
from time import perf_counter
from typing import TypedDict

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEMPORAL_TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comstock_ca_temporal_test.parquet"
)

HELD_OUT_TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comstock_ca_held_out_test.parquet"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "comstock_random_forest_tuned.joblib"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "comstock_final_test_metrics.csv"
)

BUILDING_METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "comstock_final_building_metrics.csv"
)

PREDICTIONS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
)

TEMPORAL_PREDICTIONS_PATH = (
    PREDICTIONS_DIR
    / "comstock_temporal_test_predictions.parquet"
)

HELD_OUT_PREDICTIONS_PATH = (
    PREDICTIONS_DIR
    / "comstock_held_out_test_predictions.parquet"
)


BUILDING_ID_COLUMN = "bldg_id"
TIMESTAMP_COLUMN = "timestamp"
TARGET_TIMESTAMP_COLUMN = "target_timestamp"
TARGET_COLUMN = "target_next_hour"
SEASONAL_NAIVE_COLUMN = "seasonal_naive"


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


class MetricResult(TypedDict):
    mae: float
    rmse: float
    cvrmse_percent: float
    mape_percent: float
    mape_n: int


class EvaluationResult(TypedDict):
    evaluation_set: str
    model: str
    mae: float
    rmse: float
    cvrmse_percent: float
    mape_percent: float
    mape_n: int
    mae_improvement_vs_seasonal_percent: float


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
        BUILDING_ID_COLUMN,
        TIMESTAMP_COLUMN,
        TARGET_TIMESTAMP_COLUMN,
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

    data[
        TIMESTAMP_COLUMN
    ] = pd.to_datetime(
        data[
            TIMESTAMP_COLUMN
        ],
        errors="raise",
    )

    data[
        TARGET_TIMESTAMP_COLUMN
    ] = pd.to_datetime(
        data[
            TARGET_TIMESTAMP_COLUMN
        ],
        errors="raise",
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
            f"{split_name} missing values:"
        )

        print(
            missing_counts.loc[
                missing_counts > 0
            ].to_string()
        )

        raise ValueError(
            f"{split_name} contains "
            f"{total_missing:,} missing values "
            "in required evaluation fields."
        )

    duplicate_count = int(
        data.duplicated(
            subset=[
                BUILDING_ID_COLUMN,
                TARGET_TIMESTAMP_COLUMN,
            ]
        ).sum()
    )

    if duplicate_count != 0:
        raise ValueError(
            f"{split_name} contains "
            f"{duplicate_count:,} duplicate "
            "building-target-timestamp rows."
        )


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> MetricResult:

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
    evaluation_set: str,
    model_name: str,
    actual: np.ndarray,
    predicted: np.ndarray,
    mae_improvement_vs_seasonal_percent: float,
) -> EvaluationResult:

    metrics = calculate_metrics(
        actual,
        predicted,
    )

    return {
        "evaluation_set": evaluation_set,
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
        "mae_improvement_vs_seasonal_percent": (
            mae_improvement_vs_seasonal_percent
        ),
    }


def calculate_building_metrics(
    data: pd.DataFrame,
    evaluation_set: str,
) -> pd.DataFrame:

    rows: list[
        dict[str, object]
    ] = []

    for building_id, group in data.groupby(
        BUILDING_ID_COLUMN
    ):

        actual = group[
            TARGET_COLUMN
        ].to_numpy(
            dtype=float
        )

        model_predictions = group[
            "model_prediction"
        ].to_numpy(
            dtype=float
        )

        seasonal_predictions = group[
            SEASONAL_NAIVE_COLUMN
        ].to_numpy(
            dtype=float
        )

        model_metrics = calculate_metrics(
            actual,
            model_predictions,
        )

        seasonal_metrics = calculate_metrics(
            actual,
            seasonal_predictions,
        )

        model_mae = model_metrics[
            "mae"
        ]

        seasonal_mae = seasonal_metrics[
            "mae"
        ]

        if seasonal_mae == 0.0:
            mae_improvement_percent = float(
                "nan"
            )
        else:
            mae_improvement_percent = (
                (
                    seasonal_mae
                    - model_mae
                )
                / seasonal_mae
                * 100.0
            )

        rows.append(
            {
                "evaluation_set": evaluation_set,
                "bldg_id": building_id,
                "rows": len(
                    group
                ),
                "actual_mean": float(
                    np.mean(
                        actual
                    )
                ),
                "model_mae": model_mae,
                "model_rmse": model_metrics[
                    "rmse"
                ],
                "model_cvrmse_percent": model_metrics[
                    "cvrmse_percent"
                ],
                "model_mape_percent": model_metrics[
                    "mape_percent"
                ],
                "seasonal_mae": seasonal_mae,
                "seasonal_rmse": seasonal_metrics[
                    "rmse"
                ],
                "seasonal_cvrmse_percent": seasonal_metrics[
                    "cvrmse_percent"
                ],
                "seasonal_mape_percent": seasonal_metrics[
                    "mape_percent"
                ],
                "mae_improvement_percent": (
                    mae_improvement_percent
                ),
                "model_beats_seasonal_mae": (
                    model_mae
                    < seasonal_mae
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def evaluate_dataset(
    model: Pipeline,
    data: pd.DataFrame,
    evaluation_set: str,
) -> tuple[
    list[EvaluationResult],
    pd.DataFrame,
    pd.DataFrame,
    float,
]:

    predictor_columns = [
        *NUMERIC_FEATURES,
        *CATEGORICAL_FEATURES,
    ]

    x = data[
        predictor_columns
    ]

    actual = data[
        TARGET_COLUMN
    ].to_numpy(
        dtype=float
    )

    seasonal_predictions = data[
        SEASONAL_NAIVE_COLUMN
    ].to_numpy(
        dtype=float
    )

    print(
        f"Predicting {evaluation_set} "
        "with tuned Random Forest..."
    )

    start = perf_counter()

    model_predictions = np.asarray(
        model.predict(
            x
        ),
        dtype=float,
    )

    prediction_seconds = (
        perf_counter()
        - start
    )

    model_metrics = calculate_metrics(
        actual,
        model_predictions,
    )

    seasonal_metrics = calculate_metrics(
        actual,
        seasonal_predictions,
    )

    model_mae = model_metrics[
        "mae"
    ]

    seasonal_mae = seasonal_metrics[
        "mae"
    ]

    if seasonal_mae == 0.0:
        mae_improvement_percent = float(
            "nan"
        )
    else:
        mae_improvement_percent = (
            (
                seasonal_mae
                - model_mae
            )
            / seasonal_mae
            * 100.0
        )

    model_result = evaluate_predictions(
        evaluation_set=evaluation_set,
        model_name="Tuned Random Forest",
        actual=actual,
        predicted=model_predictions,
        mae_improvement_vs_seasonal_percent=(
            mae_improvement_percent
        ),
    )

    seasonal_result = evaluate_predictions(
        evaluation_set=evaluation_set,
        model_name="Seasonal Naive",
        actual=actual,
        predicted=seasonal_predictions,
        mae_improvement_vs_seasonal_percent=0.0,
    )

    predictions = data[
        [
            BUILDING_ID_COLUMN,
            TIMESTAMP_COLUMN,
            TARGET_TIMESTAMP_COLUMN,
            TARGET_COLUMN,
            SEASONAL_NAIVE_COLUMN,
            "in.building_type",
        ]
    ].copy()

    predictions[
        "model_prediction"
    ] = model_predictions

    predictions[
        "model_residual"
    ] = (
        predictions[
            TARGET_COLUMN
        ]
        - predictions[
            "model_prediction"
        ]
    )

    predictions[
        "seasonal_residual"
    ] = (
        predictions[
            TARGET_COLUMN
        ]
        - predictions[
            SEASONAL_NAIVE_COLUMN
        ]
    )

    building_metrics = calculate_building_metrics(
        predictions,
        evaluation_set,
    )

    return (
        [
            model_result,
            seasonal_result,
        ],
        predictions,
        building_metrics,
        prediction_seconds,
    )


def print_result(
    result: EvaluationResult,
) -> None:

    print(
        f"  {result['model']}"
    )

    print(
        f"    MAE: "
        f"{result['mae']:.6f}"
    )

    print(
        f"    RMSE: "
        f"{result['rmse']:.6f}"
    )

    print(
        f"    CVRMSE: "
        f"{result['cvrmse_percent']:.3f}%"
    )

    print(
        f"    MAPE: "
        f"{result['mape_percent']:.3f}%"
    )

    print(
        f"    MAPE n: "
        f"{result['mape_n']:,}"
    )

    print(
        f"    MAE improvement vs seasonal: "
        f"{result['mae_improvement_vs_seasonal_percent']:.2f}%"
    )


def main() -> None:

    total_start = perf_counter()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Tuned Random Forest model not found: "
            f"{MODEL_PATH}"
        )

    print(
        "Loading locked tuned ComStock "
        "Random Forest model..."
    )

    loaded_model = joblib.load(
        MODEL_PATH
    )

    if not isinstance(
        loaded_model,
        Pipeline,
    ):
        raise TypeError(
            "Loaded model is not a "
            "scikit-learn Pipeline."
        )

    model = loaded_model

    print()

    print(
        "Loading temporal test split..."
    )

    temporal_test = load_split(
        TEMPORAL_TEST_PATH,
        "Temporal test",
    )

    print(
        f"Temporal test rows: "
        f"{len(temporal_test):,}"
    )

    print(
        f"Temporal test buildings: "
        f"{temporal_test[BUILDING_ID_COLUMN].nunique():,}"
    )

    print()

    print(
        "Loading held-out building test split..."
    )

    held_out_test = load_split(
        HELD_OUT_TEST_PATH,
        "Held-out test",
    )

    print(
        f"Held-out test rows: "
        f"{len(held_out_test):,}"
    )

    print(
        f"Held-out test buildings: "
        f"{held_out_test[BUILDING_ID_COLUMN].nunique():,}"
    )

    print()

    print(
        "Validating final evaluation datasets..."
    )

    validate_split(
        temporal_test,
        "Temporal test",
    )

    validate_split(
        held_out_test,
        "Held-out test",
    )

    temporal_start = temporal_test[
        TARGET_TIMESTAMP_COLUMN
    ].min()

    temporal_end = temporal_test[
        TARGET_TIMESTAMP_COLUMN
    ].max()

    held_out_start = held_out_test[
        TARGET_TIMESTAMP_COLUMN
    ].min()

    held_out_end = held_out_test[
        TARGET_TIMESTAMP_COLUMN
    ].max()

    if (
        temporal_start != held_out_start
        or temporal_end != held_out_end
    ):
        raise ValueError(
            "Temporal and held-out test periods "
            "do not match."
        )

    print(
        f"Shared final target period: "
        f"{temporal_start} "
        f"to "
        f"{temporal_end}"
    )

    print()

    (
        temporal_results,
        temporal_predictions,
        temporal_building_metrics,
        temporal_prediction_seconds,
    ) = evaluate_dataset(
        model,
        temporal_test,
        "temporal_test",
    )

    print()

    print(
        "Temporal-test results:"
    )

    for result in temporal_results:
        print_result(
            result
        )

    print(
        f"  Model prediction time: "
        f"{temporal_prediction_seconds:,.2f} seconds"
    )

    print()

    (
        held_out_results,
        held_out_predictions,
        held_out_building_metrics,
        held_out_prediction_seconds,
    ) = evaluate_dataset(
        model,
        held_out_test,
        "held_out_test",
    )

    print()

    print(
        "Held-out-building results:"
    )

    for result in held_out_results:
        print_result(
            result
        )

    print(
        f"  Model prediction time: "
        f"{held_out_prediction_seconds:,.2f} seconds"
    )

    print()

    all_results = pd.DataFrame(
        [
            *temporal_results,
            *held_out_results,
        ]
    )

    all_building_metrics = pd.concat(
        [
            temporal_building_metrics,
            held_out_building_metrics,
        ],
        ignore_index=True,
    )

    temporal_model_buildings_beaten = int(
        temporal_building_metrics[
            "model_beats_seasonal_mae"
        ].sum()
    )

    temporal_buildings = len(
        temporal_building_metrics
    )

    held_out_model_buildings_beaten = int(
        held_out_building_metrics[
            "model_beats_seasonal_mae"
        ].sum()
    )

    held_out_buildings = len(
        held_out_building_metrics
    )

    temporal_beaten_percent = (
        temporal_model_buildings_beaten
        / temporal_buildings
        * 100.0
    )

    held_out_beaten_percent = (
        held_out_model_buildings_beaten
        / held_out_buildings
        * 100.0
    )

    print(
        "Per-building MAE comparison:"
    )

    print(
        f"  Temporal test: model beat seasonal "
        f"for {temporal_model_buildings_beaten:,}/"
        f"{temporal_buildings:,} buildings "
        f"({temporal_beaten_percent:.1f}%)"
    )

    print(
        f"  Held-out test: model beat seasonal "
        f"for {held_out_model_buildings_beaten:,}/"
        f"{held_out_buildings:,} buildings "
        f"({held_out_beaten_percent:.1f}%)"
    )

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results.to_csv(
        METRICS_PATH,
        index=False,
    )

    all_building_metrics.to_csv(
        BUILDING_METRICS_PATH,
        index=False,
    )

    temporal_predictions.to_parquet(
        TEMPORAL_PREDICTIONS_PATH,
        index=False,
    )

    held_out_predictions.to_parquet(
        HELD_OUT_PREDICTIONS_PATH,
        index=False,
    )

    total_seconds = (
        perf_counter()
        - total_start
    )

    print()

    print(
        "ComStock final evaluation complete."
    )

    print()

    print(
        all_results.to_string(
            index=False
        )
    )

    print()

    print(
        f"Elapsed time: "
        f"{total_seconds:,.2f} seconds"
    )

    print()

    print(
        f"Saved final metrics to: "
        f"{METRICS_PATH}"
    )

    print(
        f"Saved building-level metrics to: "
        f"{BUILDING_METRICS_PATH}"
    )

    print(
        f"Saved temporal predictions to: "
        f"{TEMPORAL_PREDICTIONS_PATH}"
    )

    print(
        f"Saved held-out predictions to: "
        f"{HELD_OUT_PREDICTIONS_PATH}"
    )


if __name__ == "__main__":
    main()