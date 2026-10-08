from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_train.parquet"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_validation.parquet"
)

TEMPORAL_TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_temporal_test.parquet"
)

HELD_OUT_TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_held_out_test.parquet"
)

METRICS_OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "nrel_final_test_metrics.csv"
)

PREDICTIONS_OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "nrel_final_test_predictions.parquet"
)


TARGET_COLUMN = "target_1h_ahead"
TIMESTAMP_COLUMN = "timestamp"
TARGET_TIMESTAMP_COLUMN = "target_timestamp"
BUILDING_ID_COLUMN = "bldg_id"
ROLE_COLUMN = "sample_role"

SEASONAL_NAIVE_COLUMN = "seasonal_naive_24h"

RANDOM_STATE = 42

FINAL_N_ESTIMATORS = 200
FINAL_MAX_DEPTH = 20
FINAL_MIN_SAMPLES_LEAF = 5
FINAL_MAX_FEATURES = 0.7


NUMERIC_FEATURES = [
    "in.sqft",
    "in.occupants",
    "in.bedrooms",
    "in.geometry_stories",
    "electricity_lag_1h",
    "electricity_lag_24h",
    "electricity_rolling_mean_24h",
    "electricity_rolling_mean_168h",
    "weather_dry_bulb_temperature_c",
    "weather_relative_humidity_pct",
    "weather_wind_speed_m_s",
    "weather_global_horizontal_radiation_w_m2",
    "weather_direct_normal_radiation_w_m2",
    "weather_diffuse_horizontal_radiation_w_m2",
    "heating_degree_c",
    "cooling_degree_c",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "is_weekend",
]


CATEGORICAL_FEATURES = [
    "in.geometry_building_type_recs",
    "in.cec_climate_zone",
    "in.vintage",
    "in.heating_fuel",
    "in.hvac_cooling_type",
    "in.hvac_heating_type",
]


FEATURE_COLUMNS = [
    *NUMERIC_FEATURES,
    *CATEGORICAL_FEATURES,
]


def load_dataset(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    required_columns = [
        BUILDING_ID_COLUMN,
        ROLE_COLUMN,
        TIMESTAMP_COLUMN,
        TARGET_TIMESTAMP_COLUMN,
        TARGET_COLUMN,
        SEASONAL_NAIVE_COLUMN,
        *FEATURE_COLUMNS,
    ]

    df = pd.read_parquet(
        path,
        columns=required_columns,
    )

    missing_columns = (
        set(required_columns)
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df[TIMESTAMP_COLUMN] = pd.to_datetime(
        df[TIMESTAMP_COLUMN]
    )

    df[TARGET_TIMESTAMP_COLUMN] = pd.to_datetime(
        df[TARGET_TIMESTAMP_COLUMN]
    )

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"{path.name} contains missing target values."
        )

    if df[SEASONAL_NAIVE_COLUMN].isna().any():
        raise ValueError(
            f"{path.name} contains missing Seasonal Naive values."
        )

    return df


def build_preprocessor() -> ColumnTransformer:

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                    dtype=np.float64,
                ),
            ),
        ]
    )

    return ColumnTransformer(
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
    )


def build_final_model() -> Pipeline:

    model = ExtraTreesRegressor(
        n_estimators=FINAL_N_ESTIMATORS,
        max_depth=FINAL_MAX_DEPTH,
        min_samples_leaf=FINAL_MIN_SAMPLES_LEAF,
        max_features=FINAL_MAX_FEATURES,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                model,
            ),
        ]
    )


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict:

    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    error = (
        actual
        - predicted
    )

    mae = float(
        np.mean(
            np.abs(
                error
            )
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )

    mean_actual = float(
        np.mean(
            actual
        )
    )

    if mean_actual == 0:
        raise ValueError(
            "Cannot calculate CVRMSE because "
            "mean actual is zero."
        )

    cvrmse = float(
        rmse
        / mean_actual
        * 100
    )

    nonzero_mask = (
        actual
        != 0
    )

    mape_n = int(
        nonzero_mask.sum()
    )

    if mape_n == 0:
        raise ValueError(
            "Cannot calculate MAPE because "
            "all actual values are zero."
        )

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
        * 100
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "cvrmse_pct": cvrmse,
        "mape_pct": mape,
        "mape_n": mape_n,
    }


def evaluate_predictions(
    split_name: str,
    model_name: str,
    actual: np.ndarray,
    predicted: np.ndarray,
    rows: int,
    buildings: int,
) -> dict:

    metrics = calculate_metrics(
        actual,
        predicted,
    )

    return {
        "split": split_name,
        "model": model_name,
        "rows": rows,
        "buildings": buildings,
        **metrics,
    }


def validate_test_periods(
    temporal_test: pd.DataFrame,
    held_out_test: pd.DataFrame,
) -> None:

    temporal_times = set(
        temporal_test[
            TARGET_TIMESTAMP_COLUMN
        ].unique()
    )

    held_out_times = set(
        held_out_test[
            TARGET_TIMESTAMP_COLUMN
        ].unique()
    )

    if temporal_times != held_out_times:
        raise ValueError(
            "Temporal and held-out tests do not "
            "cover exactly the same target timestamps."
        )

    if set(
        temporal_test[
            ROLE_COLUMN
        ].unique()
    ) != {"development"}:
        raise ValueError(
            "Temporal test contains unexpected sample roles."
        )

    if set(
        held_out_test[
            ROLE_COLUMN
        ].unique()
    ) != {"held_out"}:
        raise ValueError(
            "Held-out test contains unexpected sample roles."
        )


def create_prediction_output(
    df: pd.DataFrame,
    split_name: str,
    model_predictions: np.ndarray,
) -> pd.DataFrame:

    output = df[
        [
            BUILDING_ID_COLUMN,
            ROLE_COLUMN,
            TIMESTAMP_COLUMN,
            TARGET_TIMESTAMP_COLUMN,
            TARGET_COLUMN,
            SEASONAL_NAIVE_COLUMN,
        ]
    ].copy()

    output.insert(
        0,
        "split",
        split_name,
    )

    output[
        "extra_trees_prediction"
    ] = np.asarray(
        model_predictions,
        dtype=float,
    )

    output[
        "extra_trees_absolute_error"
    ] = np.abs(
        output[
            TARGET_COLUMN
        ]
        - output[
            "extra_trees_prediction"
        ]
    )

    output[
        "seasonal_naive_absolute_error"
    ] = np.abs(
        output[
            TARGET_COLUMN
        ]
        - output[
            SEASONAL_NAIVE_COLUMN
        ]
    )

    return output


def main() -> None:

    print(
        "Loading NREL training and validation datasets..."
    )

    train = load_dataset(
        TRAIN_PATH
    )

    validation = load_dataset(
        VALIDATION_PATH
    )

    print(
        f"Training rows: "
        f"{len(train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation):,}"
    )

    print()

    print(
        "Combining train and validation data "
        "for final model fitting..."
    )

    development_fit = pd.concat(
        [
            train,
            validation,
        ],
        ignore_index=True,
    )

    print(
        f"Final fitting rows: "
        f"{len(development_fit):,}"
    )

    print()

    print(
        "Loading final test datasets..."
    )

    temporal_test = load_dataset(
        TEMPORAL_TEST_PATH
    )

    held_out_test = load_dataset(
        HELD_OUT_TEST_PATH
    )

    print(
        f"Temporal test rows: "
        f"{len(temporal_test):,}"
    )

    print(
        f"Held-out test rows: "
        f"{len(held_out_test):,}"
    )

    validate_test_periods(
        temporal_test,
        held_out_test,
    )

    x_fit = development_fit[
        FEATURE_COLUMNS
    ]

    y_fit = (
        development_fit[
            TARGET_COLUMN
        ]
        .to_numpy(
            dtype=float
        )
    )

    print()

    print(
        "Fitting locked Extra Trees configuration..."
    )

    pipeline = build_final_model()

    fit_start = perf_counter()

    pipeline.fit(
        x_fit,
        y_fit,
    )

    fit_seconds = (
        perf_counter()
        - fit_start
    )

    print(
        f"Final model fitting time: "
        f"{fit_seconds:.2f} seconds"
    )

    print()

    results = []
    prediction_frames = []

    test_sets = [
        (
            "temporal_test",
            temporal_test,
        ),
        (
            "held_out_test",
            held_out_test,
        ),
    ]

    for (
        split_name,
        test_df,
    ) in test_sets:

        print(
            f"Evaluating {split_name}..."
        )

        x_test = test_df[
            FEATURE_COLUMNS
        ]

        y_test = (
            test_df[
                TARGET_COLUMN
            ]
            .to_numpy(
                dtype=float
            )
        )

        prediction_start = (
            perf_counter()
        )

        model_predictions = np.asarray(
            pipeline.predict(
                x_test
            ),
            dtype=float,
        )

        prediction_seconds = (
            perf_counter()
            - prediction_start
        )

        seasonal_predictions = (
            test_df[
                SEASONAL_NAIVE_COLUMN
            ]
            .to_numpy(
                dtype=float
            )
        )

        buildings = test_df[
            BUILDING_ID_COLUMN
        ].nunique()

        model_result = evaluate_predictions(
            split_name,
            "Extra Trees",
            y_test,
            model_predictions,
            len(
                test_df
            ),
            buildings,
        )

        model_result[
            "prediction_seconds"
        ] = prediction_seconds

        model_result[
            "final_fit_seconds"
        ] = fit_seconds

        seasonal_result = (
            evaluate_predictions(
                split_name,
                "Seasonal Naive",
                y_test,
                seasonal_predictions,
                len(
                    test_df
                ),
                buildings,
            )
        )

        seasonal_result[
            "prediction_seconds"
        ] = 0.0

        seasonal_result[
            "final_fit_seconds"
        ] = 0.0

        results.extend(
            [
                model_result,
                seasonal_result,
            ]
        )

        model_mae = (
            model_result[
                "mae"
            ]
        )

        seasonal_mae = (
            seasonal_result[
                "mae"
            ]
        )

        improvement = (
            (
                seasonal_mae
                - model_mae
            )
            / seasonal_mae
            * 100
        )

        print(
            f"  Extra Trees MAE: "
            f"{model_mae:.6f}"
        )

        print(
            f"  Extra Trees RMSE: "
            f"{model_result['rmse']:.6f}"
        )

        print(
            f"  Extra Trees CVRMSE: "
            f"{model_result['cvrmse_pct']:.3f}%"
        )

        print(
            f"  Extra Trees MAPE: "
            f"{model_result['mape_pct']:.3f}%"
        )

        print(
            f"  Seasonal Naive MAE: "
            f"{seasonal_mae:.6f}"
        )

        print(
            f"  Extra Trees MAE improvement "
            f"over Seasonal Naive: "
            f"{improvement:.2f}%"
        )

        print()

        prediction_frame = (
            create_prediction_output(
                test_df,
                split_name,
                model_predictions,
            )
        )

        prediction_frames.append(
            prediction_frame
        )

    results_df = pd.DataFrame(
        results
    )

    METRICS_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        METRICS_OUTPUT_PATH,
        index=False,
    )

    prediction_output = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    prediction_output.to_parquet(
        PREDICTIONS_OUTPUT_PATH,
        index=False,
        compression="snappy",
    )

    print(
        "Final NREL evaluation complete."
    )

    print()

    print(
        results_df.to_string(
            index=False
        )
    )

    print()

    print(
        f"Saved final metrics to: "
        f"{METRICS_OUTPUT_PATH}"
    )

    print(
        f"Saved test predictions to: "
        f"{PREDICTIONS_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()