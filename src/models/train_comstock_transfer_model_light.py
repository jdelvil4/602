from pathlib import Path
import time

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from src.evaluation.metrics import calculate_all_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comstock_ca_forecast_features.parquet"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "comstock_shared_feature_random_forest_100trees.joblib"
)

VALIDATION_METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "comstock_shared_feature_validation_metrics_100trees.csv"
)

SOURCE_TIMESTAMP_COLUMN = "timestamp"
SOURCE_TARGET_COLUMN = "target_next_hour"
SOURCE_SEASONAL_COLUMN = "seasonal_naive"
SAMPLE_ROLE_COLUMN = "sample_role"

TARGET_COLUMN = "target_electricity_next_hour"
SEASONAL_COLUMN = "seasonal_naive_24h"
TARGET_TIMESTAMP_COLUMN = "target_timestamp"

RANDOM_STATE = 42

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

N_ESTIMATORS = 100
MAX_DEPTH = None
MIN_SAMPLES_LEAF = 1
MAX_FEATURES = 0.7


FEATURE_MAPPING = {
    "lag_1h":
        "electricity_lag_1",
    "lag_24h":
        "electricity_lag_24",
    "rolling_mean_24h":
        "electricity_rolling_24h_mean",
    "rolling_mean_168h":
        "electricity_rolling_168h_mean",
    "hour_sin":
        "hour_sin",
    "hour_cos":
        "hour_cos",
    "day_of_week_sin":
        "dow_sin",
    "day_of_week_cos":
        "dow_cos",
    "month_sin":
        "month_sin",
    "month_cos":
        "month_cos",
    "is_weekend":
        "is_weekend",
    "Dry Bulb Temperature [°C]":
        "airTemperature",
    "Wind Speed [m/s]":
        "windSpeed",
    "heating_degree":
        "heating_degree",
    "cooling_degree":
        "cooling_degree",
}


SHARED_FEATURE_COLUMNS = [
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


def load_and_standardize() -> pd.DataFrame:

    print(
        "Loading ComStock shared-feature source data..."
    )

    source_columns = [
        "bldg_id",
        SAMPLE_ROLE_COLUMN,
        SOURCE_TIMESTAMP_COLUMN,
        SOURCE_TARGET_COLUMN,
        SOURCE_SEASONAL_COLUMN,
        *FEATURE_MAPPING.keys(),
    ]

    data = pd.read_parquet(
        FEATURE_PATH,
        columns=source_columns,
    )

    data[
        SOURCE_TIMESTAMP_COLUMN
    ] = pd.to_datetime(
        data[
            SOURCE_TIMESTAMP_COLUMN
        ],
        errors="raise",
    )

    data[
        TARGET_TIMESTAMP_COLUMN
    ] = (
        data[
            SOURCE_TIMESTAMP_COLUMN
        ]
        + pd.Timedelta(
            hours=1
        )
    )

    rename_mapping = {
        **FEATURE_MAPPING,
        SOURCE_TARGET_COLUMN:
            TARGET_COLUMN,
        SOURCE_SEASONAL_COLUMN:
            SEASONAL_COLUMN,
    }

    data = data.rename(
        columns=rename_mapping
    )

    required_columns = [
        "bldg_id",
        SAMPLE_ROLE_COLUMN,
        TARGET_TIMESTAMP_COLUMN,
        TARGET_COLUMN,
        SEASONAL_COLUMN,
        *SHARED_FEATURE_COLUMNS,
    ]

    missing_values = int(
        data[
            required_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    if missing_values > 0:
        raise ValueError(
            "ComStock shared-feature data contain "
            f"{missing_values:,} missing required values."
        )

    return data


def split_development_data(
    data: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    development = data.loc[
        data[
            SAMPLE_ROLE_COLUMN
        ]
        == "development"
    ].copy()

    unique_timestamps = (
        pd.DatetimeIndex(
            development[
                TARGET_TIMESTAMP_COLUMN
            ]
            .drop_duplicates()
            .sort_values()
        )
    )

    n_timestamps = len(
        unique_timestamps
    )

    train_end = int(
        n_timestamps
        * TRAIN_FRACTION
    )

    validation_end = int(
        n_timestamps
        * (
            TRAIN_FRACTION
            + VALIDATION_FRACTION
        )
    )

    train_timestamps = (
        unique_timestamps[
            :train_end
        ]
    )

    validation_timestamps = (
        unique_timestamps[
            train_end:
            validation_end
        ]
    )

    train = development.loc[
        development[
            TARGET_TIMESTAMP_COLUMN
        ].isin(
            train_timestamps
        )
    ].copy()

    validation = development.loc[
        development[
            TARGET_TIMESTAMP_COLUMN
        ].isin(
            validation_timestamps
        )
    ].copy()

    return (
        train,
        validation,
    )


def main() -> None:

    data = load_and_standardize()

    print(
        f"Total standardized rows: "
        f"{len(data):,}"
    )

    print(
        f"Total buildings: "
        f"{data['bldg_id'].nunique():,}"
    )

    (
        train,
        validation,
    ) = split_development_data(
        data
    )

    print(
        f"Training rows: "
        f"{len(train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation):,}"
    )

    X_train = train[
        SHARED_FEATURE_COLUMNS
    ]

    y_train = train[
        TARGET_COLUMN
    ]

    X_validation = validation[
        SHARED_FEATURE_COLUMNS
    ]

    y_validation = validation[
        TARGET_COLUMN
    ]

    print()
    print(
        "Training 100-tree shared-feature "
        "ComStock Random Forest..."
    )

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=(
            MIN_SAMPLES_LEAF
        ),
        max_features=MAX_FEATURES,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    start = time.perf_counter()

    model.fit(
        X_train,
        y_train,
    )

    fit_seconds = (
        time.perf_counter()
        - start
    )

    print(
        f"Fit time: "
        f"{fit_seconds:.2f} seconds"
    )

    start = time.perf_counter()

    predictions = model.predict(
        X_validation
    )

    predict_seconds = (
        time.perf_counter()
        - start
    )

    metrics = calculate_all_metrics(
        y_validation,
        predictions,
    )

    seasonal_metrics = (
        calculate_all_metrics(
            y_validation,
            validation[
                SEASONAL_COLUMN
            ],
        )
    )

    results = pd.DataFrame(
        [
            {
                "model":
                    "ComStock Shared-Feature RF 100 Trees",
                "rows":
                    len(validation),
                **metrics,
                "fit_seconds":
                    fit_seconds,
                "predict_seconds":
                    predict_seconds,
            },
            {
                "model":
                    "Seasonal Naive",
                "rows":
                    len(validation),
                **seasonal_metrics,
                "fit_seconds":
                    None,
                "predict_seconds":
                    None,
            },
        ]
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    VALIDATION_METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
        compress=3,
    )

    results.to_csv(
        VALIDATION_METRICS_PATH,
        index=False,
    )

    print()
    print(
        "Validation results:"
    )

    print(
        results.to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved model: "
        f"{MODEL_PATH}"
    )

    print(
        f"Saved validation metrics: "
        f"{VALIDATION_METRICS_PATH}"
    )


if __name__ == "__main__":
    main()