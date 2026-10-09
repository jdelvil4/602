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
    / "comstock_shared_feature_random_forest.joblib"
)

VALIDATION_METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "comstock_shared_feature_validation_metrics.csv"
)

FEATURE_CONTRACT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "comstock_bdg2_shared_feature_contract.csv"
)

SPLIT_SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "comstock_shared_feature_split_summary.csv"
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

N_ESTIMATORS = 200
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

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Standardized ComStock data are missing "
            f"required columns: {missing_columns}"
        )

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

    duplicate_count = int(
        data.duplicated(
            subset=[
                "bldg_id",
                TARGET_TIMESTAMP_COLUMN,
            ]
        ).sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            "ComStock shared-feature data contain "
            f"{duplicate_count:,} duplicate "
            "building-target timestamp rows."
        )

    return data


def split_development_data(
    data: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    development = data.loc[
        data[
            SAMPLE_ROLE_COLUMN
        ]
        == "development"
    ].copy()

    held_out = data.loc[
        data[
            SAMPLE_ROLE_COLUMN
        ]
        == "held_out"
    ].copy()

    if development.empty:
        raise ValueError(
            "No development buildings found."
        )

    if held_out.empty:
        raise ValueError(
            "No held-out buildings found."
        )

    development_buildings = (
        development[
            "bldg_id"
        ].nunique()
    )

    held_out_buildings = (
        held_out[
            "bldg_id"
        ].nunique()
    )

    print(
        f"Development buildings: "
        f"{development_buildings:,}"
    )

    print(
        f"Held-out buildings: "
        f"{held_out_buildings:,}"
    )

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

    temporal_test_timestamps = (
        unique_timestamps[
            validation_end:
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

    temporal_test = development.loc[
        development[
            TARGET_TIMESTAMP_COLUMN
        ].isin(
            temporal_test_timestamps
        )
    ].copy()

    if train.empty:
        raise ValueError(
            "Training split is empty."
        )

    if validation.empty:
        raise ValueError(
            "Validation split is empty."
        )

    if temporal_test.empty:
        raise ValueError(
            "Temporal test split is empty."
        )

    return (
        train,
        validation,
        temporal_test,
    )


def evaluate_predictions(
    model_name: str,
    actual: pd.Series,
    predicted,
    seasonal_predictions: pd.Series,
    fit_seconds: float,
    predict_seconds: float,
) -> pd.DataFrame:

    model_metrics = (
        calculate_all_metrics(
            actual,
            predicted,
        )
    )

    seasonal_metrics = (
        calculate_all_metrics(
            actual,
            seasonal_predictions,
        )
    )

    rows = [
        {
            "model":
                model_name,
            "rows":
                len(actual),
            **model_metrics,
            "fit_seconds":
                fit_seconds,
            "predict_seconds":
                predict_seconds,
        },
        {
            "model":
                "Seasonal Naive",
            "rows":
                len(actual),
            **seasonal_metrics,
            "fit_seconds":
                None,
            "predict_seconds":
                None,
        },
    ]

    return pd.DataFrame(
        rows
    )


def create_feature_contract() -> pd.DataFrame:

    rows: list[
        dict[str, str]
    ] = []

    for (
        comstock_column,
        canonical_column,
    ) in FEATURE_MAPPING.items():

        rows.append(
            {
                "comstock_column":
                    comstock_column,
                "canonical_feature":
                    canonical_column,
                "bdg2_column":
                    canonical_column,
            }
        )

    return pd.DataFrame(
        rows
    )


def create_split_summary(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    temporal_test: pd.DataFrame,
) -> pd.DataFrame:

    rows: list[
        dict[str, object]
    ] = []

    datasets = [
        (
            "train",
            train,
        ),
        (
            "validation",
            validation,
        ),
        (
            "temporal_test_unused",
            temporal_test,
        ),
    ]

    for name, frame in datasets:

        rows.append(
            {
                "split":
                    name,
                "rows":
                    len(frame),
                "buildings":
                    frame[
                        "bldg_id"
                    ].nunique(),
                "unique_target_timestamps":
                    frame[
                        TARGET_TIMESTAMP_COLUMN
                    ].nunique(),
                "target_start":
                    frame[
                        TARGET_TIMESTAMP_COLUMN
                    ].min(),
                "target_end":
                    frame[
                        TARGET_TIMESTAMP_COLUMN
                    ].max(),
            }
        )

    return pd.DataFrame(
        rows
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
        temporal_test,
    ) = split_development_data(
        data
    )

    print()
    print(
        "Shared-feature chronological split:"
    )

    print(
        f"Training rows: "
        f"{len(train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation):,}"
    )

    print(
        f"Unused temporal-test rows: "
        f"{len(temporal_test):,}"
    )

    print()

    print(
        f"Training target period: "
        f"{train[TARGET_TIMESTAMP_COLUMN].min()} "
        f"to "
        f"{train[TARGET_TIMESTAMP_COLUMN].max()}"
    )

    print(
        f"Validation target period: "
        f"{validation[TARGET_TIMESTAMP_COLUMN].min()} "
        f"to "
        f"{validation[TARGET_TIMESTAMP_COLUMN].max()}"
    )

    print(
        f"Unused temporal-test target period: "
        f"{temporal_test[TARGET_TIMESTAMP_COLUMN].min()} "
        f"to "
        f"{temporal_test[TARGET_TIMESTAMP_COLUMN].max()}"
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
        "Training shared-feature ComStock "
        "Random Forest..."
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

    validation_predictions = (
        model.predict(
            X_validation
        )
    )

    predict_seconds = (
        time.perf_counter()
        - start
    )

    validation_results = (
        evaluate_predictions(
            model_name=(
                "ComStock Shared-Feature "
                "Random Forest"
            ),
            actual=y_validation,
            predicted=(
                validation_predictions
            ),
            seasonal_predictions=(
                validation[
                    SEASONAL_COLUMN
                ]
            ),
            fit_seconds=(
                fit_seconds
            ),
            predict_seconds=(
                predict_seconds
            ),
        )
    )

    print()
    print(
        "ComStock shared-feature "
        "validation results:"
    )

    print(
        validation_results.to_string(
            index=False
        )
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
    )

    validation_results.to_csv(
        VALIDATION_METRICS_PATH,
        index=False,
    )

    feature_contract = (
        create_feature_contract()
    )

    feature_contract.to_csv(
        FEATURE_CONTRACT_PATH,
        index=False,
    )

    split_summary = (
        create_split_summary(
            train,
            validation,
            temporal_test,
        )
    )

    split_summary.to_csv(
        SPLIT_SUMMARY_PATH,
        index=False,
    )

    print()
    print(
        "Shared-feature ComStock "
        "model training complete."
    )

    print()

    print(
        f"Model: "
        f"{MODEL_PATH}"
    )

    print(
        f"Validation metrics: "
        f"{VALIDATION_METRICS_PATH}"
    )

    print(
        f"Feature contract: "
        f"{FEATURE_CONTRACT_PATH}"
    )

    print(
        f"Split summary: "
        f"{SPLIT_SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()