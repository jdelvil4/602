from pathlib import Path
from time import perf_counter

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comstock_ca_forecast_features.parquet"
)

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

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comstock_ca_split_summary.csv"
)


BUILDING_ID_COLUMN = "bldg_id"
ROLE_COLUMN = "sample_role"
TIMESTAMP_COLUMN = "timestamp"
TARGET_COLUMN = "target_next_hour"
TARGET_TIMESTAMP_COLUMN = "target_timestamp"

DEVELOPMENT_ROLE = "development"
HELD_OUT_ROLE = "held_out"

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15

FORECAST_HORIZON_HOURS = 1

EXPECTED_TOTAL_BUILDINGS = 250
EXPECTED_DEVELOPMENT_BUILDINGS = 200
EXPECTED_HELD_OUT_BUILDINGS = 50

EXPECTED_ROWS_PER_BUILDING = 8_591
EXPECTED_UNIQUE_TIMESTAMPS = 8_591

EXPECTED_TRAIN_TIMESTAMPS = 6_013
EXPECTED_VALIDATION_TIMESTAMPS = 1_288
EXPECTED_TEST_TIMESTAMPS = 1_290

EXPECTED_TRAIN_ROWS = (
    EXPECTED_DEVELOPMENT_BUILDINGS
    * EXPECTED_TRAIN_TIMESTAMPS
)

EXPECTED_VALIDATION_ROWS = (
    EXPECTED_DEVELOPMENT_BUILDINGS
    * EXPECTED_VALIDATION_TIMESTAMPS
)

EXPECTED_TEMPORAL_TEST_ROWS = (
    EXPECTED_DEVELOPMENT_BUILDINGS
    * EXPECTED_TEST_TIMESTAMPS
)

EXPECTED_HELD_OUT_TEST_ROWS = (
    EXPECTED_HELD_OUT_BUILDINGS
    * EXPECTED_TEST_TIMESTAMPS
)


def load_features() -> pd.DataFrame:

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"ComStock feature dataset not found: "
            f"{INPUT_PATH}"
        )

    data = pd.read_parquet(
        INPUT_PATH
    )

    required_columns = {
        BUILDING_ID_COLUMN,
        ROLE_COLUMN,
        TIMESTAMP_COLUMN,
        TARGET_COLUMN,
    }

    missing_columns = (
        required_columns
        - set(
            data.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Feature dataset is missing required columns: "
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
    ] = (
        data[
            TIMESTAMP_COLUMN
        ]
        + pd.Timedelta(
            hours=FORECAST_HORIZON_HOURS
        )
    )

    return data


def validate_input(
    data: pd.DataFrame,
) -> None:

    total_buildings = data[
        BUILDING_ID_COLUMN
    ].nunique()

    if total_buildings != EXPECTED_TOTAL_BUILDINGS:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_BUILDINGS} "
            f"buildings but found {total_buildings}."
        )

    development_buildings = data.loc[
        data[
            ROLE_COLUMN
        ]
        == DEVELOPMENT_ROLE,
        BUILDING_ID_COLUMN,
    ].nunique()

    held_out_buildings = data.loc[
        data[
            ROLE_COLUMN
        ]
        == HELD_OUT_ROLE,
        BUILDING_ID_COLUMN,
    ].nunique()

    if (
        development_buildings
        != EXPECTED_DEVELOPMENT_BUILDINGS
    ):
        raise ValueError(
            f"Expected "
            f"{EXPECTED_DEVELOPMENT_BUILDINGS} "
            f"development buildings but found "
            f"{development_buildings}."
        )

    if (
        held_out_buildings
        != EXPECTED_HELD_OUT_BUILDINGS
    ):
        raise ValueError(
            f"Expected "
            f"{EXPECTED_HELD_OUT_BUILDINGS} "
            f"held-out buildings but found "
            f"{held_out_buildings}."
        )

    rows_per_building = (
        data
        .groupby(
            BUILDING_ID_COLUMN
        )
        .size()
    )

    invalid_rows = rows_per_building.loc[
        rows_per_building
        != EXPECTED_ROWS_PER_BUILDING
    ]

    if not invalid_rows.empty:
        raise ValueError(
            "Some buildings do not contain exactly "
            f"{EXPECTED_ROWS_PER_BUILDING:,} "
            "feature rows."
        )

    duplicate_count = int(
        data.duplicated(
            subset=[
                BUILDING_ID_COLUMN,
                TIMESTAMP_COLUMN,
            ]
        )
        .sum()
    )

    if duplicate_count != 0:
        raise ValueError(
            f"Input feature dataset contains "
            f"{duplicate_count:,} duplicate "
            "building-timestamp rows."
        )

    if data[
        TARGET_COLUMN
    ].isna().any():
        raise ValueError(
            "Input feature dataset contains "
            "missing forecast targets."
        )


def get_split_timestamps(
    development: pd.DataFrame,
) -> tuple[
    pd.DatetimeIndex,
    pd.DatetimeIndex,
    pd.DatetimeIndex,
]:

    timestamps = pd.DatetimeIndex(
        development[
            TARGET_TIMESTAMP_COLUMN
        ]
        .drop_duplicates()
        .sort_values()
    )

    total_timestamps = len(
        timestamps
    )

    if (
        total_timestamps
        != EXPECTED_UNIQUE_TIMESTAMPS
    ):
        raise ValueError(
            f"Expected "
            f"{EXPECTED_UNIQUE_TIMESTAMPS:,} "
            f"unique target timestamps but found "
            f"{total_timestamps:,}."
        )

    train_end = int(
        total_timestamps
        * TRAIN_FRACTION
    )

    validation_end = (
        train_end
        + int(
            total_timestamps
            * VALIDATION_FRACTION
        )
    )

    train_timestamps = timestamps[
        :train_end
    ]

    validation_timestamps = timestamps[
        train_end:validation_end
    ]

    test_timestamps = timestamps[
        validation_end:
    ]

    return (
        train_timestamps,
        validation_timestamps,
        test_timestamps,
    )


def validate_timestamp_splits(
    train_timestamps: pd.DatetimeIndex,
    validation_timestamps: pd.DatetimeIndex,
    test_timestamps: pd.DatetimeIndex,
) -> None:

    if (
        len(
            train_timestamps
        )
        != EXPECTED_TRAIN_TIMESTAMPS
    ):
        raise ValueError(
            f"Expected "
            f"{EXPECTED_TRAIN_TIMESTAMPS:,} "
            f"training timestamps but found "
            f"{len(train_timestamps):,}."
        )

    if (
        len(
            validation_timestamps
        )
        != EXPECTED_VALIDATION_TIMESTAMPS
    ):
        raise ValueError(
            f"Expected "
            f"{EXPECTED_VALIDATION_TIMESTAMPS:,} "
            f"validation timestamps but found "
            f"{len(validation_timestamps):,}."
        )

    if (
        len(
            test_timestamps
        )
        != EXPECTED_TEST_TIMESTAMPS
    ):
        raise ValueError(
            f"Expected "
            f"{EXPECTED_TEST_TIMESTAMPS:,} "
            f"test timestamps but found "
            f"{len(test_timestamps):,}."
        )

    if (
        train_timestamps.max()
        >= validation_timestamps.min()
    ):
        raise ValueError(
            "Training and validation timestamps "
            "are not strictly chronological."
        )

    if (
        validation_timestamps.max()
        >= test_timestamps.min()
    ):
        raise ValueError(
            "Validation and test timestamps "
            "are not strictly chronological."
        )


def create_splits(
    data: pd.DataFrame,
    train_timestamps: pd.DatetimeIndex,
    validation_timestamps: pd.DatetimeIndex,
    test_timestamps: pd.DatetimeIndex,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    development = data.loc[
        data[
            ROLE_COLUMN
        ]
        == DEVELOPMENT_ROLE
    ].copy()

    held_out = data.loc[
        data[
            ROLE_COLUMN
        ]
        == HELD_OUT_ROLE
    ].copy()

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
            test_timestamps
        )
    ].copy()

    held_out_test = held_out.loc[
        held_out[
            TARGET_TIMESTAMP_COLUMN
        ].isin(
            test_timestamps
        )
    ].copy()

    return (
        train,
        validation,
        temporal_test,
        held_out_test,
    )


def validate_split(
    data: pd.DataFrame,
    expected_rows: int,
    expected_buildings: int,
    expected_timestamps: int,
    split_name: str,
) -> None:

    if len(
        data
    ) != expected_rows:
        raise ValueError(
            f"{split_name} contains "
            f"{len(data):,} rows; expected "
            f"{expected_rows:,}."
        )

    building_count = data[
        BUILDING_ID_COLUMN
    ].nunique()

    if building_count != expected_buildings:
        raise ValueError(
            f"{split_name} contains "
            f"{building_count} buildings; expected "
            f"{expected_buildings}."
        )

    timestamp_count = data[
        TARGET_TIMESTAMP_COLUMN
    ].nunique()

    if timestamp_count != expected_timestamps:
        raise ValueError(
            f"{split_name} contains "
            f"{timestamp_count:,} unique target "
            f"timestamps; expected "
            f"{expected_timestamps:,}."
        )

    duplicate_count = int(
        data.duplicated(
            subset=[
                BUILDING_ID_COLUMN,
                TARGET_TIMESTAMP_COLUMN,
            ]
        )
        .sum()
    )

    if duplicate_count != 0:
        raise ValueError(
            f"{split_name} contains "
            f"{duplicate_count:,} duplicate "
            "building-target-timestamp rows."
        )


def create_summary_row(
    name: str,
    data: pd.DataFrame,
) -> dict[str, object]:

    return {
        "split": name,
        "rows": len(
            data
        ),
        "buildings": data[
            BUILDING_ID_COLUMN
        ].nunique(),
        "target_timestamps": data[
            TARGET_TIMESTAMP_COLUMN
        ].nunique(),
        "first_predictor_timestamp": str(
            data[
                TIMESTAMP_COLUMN
            ].min()
        ),
        "last_predictor_timestamp": str(
            data[
                TIMESTAMP_COLUMN
            ].max()
        ),
        "first_target_timestamp": str(
            data[
                TARGET_TIMESTAMP_COLUMN
            ].min()
        ),
        "last_target_timestamp": str(
            data[
                TARGET_TIMESTAMP_COLUMN
            ].max()
        ),
        "target_mean": float(
            data[
                TARGET_COLUMN
            ].mean()
        ),
        "target_median": float(
            data[
                TARGET_COLUMN
            ].median()
        ),
    }


def main() -> None:

    print(
        "Loading ComStock forecasting features..."
    )

    start = perf_counter()

    data = load_features()

    print(
        f"Rows loaded: "
        f"{len(data):,}"
    )

    print(
        f"Buildings: "
        f"{data[BUILDING_ID_COLUMN].nunique():,}"
    )

    print()

    print(
        "Validating input feature panel..."
    )

    validate_input(
        data
    )

    development = data.loc[
        data[
            ROLE_COLUMN
        ]
        == DEVELOPMENT_ROLE
    ].copy()

    print(
        "Determining chronological split "
        "from target timestamps..."
    )

    (
        train_timestamps,
        validation_timestamps,
        test_timestamps,
    ) = get_split_timestamps(
        development
    )

    validate_timestamp_splits(
        train_timestamps,
        validation_timestamps,
        test_timestamps,
    )

    print(
        f"Training target timestamps: "
        f"{len(train_timestamps):,}"
    )

    print(
        f"Validation target timestamps: "
        f"{len(validation_timestamps):,}"
    )

    print(
        f"Test target timestamps: "
        f"{len(test_timestamps):,}"
    )

    print()

    print(
        "Creating development and held-out splits..."
    )

    (
        train,
        validation,
        temporal_test,
        held_out_test,
    ) = create_splits(
        data,
        train_timestamps,
        validation_timestamps,
        test_timestamps,
    )

    validate_split(
        train,
        EXPECTED_TRAIN_ROWS,
        EXPECTED_DEVELOPMENT_BUILDINGS,
        EXPECTED_TRAIN_TIMESTAMPS,
        "Training split",
    )

    validate_split(
        validation,
        EXPECTED_VALIDATION_ROWS,
        EXPECTED_DEVELOPMENT_BUILDINGS,
        EXPECTED_VALIDATION_TIMESTAMPS,
        "Validation split",
    )

    validate_split(
        temporal_test,
        EXPECTED_TEMPORAL_TEST_ROWS,
        EXPECTED_DEVELOPMENT_BUILDINGS,
        EXPECTED_TEST_TIMESTAMPS,
        "Temporal test split",
    )

    validate_split(
        held_out_test,
        EXPECTED_HELD_OUT_TEST_ROWS,
        EXPECTED_HELD_OUT_BUILDINGS,
        EXPECTED_TEST_TIMESTAMPS,
        "Held-out test split",
    )

    development_ids = set(
        development[
            BUILDING_ID_COLUMN
        ]
        .unique()
        .tolist()
    )

    held_out_ids = set(
        data.loc[
            data[
                ROLE_COLUMN
            ]
            == HELD_OUT_ROLE,
            BUILDING_ID_COLUMN,
        ]
        .unique()
        .tolist()
    )

    overlap = (
        development_ids
        & held_out_ids
    )

    if overlap:
        raise ValueError(
            "Development and held-out building "
            f"sets overlap: {sorted(overlap)}"
        )

    summary = pd.DataFrame(
        [
            create_summary_row(
                "train",
                train,
            ),
            create_summary_row(
                "validation",
                validation,
            ),
            create_summary_row(
                "temporal_test",
                temporal_test,
            ),
            create_summary_row(
                "held_out_test",
                held_out_test,
            ),
        ]
    )

    TRAIN_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Saving ComStock splits..."
    )

    train.to_parquet(
        TRAIN_PATH,
        index=False,
    )

    validation.to_parquet(
        VALIDATION_PATH,
        index=False,
    )

    temporal_test.to_parquet(
        TEMPORAL_TEST_PATH,
        index=False,
    )

    held_out_test.to_parquet(
        HELD_OUT_TEST_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    elapsed_seconds = (
        perf_counter()
        - start
    )

    print()

    print(
        "ComStock chronological split complete."
    )

    print()

    print(
        "Training:"
    )

    print(
        f"  Rows: "
        f"{len(train):,}"
    )

    print(
        f"  Buildings: "
        f"{train[BUILDING_ID_COLUMN].nunique():,}"
    )

    print(
        f"  Target timestamps: "
        f"{train[TARGET_TIMESTAMP_COLUMN].nunique():,}"
    )

    print(
        f"  First target: "
        f"{train[TARGET_TIMESTAMP_COLUMN].min()}"
    )

    print(
        f"  Last target: "
        f"{train[TARGET_TIMESTAMP_COLUMN].max()}"
    )

    print()

    print(
        "Validation:"
    )

    print(
        f"  Rows: "
        f"{len(validation):,}"
    )

    print(
        f"  Buildings: "
        f"{validation[BUILDING_ID_COLUMN].nunique():,}"
    )

    print(
        f"  Target timestamps: "
        f"{validation[TARGET_TIMESTAMP_COLUMN].nunique():,}"
    )

    print(
        f"  First target: "
        f"{validation[TARGET_TIMESTAMP_COLUMN].min()}"
    )

    print(
        f"  Last target: "
        f"{validation[TARGET_TIMESTAMP_COLUMN].max()}"
    )

    print()

    print(
        "Temporal test:"
    )

    print(
        f"  Rows: "
        f"{len(temporal_test):,}"
    )

    print(
        f"  Buildings: "
        f"{temporal_test[BUILDING_ID_COLUMN].nunique():,}"
    )

    print(
        f"  Target timestamps: "
        f"{temporal_test[TARGET_TIMESTAMP_COLUMN].nunique():,}"
    )

    print(
        f"  First target: "
        f"{temporal_test[TARGET_TIMESTAMP_COLUMN].min()}"
    )

    print(
        f"  Last target: "
        f"{temporal_test[TARGET_TIMESTAMP_COLUMN].max()}"
    )

    print()

    print(
        "Held-out test:"
    )

    print(
        f"  Rows: "
        f"{len(held_out_test):,}"
    )

    print(
        f"  Buildings: "
        f"{held_out_test[BUILDING_ID_COLUMN].nunique():,}"
    )

    print(
        f"  Target timestamps: "
        f"{held_out_test[TARGET_TIMESTAMP_COLUMN].nunique():,}"
    )

    print(
        f"  First target: "
        f"{held_out_test[TARGET_TIMESTAMP_COLUMN].min()}"
    )

    print(
        f"  Last target: "
        f"{held_out_test[TARGET_TIMESTAMP_COLUMN].max()}"
    )

    print()

    print(
        "Development / held-out building overlap: "
        f"{len(overlap):,}"
    )

    print(
        f"Elapsed time: "
        f"{elapsed_seconds:,.2f} seconds"
    )

    print()

    print(
        f"Saved training split to: "
        f"{TRAIN_PATH}"
    )

    print(
        f"Saved validation split to: "
        f"{VALIDATION_PATH}"
    )

    print(
        f"Saved temporal test split to: "
        f"{TEMPORAL_TEST_PATH}"
    )

    print(
        f"Saved held-out test split to: "
        f"{HELD_OUT_TEST_PATH}"
    )

    print(
        f"Saved split summary to: "
        f"{SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()