from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_forecast_features.parquet"
)

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

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_split_summary.csv"
)


BUILDING_ID_COLUMN = "bldg_id"
ROLE_COLUMN = "sample_role"
TIMESTAMP_COLUMN = "timestamp"
TARGET_COLUMN = "target_1h_ahead"
TARGET_TIMESTAMP_COLUMN = "target_timestamp"

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15

EXPECTED_BUILDINGS = 250
EXPECTED_DEVELOPMENT_BUILDINGS = 200
EXPECTED_HELD_OUT_BUILDINGS = 50
EXPECTED_ROWS_PER_BUILDING = 8591
EXPECTED_UNIQUE_TIMESTAMPS = 8591


def load_features() -> pd.DataFrame:

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"NREL forecasting feature dataset not found: "
            f"{INPUT_PATH}"
        )

    df = pd.read_parquet(
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
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Forecast feature dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df[TIMESTAMP_COLUMN] = pd.to_datetime(
        df[TIMESTAMP_COLUMN]
    )

    df[BUILDING_ID_COLUMN] = pd.to_numeric(
        df[BUILDING_ID_COLUMN],
        errors="coerce",
    )

    if df[BUILDING_ID_COLUMN].isna().any():
        raise ValueError(
            "Forecast feature dataset contains invalid building IDs."
        )

    df[BUILDING_ID_COLUMN] = (
        df[BUILDING_ID_COLUMN]
        .astype("int64")
    )

    df[ROLE_COLUMN] = (
        df[ROLE_COLUMN]
        .astype(str)
        .str.strip()
    )

    df[TARGET_TIMESTAMP_COLUMN] = (
        df[TIMESTAMP_COLUMN]
        + pd.Timedelta(
            hours=1
        )
    )

    building_count = df[
        BUILDING_ID_COLUMN
    ].nunique()

    if building_count != EXPECTED_BUILDINGS:
        raise ValueError(
            f"Feature dataset contains {building_count} buildings; "
            f"expected {EXPECTED_BUILDINGS}."
        )

    development_buildings = df.loc[
        df[ROLE_COLUMN] == "development",
        BUILDING_ID_COLUMN,
    ].nunique()

    held_out_buildings = df.loc[
        df[ROLE_COLUMN] == "held_out",
        BUILDING_ID_COLUMN,
    ].nunique()

    if development_buildings != EXPECTED_DEVELOPMENT_BUILDINGS:
        raise ValueError(
            f"Found {development_buildings} development buildings; "
            f"expected {EXPECTED_DEVELOPMENT_BUILDINGS}."
        )

    if held_out_buildings != EXPECTED_HELD_OUT_BUILDINGS:
        raise ValueError(
            f"Found {held_out_buildings} held-out buildings; "
            f"expected {EXPECTED_HELD_OUT_BUILDINGS}."
        )

    rows_per_building = (
        df.groupby(
            BUILDING_ID_COLUMN
        )
        .size()
    )

    if not (
        rows_per_building
        == EXPECTED_ROWS_PER_BUILDING
    ).all():
        raise ValueError(
            f"One or more buildings do not contain exactly "
            f"{EXPECTED_ROWS_PER_BUILDING:,} usable rows."
        )

    return (
        df
        .sort_values(
            [
                TARGET_TIMESTAMP_COLUMN,
                BUILDING_ID_COLUMN,
            ]
        )
        .reset_index(drop=True)
    )


def determine_temporal_boundaries(
    development: pd.DataFrame,
) -> tuple[
    pd.DatetimeIndex,
    pd.DatetimeIndex,
    pd.DatetimeIndex,
]:

    unique_target_timestamps = pd.DatetimeIndex(
        development[
            TARGET_TIMESTAMP_COLUMN
        ]
        .drop_duplicates()
        .sort_values()
    )

    timestamp_count = len(
        unique_target_timestamps
    )

    if timestamp_count != EXPECTED_UNIQUE_TIMESTAMPS:
        raise ValueError(
            f"Development data contains "
            f"{timestamp_count:,} unique target timestamps; "
            f"expected {EXPECTED_UNIQUE_TIMESTAMPS:,}."
        )

    train_count = int(
        timestamp_count
        * TRAIN_FRACTION
    )

    validation_count = int(
        timestamp_count
        * VALIDATION_FRACTION
    )

    test_count = (
        timestamp_count
        - train_count
        - validation_count
    )

    train_timestamps = (
        unique_target_timestamps[
            :train_count
        ]
    )

    validation_timestamps = (
        unique_target_timestamps[
            train_count:
            train_count + validation_count
        ]
    )

    test_timestamps = (
        unique_target_timestamps[
            train_count + validation_count:
        ]
    )

    if len(test_timestamps) != test_count:
        raise ValueError(
            "Temporal test timestamp allocation failed."
        )

    return (
        train_timestamps,
        validation_timestamps,
        test_timestamps,
    )


def create_splits(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    development = (
        df.loc[
            df[ROLE_COLUMN]
            == "development"
        ]
        .copy()
    )

    held_out = (
        df.loc[
            df[ROLE_COLUMN]
            == "held_out"
        ]
        .copy()
    )

    (
        train_timestamps,
        validation_timestamps,
        test_timestamps,
    ) = determine_temporal_boundaries(
        development
    )

    train = (
        development.loc[
            development[
                TARGET_TIMESTAMP_COLUMN
            ].isin(
                train_timestamps
            )
        ]
        .copy()
    )

    validation = (
        development.loc[
            development[
                TARGET_TIMESTAMP_COLUMN
            ].isin(
                validation_timestamps
            )
        ]
        .copy()
    )

    temporal_test = (
        development.loc[
            development[
                TARGET_TIMESTAMP_COLUMN
            ].isin(
                test_timestamps
            )
        ]
        .copy()
    )

    held_out_test = (
        held_out.loc[
            held_out[
                TARGET_TIMESTAMP_COLUMN
            ].isin(
                test_timestamps
            )
        ]
        .copy()
    )

    return (
        train,
        validation,
        temporal_test,
        held_out_test,
    )


def validate_split(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    temporal_test: pd.DataFrame,
    held_out_test: pd.DataFrame,
) -> None:

    development_sets = {
        "train": train,
        "validation": validation,
        "temporal_test": temporal_test,
    }

    for name, frame in development_sets.items():

        building_count = frame[
            BUILDING_ID_COLUMN
        ].nunique()

        if building_count != EXPECTED_DEVELOPMENT_BUILDINGS:
            raise ValueError(
                f"{name} contains {building_count} buildings; "
                f"expected {EXPECTED_DEVELOPMENT_BUILDINGS}."
            )

        roles = set(
            frame[
                ROLE_COLUMN
            ].unique()
        )

        if roles != {"development"}:
            raise ValueError(
                f"{name} contains unexpected sample roles: "
                f"{sorted(roles)}"
            )

    held_out_building_count = (
        held_out_test[
            BUILDING_ID_COLUMN
        ]
        .nunique()
    )

    if (
        held_out_building_count
        != EXPECTED_HELD_OUT_BUILDINGS
    ):
        raise ValueError(
            f"Held-out test contains "
            f"{held_out_building_count} buildings; "
            f"expected {EXPECTED_HELD_OUT_BUILDINGS}."
        )

    held_out_roles = set(
        held_out_test[
            ROLE_COLUMN
        ].unique()
    )

    if held_out_roles != {"held_out"}:
        raise ValueError(
            "Held-out test contains unexpected sample roles."
        )

    train_times = set(
        train[
            TARGET_TIMESTAMP_COLUMN
        ].unique()
    )

    validation_times = set(
        validation[
            TARGET_TIMESTAMP_COLUMN
        ].unique()
    )

    test_times = set(
        temporal_test[
            TARGET_TIMESTAMP_COLUMN
        ].unique()
    )

    held_out_test_times = set(
        held_out_test[
            TARGET_TIMESTAMP_COLUMN
        ].unique()
    )

    if train_times & validation_times:
        raise ValueError(
            "Train and validation target timestamps overlap."
        )

    if train_times & test_times:
        raise ValueError(
            "Train and temporal-test target timestamps overlap."
        )

    if validation_times & test_times:
        raise ValueError(
            "Validation and temporal-test target timestamps overlap."
        )

    if test_times != held_out_test_times:
        raise ValueError(
            "Held-out test does not use exactly the same "
            "target timestamps as temporal test."
        )

    if (
        train[
            TARGET_TIMESTAMP_COLUMN
        ].max()
        >= validation[
            TARGET_TIMESTAMP_COLUMN
        ].min()
    ):
        raise ValueError(
            "Train period is not strictly before validation."
        )

    if (
        validation[
            TARGET_TIMESTAMP_COLUMN
        ].max()
        >= temporal_test[
            TARGET_TIMESTAMP_COLUMN
        ].min()
    ):
        raise ValueError(
            "Validation period is not strictly before temporal test."
        )

    train_buildings = set(
        train[
            BUILDING_ID_COLUMN
        ].unique()
    )

    held_out_buildings = set(
        held_out_test[
            BUILDING_ID_COLUMN
        ].unique()
    )

    if train_buildings & held_out_buildings:
        raise ValueError(
            "Held-out buildings appear in training data."
        )

    for name, frame in {
        "train": train,
        "validation": validation,
        "temporal_test": temporal_test,
        "held_out_test": held_out_test,
    }.items():

        duplicate_count = int(
            frame.duplicated(
                subset=[
                    BUILDING_ID_COLUMN,
                    TIMESTAMP_COLUMN,
                ]
            )
            .sum()
        )

        if duplicate_count != 0:
            raise ValueError(
                f"{name} contains {duplicate_count} "
                "duplicate building-timestamp rows."
            )


def summarize_split(
    name: str,
    df: pd.DataFrame,
) -> dict:

    unique_target_timestamps = (
        df[
            TARGET_TIMESTAMP_COLUMN
        ]
        .nunique()
    )

    return {
        "split": name,
        "rows": len(
            df
        ),
        "buildings": df[
            BUILDING_ID_COLUMN
        ].nunique(),
        "unique_target_timestamps": (
            unique_target_timestamps
        ),
        "predictor_start": df[
            TIMESTAMP_COLUMN
        ].min(),
        "predictor_end": df[
            TIMESTAMP_COLUMN
        ].max(),
        "target_start": df[
            TARGET_TIMESTAMP_COLUMN
        ].min(),
        "target_end": df[
            TARGET_TIMESTAMP_COLUMN
        ].max(),
        "target_mean": df[
            TARGET_COLUMN
        ].mean(),
        "target_median": df[
            TARGET_COLUMN
        ].median(),
        "target_minimum": df[
            TARGET_COLUMN
        ].min(),
        "target_maximum": df[
            TARGET_COLUMN
        ].max(),
    }


def main() -> None:

    print(
        "Loading NREL forecasting feature dataset..."
    )

    df = load_features()

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        f"Buildings: "
        f"{df[BUILDING_ID_COLUMN].nunique():,}"
    )

    print(
        f"Unique target timestamps: "
        f"{df[TARGET_TIMESTAMP_COLUMN].nunique():,}"
    )

    print()

    print(
        "Creating chronological development split "
        "using unique target timestamps..."
    )

    (
        train,
        validation,
        temporal_test,
        held_out_test,
    ) = create_splits(
        df
    )

    print(
        "Validating split integrity..."
    )

    validate_split(
        train,
        validation,
        temporal_test,
        held_out_test,
    )

    output_frames = {
        TRAIN_PATH: train,
        VALIDATION_PATH: validation,
        TEMPORAL_TEST_PATH: temporal_test,
        HELD_OUT_TEST_PATH: held_out_test,
    }

    print()

    print(
        "Saving split datasets..."
    )

    for path, frame in output_frames.items():

        frame.to_parquet(
            path,
            index=False,
            compression="snappy",
        )

        print(
            f"  {path.name}: "
            f"{len(frame):,} rows"
        )

    summary = pd.DataFrame(
        [
            summarize_split(
                "train",
                train,
            ),
            summarize_split(
                "validation",
                validation,
            ),
            summarize_split(
                "temporal_test",
                temporal_test,
            ),
            summarize_split(
                "held_out_test",
                held_out_test,
            ),
        ]
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print()

    print(
        "NREL chronological split complete."
    )

    print()

    print(
        summary.to_string(
            index=False
        )
    )

    print()

    print(
        f"Saved split summary to: "
        f"{SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()