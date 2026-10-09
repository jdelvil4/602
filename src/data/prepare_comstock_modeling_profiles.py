from pathlib import Path
from time import perf_counter

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROFILE_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nrel"
    / "comstock_ca"
    / "upgrade=0"
)

SAMPLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comstock_ca_modeling_sample.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comstock_ca_modeling_hourly.parquet"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comstock_ca_modeling_hourly_summary.csv"
)


BUILDING_ID_COLUMN = "bldg_id"
ROLE_COLUMN = "sample_role"

TIMESTAMP_COLUMN = "timestamp"

SOURCE_TARGET_COLUMN = (
    "out.electricity.total.energy_consumption"
)

OUTPUT_TARGET_COLUMN = (
    "electricity_consumption"
)

EXPECTED_BUILDINGS = 250
EXPECTED_DEVELOPMENT = 200
EXPECTED_HELD_OUT = 50

EXPECTED_QUARTER_HOUR_ROWS = 35040
EXPECTED_HOURLY_ROWS = 8760

EXPECTED_FIRST_TIMESTAMP = pd.Timestamp(
    "2018-01-01 01:00:00"
)

EXPECTED_LAST_TIMESTAMP = pd.Timestamp(
    "2019-01-01 00:00:00"
)


def load_sample() -> pd.DataFrame:

    if not SAMPLE_PATH.exists():

        raise FileNotFoundError(
            f"ComStock modeling sample not found: "
            f"{SAMPLE_PATH}"
        )

    sample = pd.read_csv(
        SAMPLE_PATH
    )

    required_columns = {
        BUILDING_ID_COLUMN,
        ROLE_COLUMN,
    }

    missing_columns = (
        required_columns
        - set(
            sample.columns
        )
    )

    if missing_columns:

        raise ValueError(
            "ComStock modeling sample is missing "
            f"required columns: "
            f"{sorted(missing_columns)}"
        )

    if len(
        sample
    ) != EXPECTED_BUILDINGS:

        raise ValueError(
            f"Expected {EXPECTED_BUILDINGS} sampled "
            f"buildings but found {len(sample)}."
        )

    if sample[
        BUILDING_ID_COLUMN
    ].duplicated().any():

        raise ValueError(
            "ComStock modeling sample contains "
            "duplicate building IDs."
        )

    development_count = int(
        (
            sample[
                ROLE_COLUMN
            ]
            == "development"
        )
        .sum()
    )

    held_out_count = int(
        (
            sample[
                ROLE_COLUMN
            ]
            == "held_out"
        )
        .sum()
    )

    if development_count != EXPECTED_DEVELOPMENT:

        raise ValueError(
            f"Expected {EXPECTED_DEVELOPMENT} "
            f"development buildings but found "
            f"{development_count}."
        )

    if held_out_count != EXPECTED_HELD_OUT:

        raise ValueError(
            f"Expected {EXPECTED_HELD_OUT} held-out "
            f"buildings but found "
            f"{held_out_count}."
        )

    return (
        sample[
            [
                BUILDING_ID_COLUMN,
                ROLE_COLUMN,
            ]
        ]
        .sort_values(
            BUILDING_ID_COLUMN
        )
        .reset_index(
            drop=True
        )
    )


def load_profile(
    bldg_id: int,
) -> pd.DataFrame:

    profile_path = (
        PROFILE_DIRECTORY
        / f"{bldg_id}-0.parquet"
    )

    if not profile_path.exists():

        raise FileNotFoundError(
            f"Profile not found for building "
            f"{bldg_id}: {profile_path}"
        )

    profile = pd.read_parquet(
        profile_path,
        columns=[
            TIMESTAMP_COLUMN,
            SOURCE_TARGET_COLUMN,
        ],
    )

    return profile


def validate_quarter_hour_profile(
    profile: pd.DataFrame,
    bldg_id: int,
) -> pd.DataFrame:

    if len(
        profile
    ) != EXPECTED_QUARTER_HOUR_ROWS:

        raise ValueError(
            f"Building {bldg_id} contains "
            f"{len(profile):,} source rows; expected "
            f"{EXPECTED_QUARTER_HOUR_ROWS:,}."
        )

    if profile[
        TIMESTAMP_COLUMN
    ].isna().any():

        raise ValueError(
            f"Building {bldg_id} contains "
            "missing timestamps."
        )

    if profile[
        SOURCE_TARGET_COLUMN
    ].isna().any():

        missing_count = int(
            profile[
                SOURCE_TARGET_COLUMN
            ]
            .isna()
            .sum()
        )

        raise ValueError(
            f"Building {bldg_id} contains "
            f"{missing_count:,} missing "
            "electricity-consumption values."
        )

    profile = (
        profile
        .sort_values(
            TIMESTAMP_COLUMN
        )
        .reset_index(
            drop=True
        )
    )

    if profile[
        TIMESTAMP_COLUMN
    ].duplicated().any():

        raise ValueError(
            f"Building {bldg_id} contains "
            "duplicate timestamps."
        )

    timestamp_differences = (
        profile[
            TIMESTAMP_COLUMN
        ]
        .diff()
        .dropna()
    )

    expected_interval = pd.Timedelta(
        minutes=15
    )

    invalid_intervals = int(
        (
            timestamp_differences
            != expected_interval
        )
        .sum()
    )

    if invalid_intervals != 0:

        raise ValueError(
            f"Building {bldg_id} contains "
            f"{invalid_intervals:,} non-15-minute "
            "timestamp intervals."
        )

    return profile


def aggregate_to_hourly(
    profile: pd.DataFrame,
    bldg_id: int,
    sample_role: str,
) -> pd.DataFrame:

    working = profile.copy()

    working[
        "hour_timestamp"
    ] = working[
        TIMESTAMP_COLUMN
    ].dt.ceil(
        "h"
    )

    grouped = (
        working
        .groupby(
            "hour_timestamp"
        )[
            SOURCE_TARGET_COLUMN
        ]
        .sum()
    )

    hourly = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: grouped.index.to_numpy(),
            OUTPUT_TARGET_COLUMN: grouped.to_numpy(),
        }
    )

    hourly.insert(
        0,
        BUILDING_ID_COLUMN,
        bldg_id,
    )

    hourly.insert(
        1,
        ROLE_COLUMN,
        sample_role,
    )

    return hourly


def validate_hourly_profile(
    hourly: pd.DataFrame,
    bldg_id: int,
) -> None:

    if len(
        hourly
    ) != EXPECTED_HOURLY_ROWS:

        raise ValueError(
            f"Building {bldg_id} contains "
            f"{len(hourly):,} hourly rows; expected "
            f"{EXPECTED_HOURLY_ROWS:,}."
        )

    first_timestamp = hourly[
        TIMESTAMP_COLUMN
    ].min()

    last_timestamp = hourly[
        TIMESTAMP_COLUMN
    ].max()

    if (
        first_timestamp
        != EXPECTED_FIRST_TIMESTAMP
    ):

        raise ValueError(
            f"Building {bldg_id} has unexpected "
            f"first hourly timestamp: "
            f"{first_timestamp}"
        )

    if (
        last_timestamp
        != EXPECTED_LAST_TIMESTAMP
    ):

        raise ValueError(
            f"Building {bldg_id} has unexpected "
            f"last hourly timestamp: "
            f"{last_timestamp}"
        )

    if hourly[
        TIMESTAMP_COLUMN
    ].duplicated().any():

        raise ValueError(
            f"Building {bldg_id} contains "
            "duplicate hourly timestamps."
        )

    if hourly[
        OUTPUT_TARGET_COLUMN
    ].isna().any():

        raise ValueError(
            f"Building {bldg_id} contains "
            "missing hourly electricity values."
        )

    differences = (
        hourly[
            TIMESTAMP_COLUMN
        ]
        .diff()
        .dropna()
    )

    invalid_intervals = int(
        (
            differences
            != pd.Timedelta(
                hours=1
            )
        )
        .sum()
    )

    if invalid_intervals != 0:

        raise ValueError(
            f"Building {bldg_id} contains "
            f"{invalid_intervals:,} non-hourly "
            "intervals after aggregation."
        )


def create_summary(
    hourly: pd.DataFrame,
) -> pd.DataFrame:

    development_mask = (
        hourly[
            ROLE_COLUMN
        ]
        == "development"
    )

    held_out_mask = (
        hourly[
            ROLE_COLUMN
        ]
        == "held_out"
    )

    summary_rows = [
        {
            "metric": "total_buildings",
            "value": hourly[
                BUILDING_ID_COLUMN
            ].nunique(),
        },
        {
            "metric": "development_buildings",
            "value": hourly.loc[
                development_mask,
                BUILDING_ID_COLUMN,
            ].nunique(),
        },
        {
            "metric": "held_out_buildings",
            "value": hourly.loc[
                held_out_mask,
                BUILDING_ID_COLUMN,
            ].nunique(),
        },
        {
            "metric": "total_rows",
            "value": len(
                hourly
            ),
        },
        {
            "metric": "development_rows",
            "value": int(
                development_mask.sum()
            ),
        },
        {
            "metric": "held_out_rows",
            "value": int(
                held_out_mask.sum()
            ),
        },
        {
            "metric": "rows_per_building",
            "value": int(
                hourly.groupby(
                    BUILDING_ID_COLUMN
                )
                .size()
                .median()
            ),
        },
        {
            "metric": "first_timestamp",
            "value": str(
                hourly[
                    TIMESTAMP_COLUMN
                ].min()
            ),
        },
        {
            "metric": "last_timestamp",
            "value": str(
                hourly[
                    TIMESTAMP_COLUMN
                ].max()
            ),
        },
        {
            "metric": "missing_target_values",
            "value": int(
                hourly[
                    OUTPUT_TARGET_COLUMN
                ]
                .isna()
                .sum()
            ),
        },
        {
            "metric": "duplicate_building_timestamps",
            "value": int(
                hourly.duplicated(
                    subset=[
                        BUILDING_ID_COLUMN,
                        TIMESTAMP_COLUMN,
                    ]
                )
                .sum()
            ),
        },
        {
            "metric": "mean_hourly_consumption",
            "value": hourly[
                OUTPUT_TARGET_COLUMN
            ].mean(),
        },
        {
            "metric": "median_hourly_consumption",
            "value": hourly[
                OUTPUT_TARGET_COLUMN
            ].median(),
        },
        {
            "metric": "minimum_hourly_consumption",
            "value": hourly[
                OUTPUT_TARGET_COLUMN
            ].min(),
        },
        {
            "metric": "maximum_hourly_consumption",
            "value": hourly[
                OUTPUT_TARGET_COLUMN
            ].max(),
        },
    ]

    return pd.DataFrame(
        summary_rows
    )


def main() -> None:

    print(
        "Loading ComStock modeling sample..."
    )

    sample = load_sample()

    print(
        f"Buildings to process: "
        f"{len(sample):,}"
    )

    print()

    print(
        "Preparing hourly ComStock profiles..."
    )

    start = perf_counter()

    hourly_frames = []

    total_buildings = len(
        sample
    )

    for number, row_index in enumerate(
        sample.index,
        start=1,
    ):

        bldg_id = int(
            str(
                sample.at[
                    row_index,
                    BUILDING_ID_COLUMN,
                ]
            )
        )

        sample_role = str(
            sample.at[
                row_index,
                ROLE_COLUMN,
            ]
        )

        print(
            f"[{number:03d}/{total_buildings:03d}] "
            f"Building {bldg_id} "
            f"({sample_role})..."
        )

        profile = load_profile(
            bldg_id
        )

        profile = validate_quarter_hour_profile(
            profile,
            bldg_id,
        )

        hourly = aggregate_to_hourly(
            profile,
            bldg_id,
            sample_role,
        )

        validate_hourly_profile(
            hourly,
            bldg_id,
        )

        hourly_frames.append(
            hourly
        )

    combined = pd.concat(
        hourly_frames,
        ignore_index=True,
    )

    combined = (
        combined
        .sort_values(
            [
                BUILDING_ID_COLUMN,
                TIMESTAMP_COLUMN,
            ]
        )
        .reset_index(
            drop=True
        )
    )

    expected_total_rows = (
        EXPECTED_BUILDINGS
        * EXPECTED_HOURLY_ROWS
    )

    if len(
        combined
    ) != expected_total_rows:

        raise ValueError(
            f"Combined dataset contains "
            f"{len(combined):,} rows; expected "
            f"{expected_total_rows:,}."
        )

    duplicate_count = int(
        combined.duplicated(
            subset=[
                BUILDING_ID_COLUMN,
                TIMESTAMP_COLUMN,
            ]
        )
        .sum()
    )

    if duplicate_count != 0:

        raise ValueError(
            f"Combined dataset contains "
            f"{duplicate_count:,} duplicate "
            "building-timestamp rows."
        )

    summary = create_summary(
        combined
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()

    print(
        "Saving hourly ComStock dataset..."
    )

    combined.to_parquet(
        OUTPUT_PATH,
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

    development_rows = int(
        (
            combined[
                ROLE_COLUMN
            ]
            == "development"
        )
        .sum()
    )

    held_out_rows = int(
        (
            combined[
                ROLE_COLUMN
            ]
            == "held_out"
        )
        .sum()
    )

    print()

    print(
        "ComStock hourly preparation complete."
    )

    print()

    print(
        f"Total buildings: "
        f"{combined[BUILDING_ID_COLUMN].nunique():,}"
    )

    print(
        f"Total rows: "
        f"{len(combined):,}"
    )

    print(
        f"Development rows: "
        f"{development_rows:,}"
    )

    print(
        f"Held-out rows: "
        f"{held_out_rows:,}"
    )

    print(
        f"Rows per building: "
        f"{EXPECTED_HOURLY_ROWS:,}"
    )

    print(
        f"First timestamp: "
        f"{combined[TIMESTAMP_COLUMN].min()}"
    )

    print(
        f"Last timestamp: "
        f"{combined[TIMESTAMP_COLUMN].max()}"
    )

    print(
        f"Missing target values: "
        f"{combined[OUTPUT_TARGET_COLUMN].isna().sum():,}"
    )

    print(
        f"Duplicate building timestamps: "
        f"{duplicate_count:,}"
    )

    print()

    print(
        "Hourly electricity-consumption summary:"
    )

    print(
        combined[
            OUTPUT_TARGET_COLUMN
        ]
        .describe()
        .to_string()
    )

    print()

    print(
        f"Elapsed time: "
        f"{elapsed_seconds:,.2f} seconds"
    )

    print()

    print(
        f"Saved hourly data to: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Saved summary to: "
        f"{SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()