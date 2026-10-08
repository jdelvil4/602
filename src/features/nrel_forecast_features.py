from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_modeling_weather.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_forecast_features.parquet"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_forecast_features_summary.csv"
)


BUILDING_ID_COLUMN = "bldg_id"
ROLE_COLUMN = "sample_role"
TIMESTAMP_COLUMN = "timestamp"
TARGET_SOURCE_COLUMN = "electricity_total_hourly"
TARGET_COLUMN = "target_1h_ahead"

TEMPERATURE_COLUMN = "weather_dry_bulb_temperature_c"

EXPECTED_BUILDINGS = 250
EXPECTED_ROWS_INPUT = 2_190_000
EXPECTED_HOURS_PER_BUILDING = 8760

HEATING_BASE_C = 18.0
COOLING_BASE_C = 22.0


def load_input() -> pd.DataFrame:

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Weather-enriched NREL dataset not found: "
            f"{INPUT_PATH}"
        )

    df = pd.read_parquet(
        INPUT_PATH
    )

    required_columns = {
        BUILDING_ID_COLUMN,
        ROLE_COLUMN,
        TIMESTAMP_COLUMN,
        TARGET_SOURCE_COLUMN,
        TEMPERATURE_COLUMN,
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Input dataset is missing required columns: "
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
            "Input dataset contains invalid building IDs."
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

    if len(df) != EXPECTED_ROWS_INPUT:
        raise ValueError(
            f"Input contains {len(df):,} rows; "
            f"expected {EXPECTED_ROWS_INPUT:,}."
        )

    building_count = df[
        BUILDING_ID_COLUMN
    ].nunique()

    if building_count != EXPECTED_BUILDINGS:
        raise ValueError(
            f"Input contains {building_count} buildings; "
            f"expected {EXPECTED_BUILDINGS}."
        )

    duplicate_count = int(
        df.duplicated(
            subset=[
                BUILDING_ID_COLUMN,
                TIMESTAMP_COLUMN,
            ]
        )
        .sum()
    )

    if duplicate_count != 0:
        raise ValueError(
            f"Input contains {duplicate_count} "
            "duplicate building-timestamp rows."
        )

    return (
        df
        .sort_values(
            [
                BUILDING_ID_COLUMN,
                TIMESTAMP_COLUMN,
            ]
        )
        .reset_index(drop=True)
    )


def add_calendar_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    working = df.copy()

    timestamp = working[
        TIMESTAMP_COLUMN
    ]

    working["hour"] = (
        timestamp.dt.hour
    )

    working["day_of_week"] = (
        timestamp.dt.dayofweek
    )

    working["month"] = (
        timestamp.dt.month
    )

    working["day_of_year"] = (
        timestamp.dt.dayofyear
    )

    working["is_weekend"] = (
        working[
            "day_of_week"
        ]
        .isin(
            [
                5,
                6,
            ]
        )
        .astype("int8")
    )

    working["hour_sin"] = np.sin(
        2
        * np.pi
        * working["hour"]
        / 24
    )

    working["hour_cos"] = np.cos(
        2
        * np.pi
        * working["hour"]
        / 24
    )

    working["day_of_week_sin"] = np.sin(
        2
        * np.pi
        * working["day_of_week"]
        / 7
    )

    working["day_of_week_cos"] = np.cos(
        2
        * np.pi
        * working["day_of_week"]
        / 7
    )

    working["month_sin"] = np.sin(
        2
        * np.pi
        * (
            working["month"]
            - 1
        )
        / 12
    )

    working["month_cos"] = np.cos(
        2
        * np.pi
        * (
            working["month"]
            - 1
        )
        / 12
    )

    working["day_of_year_sin"] = np.sin(
        2
        * np.pi
        * (
            working["day_of_year"]
            - 1
        )
        / 365
    )

    working["day_of_year_cos"] = np.cos(
        2
        * np.pi
        * (
            working["day_of_year"]
            - 1
        )
        / 365
    )

    return working


def add_weather_derived_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    working = df.copy()

    temperature = working[
        TEMPERATURE_COLUMN
    ]

    working[
        "heating_degree_c"
    ] = np.maximum(
        HEATING_BASE_C
        - temperature,
        0.0,
    )

    working[
        "cooling_degree_c"
    ] = np.maximum(
        temperature
        - COOLING_BASE_C,
        0.0,
    )

    return working


def add_building_history_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    working = df.copy()

    grouped = working.groupby(
        BUILDING_ID_COLUMN,
        sort=False,
    )[
        TARGET_SOURCE_COLUMN
    ]

    working[
        "electricity_lag_1h"
    ] = grouped.shift(
        1
    )

    working[
        "electricity_lag_24h"
    ] = grouped.shift(
        24
    )

    working[
        "seasonal_naive_24h"
    ] = grouped.shift(
        23
    )

    shifted_consumption = grouped.shift(
        1
    )

    working[
        "electricity_rolling_mean_24h"
    ] = (
        shifted_consumption
        .groupby(
            working[
                BUILDING_ID_COLUMN
            ]
        )
        .rolling(
            window=24,
            min_periods=24,
        )
        .mean()
        .reset_index(
            level=0,
            drop=True,
        )
    )

    working[
        "electricity_rolling_mean_168h"
    ] = (
        shifted_consumption
        .groupby(
            working[
                BUILDING_ID_COLUMN
            ]
        )
        .rolling(
            window=168,
            min_periods=168,
        )
        .mean()
        .reset_index(
            level=0,
            drop=True,
        )
    )

    working[
        TARGET_COLUMN
    ] = grouped.shift(
        -1
    )

    return working


def validate_history_features(
    df: pd.DataFrame,
) -> None:

    first_rows = (
        df.groupby(
            BUILDING_ID_COLUMN,
            sort=False,
        )
        .head(
            168
        )
    )

    if (
        first_rows[
            "electricity_rolling_mean_168h"
        ]
        .notna()
        .any()
    ):
        raise ValueError(
            "168-hour rolling features appear before "
            "sufficient building history is available."
        )

    final_rows = (
        df.groupby(
            BUILDING_ID_COLUMN,
            sort=False,
        )
        .tail(
            1
        )
    )

    if (
        final_rows[
            TARGET_COLUMN
        ]
        .notna()
        .any()
    ):
        raise ValueError(
            "Final row for one or more buildings "
            "contains a future target unexpectedly."
        )


def remove_incomplete_forecast_rows(
    df: pd.DataFrame,
) -> pd.DataFrame:

    required_forecast_columns = [
        "electricity_lag_1h",
        "electricity_lag_24h",
        "seasonal_naive_24h",
        "electricity_rolling_mean_24h",
        "electricity_rolling_mean_168h",
        TARGET_COLUMN,
    ]

    working = (
        df
        .dropna(
            subset=required_forecast_columns
        )
        .copy()
        .reset_index(drop=True)
    )

    return working


def validate_output(
    df: pd.DataFrame,
) -> None:

    building_count = df[
        BUILDING_ID_COLUMN
    ].nunique()

    if building_count != EXPECTED_BUILDINGS:
        raise ValueError(
            f"Feature dataset contains "
            f"{building_count} buildings; "
            f"expected {EXPECTED_BUILDINGS}."
        )

    duplicate_count = int(
        df.duplicated(
            subset=[
                BUILDING_ID_COLUMN,
                TIMESTAMP_COLUMN,
            ]
        )
        .sum()
    )

    if duplicate_count != 0:
        raise ValueError(
            f"Feature dataset contains "
            f"{duplicate_count} duplicate "
            "building-timestamp rows."
        )

    required_feature_columns = [
        "electricity_lag_1h",
        "electricity_lag_24h",
        "seasonal_naive_24h",
        "electricity_rolling_mean_24h",
        "electricity_rolling_mean_168h",
        "heating_degree_c",
        "cooling_degree_c",
        TARGET_COLUMN,
    ]

    missing_count = int(
        df[
            required_feature_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    if missing_count != 0:
        raise ValueError(
            f"Feature dataset contains "
            f"{missing_count} missing values "
            "in required forecasting features."
        )

    rows_per_building = (
        df.groupby(
            BUILDING_ID_COLUMN
        )
        .size()
    )

    if rows_per_building.nunique() != 1:
        raise ValueError(
            "Buildings do not contain equal numbers "
            "of usable forecasting rows."
        )


def create_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows_per_building = (
        df.groupby(
            BUILDING_ID_COLUMN
        )
        .size()
    )

    summary = pd.DataFrame(
        [
            {
                "buildings_total": df[
                    BUILDING_ID_COLUMN
                ].nunique(),
                "development_buildings": df.loc[
                    df[
                        ROLE_COLUMN
                    ]
                    == "development",
                    BUILDING_ID_COLUMN,
                ].nunique(),
                "held_out_buildings": df.loc[
                    df[
                        ROLE_COLUMN
                    ]
                    == "held_out",
                    BUILDING_ID_COLUMN,
                ].nunique(),
                "rows_total": len(
                    df
                ),
                "rows_per_building_min": int(
                    rows_per_building.min()
                ),
                "rows_per_building_max": int(
                    rows_per_building.max()
                ),
                "minimum_timestamp": df[
                    TIMESTAMP_COLUMN
                ].min(),
                "maximum_timestamp": df[
                    TIMESTAMP_COLUMN
                ].max(),
                "minimum_target_timestamp": (
                    df[
                        TIMESTAMP_COLUMN
                    ].min()
                    + pd.Timedelta(
                        hours=1
                    )
                ),
                "maximum_target_timestamp": (
                    df[
                        TIMESTAMP_COLUMN
                    ].max()
                    + pd.Timedelta(
                        hours=1
                    )
                ),
                "missing_required_features": int(
                    df[
                        [
                            "electricity_lag_1h",
                            "electricity_lag_24h",
                            "electricity_rolling_mean_24h",
                            "electricity_rolling_mean_168h",
                            "heating_degree_c",
                            "cooling_degree_c",
                            TARGET_COLUMN,
                        ]
                    ]
                    .isna()
                    .sum()
                    .sum()
                ),
                "duplicate_building_timestamps": int(
                    df.duplicated(
                        subset=[
                            BUILDING_ID_COLUMN,
                            TIMESTAMP_COLUMN,
                        ]
                    )
                    .sum()
                ),
            }
        ]
    )

    return summary


def main() -> None:

    print(
        "Loading NREL weather-enriched modeling dataset..."
    )

    df = load_input()

    print(
        f"Input rows: {len(df):,}"
    )

    print(
        f"Buildings: "
        f"{df[BUILDING_ID_COLUMN].nunique():,}"
    )

    print()

    print(
        "Adding calendar features..."
    )

    df = add_calendar_features(
        df
    )

    print(
        "Adding weather-derived features..."
    )

    df = add_weather_derived_features(
        df
    )

    print(
        "Adding per-building lag, rolling, "
        "seasonal-naive, and target features..."
    )

    df = add_building_history_features(
        df
    )

    print(
        "Validating per-building history construction..."
    )

    validate_history_features(
        df
    )

    rows_before_drop = len(
        df
    )

    print(
        "Removing rows without sufficient historical "
        "context or a future target..."
    )

    df = remove_incomplete_forecast_rows(
        df
    )

    rows_removed = (
        rows_before_drop
        - len(df)
    )

    print(
        f"Rows removed: {rows_removed:,}"
    )

    print(
        f"Usable forecasting rows: {len(df):,}"
    )

    print()

    print(
        "Validating final forecasting feature dataset..."
    )

    validate_output(
        df
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Saving NREL forecasting feature dataset..."
    )

    df.to_parquet(
        OUTPUT_PATH,
        index=False,
        compression="snappy",
    )

    summary = create_summary(
        df
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print()

    print(
        "NREL forecasting feature construction complete."
    )

    print()

    print(
        summary.to_string(
            index=False
        )
    )

    print()

    print(
        f"Saved feature dataset to: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Saved summary to: "
        f"{SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()