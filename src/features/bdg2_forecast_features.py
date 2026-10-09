from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bdg2_modeling_weather.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bdg2_forecast_features.parquet"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bdg2_forecast_features_summary.csv"
)

BUILDING_SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bdg2_forecast_feature_building_summary.csv"
)

HEATING_BASE_C = 18.0
COOLING_BASE_C = 22.0
WEATHER_FILL_LIMIT_HOURS = 3


def validate_input(
    data: pd.DataFrame,
) -> None:

    required = [
        "building_id",
        "timestamp",
        "electricity",
        "sample_group",
        "site_id",
        "primaryspaceusage",
        "sqft",
        "airTemperature",
        "windSpeed",
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    duplicates = int(
        data.duplicated(
            subset=[
                "building_id",
                "timestamp",
            ]
        ).sum()
    )

    if duplicates > 0:
        raise ValueError(
            f"Duplicate building-hour rows: {duplicates:,}"
        )


def prepare_weather(
    data: pd.DataFrame,
) -> pd.DataFrame:

    weather_columns = [
        "airTemperature",
        "windSpeed",
    ]

    weather = (
        data[
            [
                "site_id",
                "timestamp",
                *weather_columns,
            ]
        ]
        .drop_duplicates(
            subset=[
                "site_id",
                "timestamp",
            ]
        )
        .sort_values(
            [
                "site_id",
                "timestamp",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    duplicate_site_hours = int(
        weather.duplicated(
            subset=[
                "site_id",
                "timestamp",
            ]
        ).sum()
    )

    if duplicate_site_hours > 0:
        raise ValueError(
            f"Duplicate site-hour weather rows: "
            f"{duplicate_site_hours:,}"
        )

    for column in weather_columns:

        weather[column] = (
            weather
            .groupby(
                "site_id",
                observed=True,
            )[column]
            .ffill(
                limit=WEATHER_FILL_LIMIT_HOURS
            )
        )

    data = data.drop(
        columns=weather_columns
    )

    data = data.merge(
        weather,
        on=[
            "site_id",
            "timestamp",
        ],
        how="left",
        validate="many_to_one",
    )

    return data


def create_calendar_features(
    data: pd.DataFrame,
) -> pd.DataFrame:

    data["hour"] = (
        data["timestamp"].dt.hour
    )

    data["day_of_week"] = (
        data["timestamp"].dt.dayofweek
    )

    data["month"] = (
        data["timestamp"].dt.month
    )

    data["day_of_year"] = (
        data["timestamp"].dt.dayofyear
    )

    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype("int8")

    data["hour_sin"] = np.sin(
        2.0
        * np.pi
        * data["hour"]
        / 24.0
    )

    data["hour_cos"] = np.cos(
        2.0
        * np.pi
        * data["hour"]
        / 24.0
    )

    data["dow_sin"] = np.sin(
        2.0
        * np.pi
        * data["day_of_week"]
        / 7.0
    )

    data["dow_cos"] = np.cos(
        2.0
        * np.pi
        * data["day_of_week"]
        / 7.0
    )

    data["month_sin"] = np.sin(
        2.0
        * np.pi
        * (
            data["month"] - 1
        )
        / 12.0
    )

    data["month_cos"] = np.cos(
        2.0
        * np.pi
        * (
            data["month"] - 1
        )
        / 12.0
    )

    return data


def create_temperature_features(
    data: pd.DataFrame,
) -> pd.DataFrame:

    data[
        "heating_degree"
    ] = np.maximum(
        HEATING_BASE_C
        - data[
            "airTemperature"
        ],
        0.0,
    )

    data[
        "cooling_degree"
    ] = np.maximum(
        data[
            "airTemperature"
        ]
        - COOLING_BASE_C,
        0.0,
    )

    return data


def create_load_features(
    data: pd.DataFrame,
) -> pd.DataFrame:

    data = data.sort_values(
        [
            "building_id",
            "timestamp",
        ]
    ).copy()

    grouped = data.groupby(
        "building_id",
        observed=True,
        sort=False,
    )["electricity"]

    data[
        "electricity_lag_1"
    ] = grouped.shift(
        1
    )

    data[
        "electricity_lag_24"
    ] = grouped.shift(
        24
    )

    data[
        "seasonal_naive_24h"
    ] = grouped.shift(
        23
    )

    shifted = grouped.shift(
        1
    )

    data[
        "electricity_rolling_24h_mean"
    ] = (
        shifted
        .groupby(
            data[
                "building_id"
            ],
            observed=True,
            sort=False,
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

    data[
        "electricity_rolling_168h_mean"
    ] = (
        shifted
        .groupby(
            data[
                "building_id"
            ],
            observed=True,
            sort=False,
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

    data[
        "target_electricity_next_hour"
    ] = grouped.shift(
        -1
    )

    data[
        "target_timestamp"
    ] = (
        data[
            "timestamp"
        ]
        + pd.Timedelta(
            hours=1
        )
    )

    return data


def main() -> None:

    print(
        "Loading BDG2 modeling/weather panel..."
    )

    data = pd.read_parquet(
        INPUT_PATH
    )

    data[
        "timestamp"
    ] = pd.to_datetime(
        data[
            "timestamp"
        ],
        errors="raise",
    )

    validate_input(
        data
    )

    original_rows = len(
        data
    )

    original_missing_electricity = int(
        data[
            "electricity"
        ]
        .isna()
        .sum()
    )

    print(
        f"Input rows: "
        f"{original_rows:,}"
    )

    print(
        f"Buildings: "
        f"{data['building_id'].nunique():,}"
    )

    print(
        f"Missing electricity rows: "
        f"{original_missing_electricity:,}"
    )

    print()
    print(
        "Forward-filling short weather gaps "
        "within each site..."
    )

    before_air_missing = int(
        data[
            "airTemperature"
        ]
        .isna()
        .sum()
    )

    before_wind_missing = int(
        data[
            "windSpeed"
        ]
        .isna()
        .sum()
    )

    data = prepare_weather(
        data
    )

    after_air_missing = int(
        data[
            "airTemperature"
        ]
        .isna()
        .sum()
    )

    after_wind_missing = int(
        data[
            "windSpeed"
        ]
        .isna()
        .sum()
    )

    print(
        f"airTemperature missing: "
        f"{before_air_missing:,} -> "
        f"{after_air_missing:,}"
    )

    print(
        f"windSpeed missing: "
        f"{before_wind_missing:,} -> "
        f"{after_wind_missing:,}"
    )

    data = create_calendar_features(
        data
    )

    data = create_temperature_features(
        data
    )

    data = create_load_features(
        data
    )

    required_features = [
        "electricity_lag_1",
        "electricity_lag_24",
        "seasonal_naive_24h",
        "electricity_rolling_24h_mean",
        "electricity_rolling_168h_mean",
        "target_electricity_next_hour",
        "airTemperature",
        "windSpeed",
        "heating_degree",
        "cooling_degree",
    ]

    missing_by_feature = (
        data[
            required_features
        ]
        .isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    print()
    print(
        "Missing values before final "
        "forecast-row filtering:"
    )

    print(
        missing_by_feature.to_string()
    )

    usable = data.dropna(
        subset=required_features
    ).copy()

    usable = usable.sort_values(
        [
            "building_id",
            "timestamp",
        ]
    ).reset_index(
        drop=True
    )

    removed_rows = (
        original_rows
        - len(
            usable
        )
    )

    duplicates = int(
        usable.duplicated(
            subset=[
                "building_id",
                "timestamp",
            ]
        ).sum()
    )

    if duplicates > 0:
        raise ValueError(
            f"Duplicate usable rows: "
            f"{duplicates:,}"
        )

    building_summary = (
        usable
        .groupby(
            "building_id",
            observed=True,
        )
        .agg(
            sample_group=(
                "sample_group",
                "first",
            ),
            primaryspaceusage=(
                "primaryspaceusage",
                "first",
            ),
            site_id=(
                "site_id",
                "first",
            ),
            usable_rows=(
                "timestamp",
                "size",
            ),
            first_predictor_timestamp=(
                "timestamp",
                "min",
            ),
            last_predictor_timestamp=(
                "timestamp",
                "max",
            ),
            first_target_timestamp=(
                "target_timestamp",
                "min",
            ),
            last_target_timestamp=(
                "target_timestamp",
                "max",
            ),
        )
        .reset_index()
    )

    summary = pd.DataFrame(
        [
            {
                "input_rows":
                    original_rows,
                "usable_rows":
                    len(
                        usable
                    ),
                "removed_rows":
                    removed_rows,
                "buildings":
                    usable[
                        "building_id"
                    ].nunique(),
                "development_buildings":
                    usable.loc[
                        usable[
                            "sample_group"
                        ]
                        == "development",
                        "building_id",
                    ].nunique(),
                "held_out_buildings":
                    usable.loc[
                        usable[
                            "sample_group"
                        ]
                        == "held_out",
                        "building_id",
                    ].nunique(),
                "input_missing_electricity":
                    original_missing_electricity,
                "remaining_missing_required_features":
                    int(
                        usable[
                            required_features
                        ]
                        .isna()
                        .sum()
                        .sum()
                    ),
                "duplicate_building_timestamps":
                    duplicates,
                "minimum_predictor_timestamp":
                    usable[
                        "timestamp"
                    ].min(),
                "maximum_predictor_timestamp":
                    usable[
                        "timestamp"
                    ].max(),
                "minimum_target_timestamp":
                    usable[
                        "target_timestamp"
                    ].min(),
                "maximum_target_timestamp":
                    usable[
                        "target_timestamp"
                    ].max(),
            }
        ]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    usable.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    building_summary.to_csv(
        BUILDING_SUMMARY_PATH,
        index=False,
    )

    print()
    print(
        "BDG2 forecast feature engineering complete."
    )

    print()

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Usable rows per building:"
    )

    print(
        building_summary[
            "usable_rows"
        ]
        .describe()
        .to_string()
    )

    print()
    print(
        "Usable rows by sample group:"
    )

    print(
        usable[
            "sample_group"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        f"Forecast features: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Summary: "
        f"{SUMMARY_PATH}"
    )

    print(
        f"Building summary: "
        f"{BUILDING_SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()