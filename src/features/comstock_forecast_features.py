from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comstock_ca_modeling_weather.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comstock_ca_forecast_features.parquet"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comstock_ca_forecast_features_summary.csv"
)

COLUMN_REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comstock_ca_forecast_feature_columns.csv"
)


BUILDING_ID_COLUMN = "bldg_id"
ROLE_COLUMN = "sample_role"
TIMESTAMP_COLUMN = "timestamp"
CONSUMPTION_COLUMN = "electricity_consumption"

TEMPERATURE_COLUMN = "Dry Bulb Temperature [°C]"
RELATIVE_HUMIDITY_COLUMN = "Relative Humidity [%]"
WIND_SPEED_COLUMN = "Wind Speed [m/s]"
WIND_DIRECTION_COLUMN = "Wind Direction [Deg]"
GLOBAL_RADIATION_COLUMN = (
    "Global Horizontal Radiation [W/m2]"
)
DIRECT_RADIATION_COLUMN = (
    "Direct Normal Radiation [W/m2]"
)
DIFFUSE_RADIATION_COLUMN = (
    "Diffuse Horizontal Radiation [W/m2]"
)

TARGET_COLUMN = "target_next_hour"

HEATING_BASE_C = 18.0
COOLING_BASE_C = 22.0

EXPECTED_INPUT_BUILDINGS = 250
EXPECTED_INPUT_ROWS = 2_190_000
EXPECTED_INPUT_ROWS_PER_BUILDING = 8_760

EXPECTED_REMOVED_ROWS_PER_BUILDING = 169
EXPECTED_OUTPUT_ROWS_PER_BUILDING = (
    EXPECTED_INPUT_ROWS_PER_BUILDING
    - EXPECTED_REMOVED_ROWS_PER_BUILDING
)
EXPECTED_OUTPUT_ROWS = (
    EXPECTED_INPUT_BUILDINGS
    * EXPECTED_OUTPUT_ROWS_PER_BUILDING
)


STATIC_MODELING_COLUMNS = [
    "in.building_type",
    "in.sqft",
    "in.number_of_stories",
    "in.hvac_system_type",
    "in.heating_fuel",
    "in.service_water_heating_fuel",
    "in.climate_zone_ashrae_2004",
    "in.climate_zone_building_america",
    "stat.occupant_density_ppl_per_m_2",
]

AUXILIARY_COLUMNS = [
    "in.county",
    "in.resstock_county_id",
    "in.state_abbreviation",
    "in.state_name",
]

WEATHER_MODELING_COLUMNS = [
    TEMPERATURE_COLUMN,
    RELATIVE_HUMIDITY_COLUMN,
    WIND_SPEED_COLUMN,
    GLOBAL_RADIATION_COLUMN,
    DIRECT_RADIATION_COLUMN,
    DIFFUSE_RADIATION_COLUMN,
]

ENGINEERED_TIME_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
]

ENGINEERED_WEATHER_COLUMNS = [
    "heating_degree",
    "cooling_degree",
]

HISTORICAL_LOAD_COLUMNS = [
    "lag_1h",
    "lag_24h",
    "seasonal_naive",
    "rolling_mean_24h",
    "rolling_mean_168h",
]


def load_input() -> pd.DataFrame:

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"ComStock weather dataset not found: "
            f"{INPUT_PATH}"
        )

    data = pd.read_parquet(
        INPUT_PATH
    )

    required_columns = {
        BUILDING_ID_COLUMN,
        ROLE_COLUMN,
        TIMESTAMP_COLUMN,
        CONSUMPTION_COLUMN,
        TEMPERATURE_COLUMN,
        RELATIVE_HUMIDITY_COLUMN,
        WIND_SPEED_COLUMN,
        WIND_DIRECTION_COLUMN,
        GLOBAL_RADIATION_COLUMN,
        DIRECT_RADIATION_COLUMN,
        DIFFUSE_RADIATION_COLUMN,
        *STATIC_MODELING_COLUMNS,
        *AUXILIARY_COLUMNS,
    }

    missing_columns = (
        required_columns
        - set(
            data.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Input dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return data


def validate_input(
    data: pd.DataFrame,
) -> None:

    if len(
        data
    ) != EXPECTED_INPUT_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_INPUT_ROWS:,} rows "
            f"but found {len(data):,}."
        )

    building_count = data[
        BUILDING_ID_COLUMN
    ].nunique()

    if building_count != EXPECTED_INPUT_BUILDINGS:
        raise ValueError(
            f"Expected {EXPECTED_INPUT_BUILDINGS} "
            f"buildings but found {building_count}."
        )

    rows_per_building = (
        data
        .groupby(
            BUILDING_ID_COLUMN
        )
        .size()
    )

    invalid_buildings = rows_per_building.loc[
        rows_per_building
        != EXPECTED_INPUT_ROWS_PER_BUILDING
    ]

    if not invalid_buildings.empty:
        raise ValueError(
            "Some buildings do not contain exactly "
            f"{EXPECTED_INPUT_ROWS_PER_BUILDING:,} "
            "input rows."
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
            f"Input dataset contains "
            f"{duplicate_count:,} duplicate "
            "building-timestamp rows."
        )

    required_no_missing = [
        BUILDING_ID_COLUMN,
        ROLE_COLUMN,
        TIMESTAMP_COLUMN,
        CONSUMPTION_COLUMN,
        *STATIC_MODELING_COLUMNS,
        *WEATHER_MODELING_COLUMNS,
    ]

    missing_count = int(
        data[
            required_no_missing
        ]
        .isna()
        .sum()
        .sum()
    )

    if missing_count != 0:
        raise ValueError(
            f"Input dataset contains "
            f"{missing_count:,} missing values "
            "in required modeling fields."
        )


def add_time_features(
    data: pd.DataFrame,
) -> pd.DataFrame:

    result = data.copy()

    timestamp = result[
        TIMESTAMP_COLUMN
    ]

    result[
        "hour"
    ] = timestamp.dt.hour

    result[
        "day_of_week"
    ] = timestamp.dt.dayofweek

    result[
        "month"
    ] = timestamp.dt.month

    result[
        "day_of_year"
    ] = timestamp.dt.dayofyear

    result[
        "is_weekend"
    ] = (
        result[
            "day_of_week"
        ]
        >= 5
    ).astype(
        "int8"
    )

    result[
        "hour_sin"
    ] = np.sin(
        2.0
        * np.pi
        * result[
            "hour"
        ]
        / 24.0
    )

    result[
        "hour_cos"
    ] = np.cos(
        2.0
        * np.pi
        * result[
            "hour"
        ]
        / 24.0
    )

    result[
        "day_of_week_sin"
    ] = np.sin(
        2.0
        * np.pi
        * result[
            "day_of_week"
        ]
        / 7.0
    )

    result[
        "day_of_week_cos"
    ] = np.cos(
        2.0
        * np.pi
        * result[
            "day_of_week"
        ]
        / 7.0
    )

    result[
        "month_sin"
    ] = np.sin(
        2.0
        * np.pi
        * (
            result[
                "month"
            ]
            - 1
        )
        / 12.0
    )

    result[
        "month_cos"
    ] = np.cos(
        2.0
        * np.pi
        * (
            result[
                "month"
            ]
            - 1
        )
        / 12.0
    )

    return result


def add_weather_features(
    data: pd.DataFrame,
) -> pd.DataFrame:

    result = data.copy()

    temperature = result[
        TEMPERATURE_COLUMN
    ]

    result[
        "heating_degree"
    ] = (
        HEATING_BASE_C
        - temperature
    ).clip(
        lower=0.0
    )

    result[
        "cooling_degree"
    ] = (
        temperature
        - COOLING_BASE_C
    ).clip(
        lower=0.0
    )

    return result


def add_historical_load_features(
    data: pd.DataFrame,
) -> pd.DataFrame:

    result = data.copy()

    result = result.sort_values(
        [
            BUILDING_ID_COLUMN,
            TIMESTAMP_COLUMN,
        ]
    ).reset_index(
        drop=True
    )

    grouped_consumption = result.groupby(
        BUILDING_ID_COLUMN,
        sort=False,
    )[
        CONSUMPTION_COLUMN
    ]

    result[
        "lag_1h"
    ] = grouped_consumption.shift(
        1
    )

    result[
        "lag_24h"
    ] = grouped_consumption.shift(
        24
    )

    result[
        "seasonal_naive"
    ] = grouped_consumption.shift(
        23
    )

    result[
        "rolling_mean_24h"
    ] = (
        grouped_consumption
        .transform(
            lambda series: (
                series
                .shift(
                    1
                )
                .rolling(
                    window=24,
                    min_periods=24,
                )
                .mean()
            )
        )
    )

    result[
        "rolling_mean_168h"
    ] = (
        grouped_consumption
        .transform(
            lambda series: (
                series
                .shift(
                    1
                )
                .rolling(
                    window=168,
                    min_periods=168,
                )
                .mean()
            )
        )
    )

    result[
        TARGET_COLUMN
    ] = grouped_consumption.shift(
        -1
    )

    return result


def remove_unusable_rows(
    data: pd.DataFrame,
) -> pd.DataFrame:

    required_engineered_columns = [
        *HISTORICAL_LOAD_COLUMNS,
        TARGET_COLUMN,
    ]

    result = data.dropna(
        subset=required_engineered_columns
    ).copy()

    result = result.reset_index(
        drop=True
    )

    return result


def validate_output(
    data: pd.DataFrame,
) -> None:

    building_count = data[
        BUILDING_ID_COLUMN
    ].nunique()

    if building_count != EXPECTED_INPUT_BUILDINGS:
        raise ValueError(
            f"Expected {EXPECTED_INPUT_BUILDINGS} "
            f"buildings after feature engineering "
            f"but found {building_count}."
        )

    if len(
        data
    ) != EXPECTED_OUTPUT_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_OUTPUT_ROWS:,} "
            f"feature rows but found "
            f"{len(data):,}."
        )

    rows_per_building = (
        data
        .groupby(
            BUILDING_ID_COLUMN
        )
        .size()
    )

    invalid_buildings = rows_per_building.loc[
        rows_per_building
        != EXPECTED_OUTPUT_ROWS_PER_BUILDING
    ]

    if not invalid_buildings.empty:
        raise ValueError(
            "Some buildings do not contain exactly "
            f"{EXPECTED_OUTPUT_ROWS_PER_BUILDING:,} "
            "usable feature rows."
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
            f"Feature dataset contains "
            f"{duplicate_count:,} duplicate "
            "building-timestamp rows."
        )

    required_complete_columns = [
        CONSUMPTION_COLUMN,
        TARGET_COLUMN,
        *WEATHER_MODELING_COLUMNS,
        *ENGINEERED_TIME_COLUMNS,
        *ENGINEERED_WEATHER_COLUMNS,
        *HISTORICAL_LOAD_COLUMNS,
        *STATIC_MODELING_COLUMNS,
    ]

    missing_count = int(
        data[
            required_complete_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    if missing_count != 0:
        raise ValueError(
            f"Feature dataset contains "
            f"{missing_count:,} missing values "
            "in required forecasting fields."
        )


def create_summary(
    input_data: pd.DataFrame,
    output_data: pd.DataFrame,
) -> pd.DataFrame:

    input_rows = len(
        input_data
    )

    output_rows = len(
        output_data
    )

    rows_removed = (
        input_rows
        - output_rows
    )

    development_rows = int(
        (
            output_data[
                ROLE_COLUMN
            ]
            == "development"
        )
        .sum()
    )

    held_out_rows = int(
        (
            output_data[
                ROLE_COLUMN
            ]
            == "held_out"
        )
        .sum()
    )

    summary_rows = [
        {
            "metric": "input_rows",
            "value": input_rows,
        },
        {
            "metric": "output_rows",
            "value": output_rows,
        },
        {
            "metric": "rows_removed",
            "value": rows_removed,
        },
        {
            "metric": "buildings",
            "value": output_data[
                BUILDING_ID_COLUMN
            ].nunique(),
        },
        {
            "metric": "rows_per_building",
            "value": int(
                output_data.groupby(
                    BUILDING_ID_COLUMN
                )
                .size()
                .median()
            ),
        },
        {
            "metric": "development_rows",
            "value": development_rows,
        },
        {
            "metric": "held_out_rows",
            "value": held_out_rows,
        },
        {
            "metric": "minimum_predictor_timestamp",
            "value": str(
                output_data[
                    TIMESTAMP_COLUMN
                ].min()
            ),
        },
        {
            "metric": "maximum_predictor_timestamp",
            "value": str(
                output_data[
                    TIMESTAMP_COLUMN
                ].max()
            ),
        },
        {
            "metric": "heating_base_c",
            "value": HEATING_BASE_C,
        },
        {
            "metric": "cooling_base_c",
            "value": COOLING_BASE_C,
        },
        {
            "metric": "weather_predictors",
            "value": len(
                WEATHER_MODELING_COLUMNS
            ),
        },
        {
            "metric": "historical_load_features",
            "value": len(
                HISTORICAL_LOAD_COLUMNS
            ),
        },
        {
            "metric": "time_features",
            "value": len(
                ENGINEERED_TIME_COLUMNS
            ),
        },
        {
            "metric": "missing_required_values",
            "value": int(
                output_data[
                    [
                        TARGET_COLUMN,
                        *WEATHER_MODELING_COLUMNS,
                        *ENGINEERED_TIME_COLUMNS,
                        *ENGINEERED_WEATHER_COLUMNS,
                        *HISTORICAL_LOAD_COLUMNS,
                        *STATIC_MODELING_COLUMNS,
                    ]
                ]
                .isna()
                .sum()
                .sum()
            ),
        },
        {
            "metric": "duplicate_building_timestamps",
            "value": int(
                output_data.duplicated(
                    subset=[
                        BUILDING_ID_COLUMN,
                        TIMESTAMP_COLUMN,
                    ]
                )
                .sum()
            ),
        },
    ]

    return pd.DataFrame(
        summary_rows
    )


def create_column_report(
    data: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for column in data.columns:

        if column in {
            BUILDING_ID_COLUMN,
            ROLE_COLUMN,
            TIMESTAMP_COLUMN,
            CONSUMPTION_COLUMN,
            TARGET_COLUMN,
        }:
            category = "core"

        elif column in STATIC_MODELING_COLUMNS:
            category = "static_modeling"

        elif column in AUXILIARY_COLUMNS:
            category = "auxiliary"

        elif column in WEATHER_MODELING_COLUMNS:
            category = "weather_modeling"

        elif column == WIND_DIRECTION_COLUMN:
            category = "weather_excluded"

        elif column in ENGINEERED_TIME_COLUMNS:
            category = "engineered_time"

        elif column in ENGINEERED_WEATHER_COLUMNS:
            category = "engineered_weather"

        elif column in HISTORICAL_LOAD_COLUMNS:
            category = "historical_load"

        else:
            category = "other"

        rows.append(
            {
                "column": column,
                "dtype": str(
                    data[
                        column
                    ].dtype
                ),
                "category": category,
                "missing_values": int(
                    data[
                        column
                    ]
                    .isna()
                    .sum()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main() -> None:

    print(
        "Loading weather-augmented ComStock dataset..."
    )

    start = perf_counter()

    data = load_input()

    print(
        f"Input rows: "
        f"{len(data):,}"
    )

    print(
        f"Buildings: "
        f"{data[BUILDING_ID_COLUMN].nunique():,}"
    )

    print()

    print(
        "Validating input panel..."
    )

    validate_input(
        data
    )

    print(
        "Sorting by building and timestamp..."
    )

    data = data.sort_values(
        [
            BUILDING_ID_COLUMN,
            TIMESTAMP_COLUMN,
        ]
    ).reset_index(
        drop=True
    )

    print(
        "Creating calendar and cyclical features..."
    )

    data = add_time_features(
        data
    )

    print(
        "Creating heating and cooling features..."
    )

    data = add_weather_features(
        data
    )

    print(
        "Creating lag, rolling, baseline, "
        "and forecast-target features..."
    )

    data = add_historical_load_features(
        data
    )

    input_row_count = len(
        data
    )

    print(
        "Removing rows without complete historical "
        "or target information..."
    )

    feature_data = remove_unusable_rows(
        data
    )

    validate_output(
        feature_data
    )

    summary = create_summary(
        data,
        feature_data,
    )

    column_report = create_column_report(
        feature_data
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Saving ComStock forecasting features..."
    )

    feature_data.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    column_report.to_csv(
        COLUMN_REPORT_PATH,
        index=False,
    )

    elapsed_seconds = (
        perf_counter()
        - start
    )

    rows_removed = (
        input_row_count
        - len(
            feature_data
        )
    )

    output_size_mb = (
        OUTPUT_PATH.stat().st_size
        / (
            1024 ** 2
        )
    )

    development_rows = int(
        (
            feature_data[
                ROLE_COLUMN
            ]
            == "development"
        )
        .sum()
    )

    held_out_rows = int(
        (
            feature_data[
                ROLE_COLUMN
            ]
            == "held_out"
        )
        .sum()
    )

    print()

    print(
        "ComStock forecasting feature "
        "engineering complete."
    )

    print()

    print(
        f"Input rows: "
        f"{input_row_count:,}"
    )

    print(
        f"Usable feature rows: "
        f"{len(feature_data):,}"
    )

    print(
        f"Rows removed: "
        f"{rows_removed:,}"
    )

    print(
        f"Rows removed per building: "
        f"{EXPECTED_REMOVED_ROWS_PER_BUILDING:,}"
    )

    print(
        f"Rows per building after engineering: "
        f"{EXPECTED_OUTPUT_ROWS_PER_BUILDING:,}"
    )

    print(
        f"Buildings: "
        f"{feature_data[BUILDING_ID_COLUMN].nunique():,}"
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
        f"First predictor timestamp: "
        f"{feature_data[TIMESTAMP_COLUMN].min()}"
    )

    print(
        f"Last predictor timestamp: "
        f"{feature_data[TIMESTAMP_COLUMN].max()}"
    )

    print(
        f"Missing target values: "
        f"{int(feature_data[TARGET_COLUMN].isna().sum()):,}"
    )

    print(
        f"Duplicate building timestamps: "
        f"{int(feature_data.duplicated(subset=[BUILDING_ID_COLUMN, TIMESTAMP_COLUMN]).sum()):,}"
    )

    print(
        "Wind direction retained for auditing "
        "but excluded from the modeling-weather "
        "predictor list."
    )

    print(
        f"Output parquet size: "
        f"{output_size_mb:,.2f} MB"
    )

    print(
        f"Elapsed time: "
        f"{elapsed_seconds:,.2f} seconds"
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

    print(
        f"Saved column report to: "
        f"{COLUMN_REPORT_PATH}"
    )


if __name__ == "__main__":
    main()