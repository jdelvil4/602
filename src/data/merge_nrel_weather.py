from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELING_BASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_modeling_base.parquet"
)

WEATHER_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nrel"
    / "weather"
    / "amy2018"
)

WEATHER_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_weather_download_manifest.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_modeling_weather.parquet"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_modeling_weather_summary.csv"
)

WEATHER_COLUMN_REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_modeling_weather_columns.csv"
)


BUILDING_ID_COLUMN = "bldg_id"
ROLE_COLUMN = "sample_role"
COUNTY_COLUMN = "in.county"
TIMESTAMP_COLUMN = "timestamp"
TARGET_COLUMN = "electricity_total_hourly"

WEATHER_TIMESTAMP_COLUMN = "date_time"

EXPECTED_BUILDINGS = 250
EXPECTED_DEVELOPMENT_BUILDINGS = 200
EXPECTED_HELD_OUT_BUILDINGS = 50
EXPECTED_COUNTIES = 41
EXPECTED_ROWS = 2_190_000
EXPECTED_HOURS_PER_BUILDING = 8760
EXPECTED_HOURS_PER_COUNTY = 8760

EXPECTED_MIN_TIMESTAMP = pd.Timestamp(
    "2018-01-01 01:00:00"
)

EXPECTED_MAX_TIMESTAMP = pd.Timestamp(
    "2019-01-01 00:00:00"
)


WEATHER_COLUMN_RENAME_MAP = {
    "Dry Bulb Temperature [°C]": "weather_dry_bulb_temperature_c",
    "Relative Humidity [%]": "weather_relative_humidity_pct",
    "Wind Speed [m/s]": "weather_wind_speed_m_s",
    "Wind Direction [Deg]": "weather_wind_direction_deg",
    "Global Horizontal Radiation [W/m2]": "weather_global_horizontal_radiation_w_m2",
    "Direct Normal Radiation [W/m2]": "weather_direct_normal_radiation_w_m2",
    "Diffuse Horizontal Radiation [W/m2]": "weather_diffuse_horizontal_radiation_w_m2",
}


def load_modeling_base() -> pd.DataFrame:

    if not MODELING_BASE_PATH.exists():
        raise FileNotFoundError(
            f"Modeling base dataset not found: "
            f"{MODELING_BASE_PATH}"
        )

    df = pd.read_parquet(
        MODELING_BASE_PATH
    )

    required_columns = {
        BUILDING_ID_COLUMN,
        ROLE_COLUMN,
        COUNTY_COLUMN,
        TIMESTAMP_COLUMN,
        TARGET_COLUMN,
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Modeling base dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df[BUILDING_ID_COLUMN] = pd.to_numeric(
        df[BUILDING_ID_COLUMN],
        errors="coerce",
    )

    if df[BUILDING_ID_COLUMN].isna().any():
        raise ValueError(
            "Modeling base dataset contains invalid building IDs."
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

    df[COUNTY_COLUMN] = (
        df[COUNTY_COLUMN]
        .astype(str)
        .str.strip()
    )

    df[TIMESTAMP_COLUMN] = pd.to_datetime(
        df[TIMESTAMP_COLUMN]
    )

    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Modeling base contains {len(df):,} rows; "
            f"expected {EXPECTED_ROWS:,}."
        )

    building_count = df[
        BUILDING_ID_COLUMN
    ].nunique()

    if building_count != EXPECTED_BUILDINGS:
        raise ValueError(
            f"Modeling base contains {building_count} buildings; "
            f"expected {EXPECTED_BUILDINGS}."
        )

    county_count = df[
        COUNTY_COLUMN
    ].nunique()

    if county_count != EXPECTED_COUNTIES:
        raise ValueError(
            f"Modeling base contains {county_count} counties; "
            f"expected {EXPECTED_COUNTIES}."
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
            f"Modeling base contains {duplicate_count} "
            "duplicate building-timestamp rows."
        )

    return df


def load_weather_manifest() -> pd.DataFrame:

    if not WEATHER_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Weather manifest not found: "
            f"{WEATHER_MANIFEST_PATH}"
        )

    manifest = pd.read_csv(
        WEATHER_MANIFEST_PATH
    )

    required_columns = {
        "county_id",
        "filename",
        "status",
        "validation_status",
    }

    missing_columns = (
        required_columns
        - set(manifest.columns)
    )

    if missing_columns:
        raise ValueError(
            "Weather manifest is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    manifest["county_id"] = (
        manifest["county_id"]
        .astype(str)
        .str.strip()
    )

    usable = (
        manifest.loc[
            (
                manifest["status"].isin(
                    [
                        "downloaded",
                        "already_exists",
                    ]
                )
            )
            &
            (
                manifest[
                    "validation_status"
                ]
                == "valid"
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    if len(usable) != EXPECTED_COUNTIES:
        raise ValueError(
            f"Weather manifest contains {len(usable)} usable counties; "
            f"expected {EXPECTED_COUNTIES}."
        )

    duplicate_count = int(
        usable[
            "county_id"
        ]
        .duplicated()
        .sum()
    )

    if duplicate_count != 0:
        raise ValueError(
            f"Weather manifest contains {duplicate_count} "
            "duplicate county IDs."
        )

    return (
        usable
        .sort_values(
            "county_id"
        )
        .reset_index(drop=True)
    )


def load_county_weather(
    county_id: str,
    filename: str,
) -> pd.DataFrame:

    path = (
        WEATHER_DIR
        / filename
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Weather file not found for county "
            f"{county_id}: {path}"
        )

    df = pd.read_csv(
        path
    )

    required_columns = {
        WEATHER_TIMESTAMP_COLUMN,
        *WEATHER_COLUMN_RENAME_MAP.keys(),
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Weather file for {county_id} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    df[
        WEATHER_TIMESTAMP_COLUMN
    ] = pd.to_datetime(
        df[
            WEATHER_TIMESTAMP_COLUMN
        ]
    )

    if len(df) != EXPECTED_HOURS_PER_COUNTY:
        raise ValueError(
            f"Weather file for {county_id} contains "
            f"{len(df):,} rows; expected "
            f"{EXPECTED_HOURS_PER_COUNTY:,}."
        )

    duplicate_count = int(
        df[
            WEATHER_TIMESTAMP_COLUMN
        ]
        .duplicated()
        .sum()
    )

    if duplicate_count != 0:
        raise ValueError(
            f"Weather file for {county_id} contains "
            f"{duplicate_count} duplicate timestamps."
        )

    missing_count = int(
        df.isna()
        .sum()
        .sum()
    )

    if missing_count != 0:
        raise ValueError(
            f"Weather file for {county_id} contains "
            f"{missing_count} missing values."
        )

    minimum_timestamp = df[
        WEATHER_TIMESTAMP_COLUMN
    ].min()

    maximum_timestamp = df[
        WEATHER_TIMESTAMP_COLUMN
    ].max()

    if minimum_timestamp != EXPECTED_MIN_TIMESTAMP:
        raise ValueError(
            f"Weather file for {county_id} begins at "
            f"{minimum_timestamp}; expected "
            f"{EXPECTED_MIN_TIMESTAMP}."
        )

    if maximum_timestamp != EXPECTED_MAX_TIMESTAMP:
        raise ValueError(
            f"Weather file for {county_id} ends at "
            f"{maximum_timestamp}; expected "
            f"{EXPECTED_MAX_TIMESTAMP}."
        )

    timestamp_differences = (
        df[
            WEATHER_TIMESTAMP_COLUMN
        ]
        .sort_values()
        .diff()
        .dropna()
    )

    invalid_intervals = int(
        (
            timestamp_differences
            != pd.Timedelta(
                hours=1
            )
        )
        .sum()
    )

    if invalid_intervals != 0:
        raise ValueError(
            f"Weather file for {county_id} contains "
            f"{invalid_intervals} non-hourly intervals."
        )

    selected = (
        df[
            [
                WEATHER_TIMESTAMP_COLUMN,
                *WEATHER_COLUMN_RENAME_MAP.keys(),
            ]
        ]
        .copy()
    )

    selected = selected.rename(
        columns={
            WEATHER_TIMESTAMP_COLUMN: TIMESTAMP_COLUMN,
            **WEATHER_COLUMN_RENAME_MAP,
        }
    )

    selected.insert(
        0,
        COUNTY_COLUMN,
        county_id,
    )

    return selected


def build_weather_table(
    manifest: pd.DataFrame,
) -> pd.DataFrame:

    frames = []

    for position, row in enumerate(
        manifest.itertuples(
            index=False
        ),
        start=1,
    ):

        county_id = str(
            row.county_id
        )

        filename = str(
            row.filename
        )

        print(
            f"[{position}/{len(manifest)}] "
            f"Loading weather for "
            f"{county_id}"
        )

        county_weather = load_county_weather(
            county_id,
            filename,
        )

        frames.append(
            county_weather
        )

    weather = pd.concat(
        frames,
        ignore_index=True,
    )

    expected_rows = (
        EXPECTED_COUNTIES
        * EXPECTED_HOURS_PER_COUNTY
    )

    if len(weather) != expected_rows:
        raise ValueError(
            f"Combined weather table contains "
            f"{len(weather):,} rows; "
            f"expected {expected_rows:,}."
        )

    duplicate_count = int(
        weather.duplicated(
            subset=[
                COUNTY_COLUMN,
                TIMESTAMP_COLUMN,
            ]
        )
        .sum()
    )

    if duplicate_count != 0:
        raise ValueError(
            f"Combined weather table contains "
            f"{duplicate_count} duplicate county-timestamp rows."
        )

    missing_count = int(
        weather.isna()
        .sum()
        .sum()
    )

    if missing_count != 0:
        raise ValueError(
            f"Combined weather table contains "
            f"{missing_count} missing values."
        )

    return (
        weather
        .sort_values(
            [
                COUNTY_COLUMN,
                TIMESTAMP_COLUMN,
            ]
        )
        .reset_index(drop=True)
    )


def merge_weather(
    modeling_base: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:

    original_row_count = len(
        modeling_base
    )

    merged = modeling_base.merge(
        weather,
        on=[
            COUNTY_COLUMN,
            TIMESTAMP_COLUMN,
        ],
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    if len(merged) != original_row_count:
        raise ValueError(
            f"Weather merge changed row count from "
            f"{original_row_count:,} to "
            f"{len(merged):,}."
        )

    unmatched_rows = int(
        (
            merged["_merge"]
            != "both"
        )
        .sum()
    )

    if unmatched_rows != 0:
        unmatched_examples = (
            merged.loc[
                merged["_merge"]
                != "both",
                [
                    BUILDING_ID_COLUMN,
                    COUNTY_COLUMN,
                    TIMESTAMP_COLUMN,
                ],
            ]
            .head(
                20
            )
        )

        raise ValueError(
            f"Weather merge left "
            f"{unmatched_rows:,} unmatched rows.\n"
            f"{unmatched_examples.to_string(index=False)}"
        )

    merged = (
        merged
        .drop(
            columns=[
                "_merge",
            ]
        )
    )

    return merged


def validate_merged_dataset(
    merged: pd.DataFrame,
) -> None:

    if len(merged) != EXPECTED_ROWS:
        raise ValueError(
            f"Merged dataset contains {len(merged):,} rows; "
            f"expected {EXPECTED_ROWS:,}."
        )

    building_count = merged[
        BUILDING_ID_COLUMN
    ].nunique()

    if building_count != EXPECTED_BUILDINGS:
        raise ValueError(
            f"Merged dataset contains {building_count} buildings; "
            f"expected {EXPECTED_BUILDINGS}."
        )

    county_count = merged[
        COUNTY_COLUMN
    ].nunique()

    if county_count != EXPECTED_COUNTIES:
        raise ValueError(
            f"Merged dataset contains {county_count} counties; "
            f"expected {EXPECTED_COUNTIES}."
        )

    duplicate_count = int(
        merged.duplicated(
            subset=[
                BUILDING_ID_COLUMN,
                TIMESTAMP_COLUMN,
            ]
        )
        .sum()
    )

    if duplicate_count != 0:
        raise ValueError(
            f"Merged dataset contains "
            f"{duplicate_count} duplicate building-timestamp rows."
        )

    rows_per_building = (
        merged.groupby(
            BUILDING_ID_COLUMN
        )
        .size()
    )

    invalid_buildings = (
        rows_per_building[
            rows_per_building
            != EXPECTED_HOURS_PER_BUILDING
        ]
    )

    if not invalid_buildings.empty:
        raise ValueError(
            "One or more buildings do not contain exactly "
            f"{EXPECTED_HOURS_PER_BUILDING:,} rows.\n"
            f"{invalid_buildings.to_string()}"
        )

    minimum_timestamp = merged[
        TIMESTAMP_COLUMN
    ].min()

    maximum_timestamp = merged[
        TIMESTAMP_COLUMN
    ].max()

    if minimum_timestamp != EXPECTED_MIN_TIMESTAMP:
        raise ValueError(
            f"Merged dataset begins at "
            f"{minimum_timestamp}; expected "
            f"{EXPECTED_MIN_TIMESTAMP}."
        )

    if maximum_timestamp != EXPECTED_MAX_TIMESTAMP:
        raise ValueError(
            f"Merged dataset ends at "
            f"{maximum_timestamp}; expected "
            f"{EXPECTED_MAX_TIMESTAMP}."
        )

    core_columns = [
        BUILDING_ID_COLUMN,
        ROLE_COLUMN,
        COUNTY_COLUMN,
        TIMESTAMP_COLUMN,
        TARGET_COLUMN,
    ]

    core_missing_count = int(
        merged[
            core_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    if core_missing_count != 0:
        raise ValueError(
            f"Merged dataset contains "
            f"{core_missing_count} missing core values."
        )

    weather_columns = list(
        WEATHER_COLUMN_RENAME_MAP.values()
    )

    weather_missing_count = int(
        merged[
            weather_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    if weather_missing_count != 0:
        raise ValueError(
            f"Merged dataset contains "
            f"{weather_missing_count} missing weather values."
        )

    development_buildings = merged.loc[
        merged[
            ROLE_COLUMN
        ]
        == "development",
        BUILDING_ID_COLUMN,
    ].nunique()

    held_out_buildings = merged.loc[
        merged[
            ROLE_COLUMN
        ]
        == "held_out",
        BUILDING_ID_COLUMN,
    ].nunique()

    if (
        development_buildings
        != EXPECTED_DEVELOPMENT_BUILDINGS
    ):
        raise ValueError(
            f"Merged dataset contains "
            f"{development_buildings} development buildings; "
            f"expected {EXPECTED_DEVELOPMENT_BUILDINGS}."
        )

    if (
        held_out_buildings
        != EXPECTED_HELD_OUT_BUILDINGS
    ):
        raise ValueError(
            f"Merged dataset contains "
            f"{held_out_buildings} held-out buildings; "
            f"expected {EXPECTED_HELD_OUT_BUILDINGS}."
        )


def create_summary(
    merged: pd.DataFrame,
) -> pd.DataFrame:

    weather_columns = list(
        WEATHER_COLUMN_RENAME_MAP.values()
    )

    weather_missing_values = int(
        merged[
            weather_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    output_file_size_mb = (
        OUTPUT_PATH.stat().st_size
        / 1_048_576
        if OUTPUT_PATH.exists()
        else None
    )

    summary = pd.DataFrame(
        [
            {
                "buildings_total": merged[
                    BUILDING_ID_COLUMN
                ].nunique(),
                "development_buildings": merged.loc[
                    merged[
                        ROLE_COLUMN
                    ]
                    == "development",
                    BUILDING_ID_COLUMN,
                ].nunique(),
                "held_out_buildings": merged.loc[
                    merged[
                        ROLE_COLUMN
                    ]
                    == "held_out",
                    BUILDING_ID_COLUMN,
                ].nunique(),
                "counties_total": merged[
                    COUNTY_COLUMN
                ].nunique(),
                "rows_total": len(
                    merged
                ),
                "minimum_timestamp": merged[
                    TIMESTAMP_COLUMN
                ].min(),
                "maximum_timestamp": merged[
                    TIMESTAMP_COLUMN
                ].max(),
                "weather_feature_count": len(
                    weather_columns
                ),
                "weather_missing_values": (
                    weather_missing_values
                ),
                "duplicate_building_timestamps": int(
                    merged.duplicated(
                        subset=[
                            BUILDING_ID_COLUMN,
                            TIMESTAMP_COLUMN,
                        ]
                    )
                    .sum()
                ),
                "output_parquet_size_mb": (
                    output_file_size_mb
                ),
            }
        ]
    )

    return summary


def create_weather_column_report(
    merged: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for column in (
        WEATHER_COLUMN_RENAME_MAP.values()
    ):

        series = merged[
            column
        ]

        rows.append(
            {
                "column": column,
                "dtype": str(
                    series.dtype
                ),
                "missing_count": int(
                    series
                    .isna()
                    .sum()
                ),
                "minimum": float(
                    series.min()
                ),
                "maximum": float(
                    series.max()
                ),
                "mean": float(
                    series.mean()
                ),
                "median": float(
                    series.median()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main() -> None:

    print(
        "Loading NREL modeling base dataset..."
    )

    modeling_base = load_modeling_base()

    print(
        f"Modeling rows loaded: "
        f"{len(modeling_base):,}"
    )

    print(
        f"Buildings: "
        f"{modeling_base[BUILDING_ID_COLUMN].nunique():,}"
    )

    print(
        f"Counties: "
        f"{modeling_base[COUNTY_COLUMN].nunique():,}"
    )

    print()

    print(
        "Loading NREL weather manifest..."
    )

    weather_manifest = (
        load_weather_manifest()
    )

    print(
        f"Usable weather files: "
        f"{len(weather_manifest):,}"
    )

    print()

    print(
        "Building combined county weather table..."
    )

    weather = build_weather_table(
        weather_manifest
    )

    print()

    print(
        f"Combined weather rows: "
        f"{len(weather):,}"
    )

    print(
        f"Combined weather counties: "
        f"{weather[COUNTY_COLUMN].nunique():,}"
    )

    print()

    print(
        "Merging weather onto modeling base..."
    )

    merged = merge_weather(
        modeling_base,
        weather,
    )

    print(
        "Validating weather-enriched modeling dataset..."
    )

    validate_merged_dataset(
        merged
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Saving weather-enriched modeling dataset..."
    )

    merged.to_parquet(
        OUTPUT_PATH,
        index=False,
        compression="snappy",
    )

    summary = create_summary(
        merged
    )

    weather_report = (
        create_weather_column_report(
            merged
        )
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    weather_report.to_csv(
        WEATHER_COLUMN_REPORT_PATH,
        index=False,
    )

    print()

    print(
        "NREL weather merge complete."
    )

    print()

    print(
        summary.to_string(
            index=False
        )
    )

    print()

    print(
        "Weather column report:"
    )

    print(
        weather_report.to_string(
            index=False
        )
    )

    print()

    print(
        f"Saved weather-enriched dataset to: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Saved summary to: "
        f"{SUMMARY_PATH}"
    )

    print(
        f"Saved weather column report to: "
        f"{WEATHER_COLUMN_REPORT_PATH}"
    )


if __name__ == "__main__":
    main()