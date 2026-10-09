from pathlib import Path
from time import perf_counter

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comstock_ca_modeling_base.parquet"
)

WEATHER_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nrel"
    / "weather"
    / "amy2018"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comstock_ca_modeling_weather.parquet"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comstock_ca_modeling_weather_summary.csv"
)


BUILDING_ID_COLUMN = "bldg_id"
TIMESTAMP_COLUMN = "timestamp"
COUNTY_COLUMN = "in.county"

WEATHER_TIMESTAMP_COLUMN = "date_time"

WEATHER_COLUMNS = [
    "Dry Bulb Temperature [°C]",
    "Relative Humidity [%]",
    "Wind Speed [m/s]",
    "Wind Direction [Deg]",
    "Global Horizontal Radiation [W/m2]",
    "Direct Normal Radiation [W/m2]",
    "Diffuse Horizontal Radiation [W/m2]",
]

EXPECTED_BUILDINGS = 250
EXPECTED_ROWS = 2_190_000
EXPECTED_WEATHER_ROWS = 8_760


def load_base() -> pd.DataFrame:

    if not BASE_PATH.exists():
        raise FileNotFoundError(
            f"ComStock modeling base not found: "
            f"{BASE_PATH}"
        )

    base = pd.read_parquet(
        BASE_PATH
    )

    required_columns = {
        BUILDING_ID_COLUMN,
        TIMESTAMP_COLUMN,
        COUNTY_COLUMN,
    }

    missing_columns = (
        required_columns
        - set(
            base.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "ComStock modeling base is missing "
            f"required columns: "
            f"{sorted(missing_columns)}"
        )

    return base


def load_weather(
    county: str,
) -> pd.DataFrame:

    weather_path = (
        WEATHER_DIRECTORY
        / f"{county}_2018.csv"
    )

    if not weather_path.exists():
        raise FileNotFoundError(
            f"Weather file not found for "
            f"{county}: {weather_path}"
        )

    weather = pd.read_csv(
        weather_path
    )

    required_columns = {
        WEATHER_TIMESTAMP_COLUMN,
        *WEATHER_COLUMNS,
    }

    missing_columns = (
        required_columns
        - set(
            weather.columns
        )
    )

    if missing_columns:
        raise ValueError(
            f"Weather file for {county} is missing "
            f"required columns: "
            f"{sorted(missing_columns)}"
        )

    if len(
        weather
    ) != EXPECTED_WEATHER_ROWS:
        raise ValueError(
            f"Weather file for {county} contains "
            f"{len(weather):,} rows; expected "
            f"{EXPECTED_WEATHER_ROWS:,}."
        )

    weather = weather[
        [
            WEATHER_TIMESTAMP_COLUMN,
            *WEATHER_COLUMNS,
        ]
    ].copy()

    weather[
        WEATHER_TIMESTAMP_COLUMN
    ] = pd.to_datetime(
        weather[
            WEATHER_TIMESTAMP_COLUMN
        ],
        errors="raise",
    )

    if weather[
        WEATHER_TIMESTAMP_COLUMN
    ].duplicated().any():
        raise ValueError(
            f"Weather file for {county} contains "
            "duplicate timestamps."
        )

    weather[
        COUNTY_COLUMN
    ] = county

    weather = weather.rename(
        columns={
            WEATHER_TIMESTAMP_COLUMN: TIMESTAMP_COLUMN,
        }
    )

    return weather


def build_weather_table(
    counties: list[str],
) -> pd.DataFrame:

    frames = []

    total = len(
        counties
    )

    for number, county in enumerate(
        counties,
        start=1,
    ):

        print(
            f"[{number:02d}/{total:02d}] "
            f"Loading weather for {county}..."
        )

        weather = load_weather(
            county
        )

        frames.append(
            weather
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    duplicate_count = int(
        combined.duplicated(
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
            f"{duplicate_count:,} duplicate "
            "county-timestamp rows."
        )

    return combined


def validate_base(
    base: pd.DataFrame,
) -> None:

    if len(
        base
    ) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} base rows "
            f"but found {len(base):,}."
        )

    if base[
        BUILDING_ID_COLUMN
    ].nunique() != EXPECTED_BUILDINGS:
        raise ValueError(
            f"Expected {EXPECTED_BUILDINGS} buildings "
            f"but found "
            f"{base[BUILDING_ID_COLUMN].nunique()}."
        )

    if base[
        COUNTY_COLUMN
    ].isna().any():
        raise ValueError(
            "ComStock modeling base contains "
            "missing county values."
        )


def validate_merged(
    merged: pd.DataFrame,
) -> None:

    if len(
        merged
    ) != EXPECTED_ROWS:
        raise ValueError(
            f"Weather-merged dataset contains "
            f"{len(merged):,} rows; expected "
            f"{EXPECTED_ROWS:,}."
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
            f"Weather-merged dataset contains "
            f"{duplicate_count:,} duplicate "
            "building-timestamp rows."
        )

    missing_weather = (
        merged[
            WEATHER_COLUMNS
        ]
        .isna()
        .sum()
    )

    if int(
        missing_weather.sum()
    ) != 0:
        print()
        print(
            "Weather missing-value counts:"
        )
        print(
            missing_weather.to_string()
        )

        raise ValueError(
            "Weather merge produced missing "
            "weather values."
        )


def create_summary(
    merged: pd.DataFrame,
    counties: list[str],
) -> pd.DataFrame:

    rows = [
        {
            "metric": "total_rows",
            "value": len(
                merged
            ),
        },
        {
            "metric": "total_buildings",
            "value": merged[
                BUILDING_ID_COLUMN
            ].nunique(),
        },
        {
            "metric": "counties",
            "value": len(
                counties
            ),
        },
        {
            "metric": "weather_features",
            "value": len(
                WEATHER_COLUMNS
            ),
        },
        {
            "metric": "missing_weather_values",
            "value": int(
                merged[
                    WEATHER_COLUMNS
                ]
                .isna()
                .sum()
                .sum()
            ),
        },
        {
            "metric": "duplicate_building_timestamps",
            "value": int(
                merged.duplicated(
                    subset=[
                        BUILDING_ID_COLUMN,
                        TIMESTAMP_COLUMN,
                    ]
                )
                .sum()
            ),
        },
        {
            "metric": "minimum_timestamp",
            "value": str(
                merged[
                    TIMESTAMP_COLUMN
                ].min()
            ),
        },
        {
            "metric": "maximum_timestamp",
            "value": str(
                merged[
                    TIMESTAMP_COLUMN
                ].max()
            ),
        },
    ]

    return pd.DataFrame(
        rows
    )


def main() -> None:

    print(
        "Loading ComStock modeling base..."
    )

    base = load_base()

    validate_base(
        base
    )

    print(
        f"Base rows: "
        f"{len(base):,}"
    )

    print(
        f"Buildings: "
        f"{base[BUILDING_ID_COLUMN].nunique():,}"
    )

    counties = sorted(
        base[
            COUNTY_COLUMN
        ]
        .astype(
            str
        )
        .unique()
        .tolist()
    )

    print(
        f"Counties: "
        f"{len(counties):,}"
    )

    print()

    print(
        "Loading county weather files..."
    )

    start = perf_counter()

    weather = build_weather_table(
        counties
    )

    print()

    print(
        f"Weather rows loaded: "
        f"{len(weather):,}"
    )

    print(
        "Merging weather onto ComStock panel..."
    )

    merged = base.merge(
        weather,
        on=[
            COUNTY_COLUMN,
            TIMESTAMP_COLUMN,
        ],
        how="left",
        validate="many_to_one",
    )

    validate_merged(
        merged
    )

    summary = create_summary(
        merged,
        counties,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Saving weather-merged ComStock dataset..."
    )

    merged.to_parquet(
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

    file_size_mb = (
        OUTPUT_PATH.stat().st_size
        / (
            1024 ** 2
        )
    )

    print()

    print(
        "ComStock weather merge complete."
    )

    print()

    print(
        f"Rows: "
        f"{len(merged):,}"
    )

    print(
        f"Buildings: "
        f"{merged[BUILDING_ID_COLUMN].nunique():,}"
    )

    print(
        f"Counties: "
        f"{len(counties):,}"
    )

    print(
        f"Weather features: "
        f"{len(WEATHER_COLUMNS):,}"
    )

    print(
        f"Missing weather values: "
        f"{int(merged[WEATHER_COLUMNS].isna().sum().sum()):,}"
    )

    print(
        f"Duplicate building timestamps: "
        f"{int(merged.duplicated(subset=[BUILDING_ID_COLUMN, TIMESTAMP_COLUMN]).sum()):,}"
    )

    print(
        f"First timestamp: "
        f"{merged[TIMESTAMP_COLUMN].min()}"
    )

    print(
        f"Last timestamp: "
        f"{merged[TIMESTAMP_COLUMN].max()}"
    )

    print(
        f"Output parquet size: "
        f"{file_size_mb:,.2f} MB"
    )

    print(
        f"Elapsed time: "
        f"{elapsed_seconds:,.2f} seconds"
    )

    print()

    print(
        f"Saved weather-merged data to: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Saved summary to: "
        f"{SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()