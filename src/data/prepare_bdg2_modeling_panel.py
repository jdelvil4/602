from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ELECTRICITY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bdg2_electricity_hourly.parquet"
)

SAMPLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bdg2_modeling_sample.csv"
)

WEATHER_PATH = (
    PROJECT_ROOT
    / "building-data-genome-project-2"
    / "data"
    / "weather"
    / "weather.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bdg2_modeling_weather.parquet"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bdg2_modeling_weather_summary.csv"
)

BUILDING_COVERAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bdg2_modeling_weather_building_coverage.csv"
)

WEATHER_COLUMNS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bdg2_weather_columns.csv"
)

YEAR = 2017


def load_sample() -> pd.DataFrame:

    print("Loading BDG2 modeling sample...")

    sample = pd.read_csv(
        SAMPLE_PATH,
        low_memory=False,
    )

    required = [
        "building_id",
        "site_id",
        "primaryspaceusage",
        "sqft",
        "sample_group",
    ]

    missing = [
        column
        for column in required
        if column not in sample.columns
    ]

    if missing:
        raise ValueError(
            f"Missing sample columns: {missing}"
        )

    sample["building_id"] = (
        sample["building_id"]
        .astype("string")
        .str.strip()
    )

    sample["site_id"] = (
        sample["site_id"]
        .astype("string")
        .str.strip()
    )

    if sample["building_id"].duplicated().any():
        raise ValueError(
            "Duplicate building IDs found in BDG2 sample."
        )

    print(
        f"Selected buildings: "
        f"{sample['building_id'].nunique():,}"
    )

    print(
        f"Development buildings: "
        f"{(sample['sample_group'] == 'development').sum():,}"
    )

    print(
        f"Held-out buildings: "
        f"{(sample['sample_group'] == 'held_out').sum():,}"
    )

    return sample


def load_selected_electricity(
    sample: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print(
        "Loading selected 2017 electricity observations..."
    )

    building_ids = (
        sample["building_id"]
        .astype(str)
        .tolist()
    )

    start = pd.Timestamp(
        f"{YEAR}-01-01 00:00:00"
    )

    end = pd.Timestamp(
        f"{YEAR}-12-31 23:00:00"
    )

    data = pd.read_parquet(
        ELECTRICITY_PATH,
        filters=[
            (
                "building_id",
                "in",
                building_ids,
            ),
            (
                "timestamp",
                ">=",
                start,
            ),
            (
                "timestamp",
                "<=",
                end,
            ),
        ],
    )

    data["building_id"] = (
        data["building_id"]
        .astype("string")
        .str.strip()
    )

    data["site_id"] = (
        data["site_id"]
        .astype("string")
        .str.strip()
    )

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="raise",
    )

    print(
        f"Electricity rows loaded: "
        f"{len(data):,}"
    )

    print(
        f"Buildings loaded: "
        f"{data['building_id'].nunique():,}"
    )

    print(
        f"Sites represented: "
        f"{data['site_id'].nunique():,}"
    )

    print(
        f"Timestamp range: "
        f"{data['timestamp'].min()} "
        f"to {data['timestamp'].max()}"
    )

    return data


def load_weather(
    selected_sites: list[str],
) -> pd.DataFrame:

    print()
    print("Loading BDG2 weather...")

    weather = pd.read_csv(
        WEATHER_PATH,
        low_memory=False,
    )

    required = [
        "timestamp",
        "site_id",
    ]

    missing = [
        column
        for column in required
        if column not in weather.columns
    ]

    if missing:
        raise ValueError(
            f"Missing weather columns: {missing}"
        )

    weather["timestamp"] = pd.to_datetime(
        weather["timestamp"],
        errors="coerce",
    )

    if weather["timestamp"].isna().any():
        raise ValueError(
            "Unparseable timestamps found in weather data."
        )

    weather["site_id"] = (
        weather["site_id"]
        .astype("string")
        .str.strip()
    )

    weather = weather[
        (
            weather["timestamp"].dt.year
            == YEAR
        )
        & (
            weather["site_id"].isin(
                selected_sites
            )
        )
    ].copy()

    duplicate_weather = int(
        weather.duplicated(
            subset=[
                "site_id",
                "timestamp",
            ]
        ).sum()
    )

    if duplicate_weather > 0:
        raise ValueError(
            f"Duplicate site-hour weather rows: "
            f"{duplicate_weather:,}"
        )

    weather_report = pd.DataFrame(
        {
            "column":
                weather.columns,
            "dtype":
                [
                    str(dtype)
                    for dtype
                    in weather.dtypes
                ],
            "missing_count":
                [
                    int(
                        weather[column]
                        .isna()
                        .sum()
                    )
                    for column
                    in weather.columns
                ],
            "missing_percentage":
                [
                    (
                        weather[column]
                        .isna()
                        .mean()
                        * 100.0
                    )
                    for column
                    in weather.columns
                ],
        }
    )

    weather_report.to_csv(
        WEATHER_COLUMNS_PATH,
        index=False,
    )

    print(
        f"Weather rows retained: "
        f"{len(weather):,}"
    )

    print(
        f"Weather sites retained: "
        f"{weather['site_id'].nunique():,}"
    )

    print()
    print("Weather columns:")

    for column in weather.columns:
        missing_percentage = (
            weather[column]
            .isna()
            .mean()
            * 100.0
        )

        print(
            f"  {column}: "
            f"{missing_percentage:.3f}% missing"
        )

    return weather


def build_coverage_summary(
    data: pd.DataFrame,
) -> pd.DataFrame:

    coverage = (
        data
        .groupby(
            "building_id",
            observed=True,
        )
        .agg(
            sample_group=(
                "sample_group",
                "first",
            ),
            site_id=(
                "site_id",
                "first",
            ),
            primaryspaceusage=(
                "primaryspaceusage",
                "first",
            ),
            sqft=(
                "sqft",
                "first",
            ),
            total_rows=(
                "timestamp",
                "size",
            ),
            observed_electricity=(
                "electricity",
                "count",
            ),
            first_timestamp=(
                "timestamp",
                "min",
            ),
            last_timestamp=(
                "timestamp",
                "max",
            ),
        )
        .reset_index()
    )

    coverage[
        "missing_electricity"
    ] = (
        coverage["total_rows"]
        - coverage["observed_electricity"]
    )

    coverage[
        "electricity_coverage_percentage"
    ] = (
        coverage["observed_electricity"]
        / coverage["total_rows"]
        * 100.0
    )

    return coverage


def main() -> None:

    sample = load_sample()

    electricity = (
        load_selected_electricity(
            sample
        )
    )

    selected_sites = (
        sample["site_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    weather = load_weather(
        selected_sites
    )

    metadata_columns = [
        "building_id",
        "sample_group",
    ]

    electricity = (
        electricity
        .drop(
            columns=[
                "sample_group"
            ],
            errors="ignore",
        )
        .merge(
            sample[
                metadata_columns
            ],
            on="building_id",
            how="left",
            validate="many_to_one",
        )
    )

    print()
    print(
        "Merging site-level weather..."
    )

    merged = electricity.merge(
        weather,
        on=[
            "site_id",
            "timestamp",
        ],
        how="left",
        validate="many_to_one",
        suffixes=(
            "",
            "_weather",
        ),
    )

    if len(merged) != len(electricity):
        raise ValueError(
            "Weather merge changed row count."
        )

    duplicates = int(
        merged.duplicated(
            subset=[
                "building_id",
                "timestamp",
            ]
        ).sum()
    )

    if duplicates > 0:
        raise ValueError(
            f"Duplicate building-hour rows: "
            f"{duplicates:,}"
        )

    weather_feature_columns = [
        column
        for column in weather.columns
        if column
        not in [
            "timestamp",
            "site_id",
        ]
    ]

    missing_weather_cells = int(
        merged[
            weather_feature_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    rows_with_missing_weather = int(
        merged[
            weather_feature_columns
        ]
        .isna()
        .any(
            axis=1
        )
        .sum()
    )

    coverage = build_coverage_summary(
        merged
    )

    summary = pd.DataFrame(
        [
            {
                "year":
                    YEAR,
                "buildings_total":
                    merged[
                        "building_id"
                    ].nunique(),
                "development_buildings":
                    merged.loc[
                        merged[
                            "sample_group"
                        ]
                        == "development",
                        "building_id",
                    ].nunique(),
                "held_out_buildings":
                    merged.loc[
                        merged[
                            "sample_group"
                        ]
                        == "held_out",
                        "building_id",
                    ].nunique(),
                "sites":
                    merged[
                        "site_id"
                    ].nunique(),
                "rows_total":
                    len(
                        merged
                    ),
                "observed_electricity_rows":
                    int(
                        merged[
                            "electricity"
                        ]
                        .notna()
                        .sum()
                    ),
                "missing_electricity_rows":
                    int(
                        merged[
                            "electricity"
                        ]
                        .isna()
                        .sum()
                    ),
                "electricity_missing_percentage":
                    (
                        merged[
                            "electricity"
                        ]
                        .isna()
                        .mean()
                        * 100.0
                    ),
                "weather_feature_count":
                    len(
                        weather_feature_columns
                    ),
                "weather_missing_cells":
                    missing_weather_cells,
                "rows_with_missing_weather":
                    rows_with_missing_weather,
                "duplicate_building_timestamps":
                    duplicates,
                "minimum_timestamp":
                    merged[
                        "timestamp"
                    ].min(),
                "maximum_timestamp":
                    merged[
                        "timestamp"
                    ].max(),
            }
        ]
    )

    print()
    print(
        "Saving BDG2 modeling/weather panel..."
    )

    merged.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    coverage.to_csv(
        BUILDING_COVERAGE_PATH,
        index=False,
    )

    print()
    print(
        "BDG2 modeling/weather preparation complete."
    )
    print()

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Selected-building electricity coverage:"
    )

    print(
        coverage[
            "electricity_coverage_percentage"
        ]
        .describe()
        .to_string()
    )

    print()
    print(
        "Weather missingness:"
    )

    for column in (
        weather_feature_columns
    ):
        count = int(
            merged[column]
            .isna()
            .sum()
        )

        percentage = (
            merged[column]
            .isna()
            .mean()
            * 100.0
        )

        print(
            f"  {column}: "
            f"{count:,} "
            f"({percentage:.3f}%)"
        )

    print()
    print(
        f"Modeling/weather panel: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Summary: "
        f"{SUMMARY_PATH}"
    )

    print(
        f"Building coverage: "
        f"{BUILDING_COVERAGE_PATH}"
    )

    print(
        f"Weather column report: "
        f"{WEATHER_COLUMNS_PATH}"
    )


if __name__ == "__main__":
    main()