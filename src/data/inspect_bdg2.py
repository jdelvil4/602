from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BDG2_ROOT = (
    PROJECT_ROOT
    / "building-data-genome-project-2"
    / "data"
)

METADATA_PATH = (
    BDG2_ROOT
    / "metadata"
    / "metadata.csv"
)

WEATHER_PATH = (
    BDG2_ROOT
    / "weather"
    / "weather.csv"
)

CLEANED_METERS_DIR = (
    BDG2_ROOT
    / "meters"
    / "cleaned"
)


def print_section(title: str) -> None:

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def inspect_metadata() -> None:

    print_section(
        "BDG2 METADATA"
    )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: "
            f"{METADATA_PATH}"
        )

    metadata = pd.read_csv(
        METADATA_PATH,
        low_memory=False,
    )

    print(
        f"Path: {METADATA_PATH}"
    )

    print(
        f"Rows: {len(metadata):,}"
    )

    print(
        f"Columns: {len(metadata.columns):,}"
    )

    print()

    print(
        "Columns:"
    )

    for column in metadata.columns:
        print(
            f"  {column}"
        )

    print()

    print(
        "First five rows:"
    )

    print(
        metadata.head().to_string(
            index=False
        )
    )

    print()

    print(
        "Missing-value percentages:"
    )

    missing = (
        metadata
        .isna()
        .mean()
        .mul(
            100.0
        )
        .sort_values(
            ascending=False
        )
    )

    print(
        missing.to_string()
    )

    for candidate in [
        "building_id",
        "building",
        "site_id",
        "primaryspaceusage",
        "primary_space_usage",
        "timezone",
        "sqft",
        "sqm",
        "yearbuilt",
        "numberoffloors",
    ]:

        if candidate in metadata.columns:

            print()

            print(
                f"Value counts for "
                f"{candidate}:"
            )

            print(
                metadata[
                    candidate
                ]
                .value_counts(
                    dropna=False
                )
                .head(
                    30
                )
                .to_string()
            )


def find_cleaned_electricity_files() -> list[Path]:

    if not CLEANED_METERS_DIR.exists():
        raise FileNotFoundError(
            f"Cleaned meter directory not found: "
            f"{CLEANED_METERS_DIR}"
        )

    files = sorted(
        [
            path
            for path in CLEANED_METERS_DIR.rglob(
                "*"
            )
            if (
                path.is_file()
                and "electric" in path.name.lower()
                and path.suffix.lower()
                in {
                    ".csv",
                    ".parquet",
                }
            )
        ]
    )

    return files


def inspect_meter_file(
    path: Path,
) -> None:

    print_section(
        "BDG2 CLEANED ELECTRICITY METER DATA"
    )

    print(
        f"Path: {path}"
    )

    print(
        f"File size: "
        f"{path.stat().st_size / 1024 / 1024:,.2f} MB"
    )

    if (
        path.suffix.lower()
        == ".csv"
    ):

        sample = pd.read_csv(
            path,
            nrows=10,
            low_memory=False,
        )

    elif (
        path.suffix.lower()
        == ".parquet"
    ):

        sample = pd.read_parquet(
            path
        ).head(
            10
        )

    else:
        raise ValueError(
            f"Unsupported meter format: "
            f"{path.suffix}"
        )

    print()

    print(
        f"Sample columns: "
        f"{len(sample.columns):,}"
    )

    print()

    print(
        "Columns:"
    )

    for column in sample.columns:
        print(
            f"  {column}"
        )

    print()

    print(
        "First ten rows:"
    )

    print(
        sample.to_string(
            index=False
        )
    )

    print()

    first_column = str(
        sample.columns[
            0
        ]
    )

    print(
        f"First column: "
        f"{first_column}"
    )

    print()

    print(
        "Sample dtypes:"
    )

    print(
        sample.dtypes.to_string()
    )


def inspect_weather() -> None:

    print_section(
        "BDG2 WEATHER"
    )

    if not WEATHER_PATH.exists():
        raise FileNotFoundError(
            f"Weather file not found: "
            f"{WEATHER_PATH}"
        )

    weather = pd.read_csv(
        WEATHER_PATH,
        low_memory=False,
    )

    print(
        f"Path: {WEATHER_PATH}"
    )

    print(
        f"Rows: {len(weather):,}"
    )

    print(
        f"Columns: {len(weather.columns):,}"
    )

    print()

    print(
        "Columns:"
    )

    for column in weather.columns:
        print(
            f"  {column}"
        )

    print()

    print(
        "First five rows:"
    )

    print(
        weather.head().to_string(
            index=False
        )
    )

    print()

    print(
        "Missing-value percentages:"
    )

    missing = (
        weather
        .isna()
        .mean()
        .mul(
            100.0
        )
        .sort_values(
            ascending=False
        )
    )

    print(
        missing.to_string()
    )

    for candidate in [
        "timestamp",
        "site_id",
        "airTemperature",
        "dewTemperature",
        "seaLvlPressure",
        "windDirection",
        "windSpeed",
    ]:

        if candidate not in weather.columns:
            continue

        if candidate == "timestamp":

            parsed = pd.to_datetime(
                weather[
                    candidate
                ],
                errors="coerce",
            )

            print()

            print(
                f"Weather timestamp minimum: "
                f"{parsed.min()}"
            )

            print(
                f"Weather timestamp maximum: "
                f"{parsed.max()}"
            )

            print(
                f"Unparseable timestamps: "
                f"{parsed.isna().sum():,}"
            )

        elif candidate == "site_id":

            print()

            print(
                f"Weather sites: "
                f"{weather[candidate].nunique():,}"
            )

            print(
                weather[
                    candidate
                ]
                .value_counts(
                    dropna=False
                )
                .head(
                    30
                )
                .to_string()
            )


def main() -> None:

    print(
        "Inspecting Building Data Genome Project 2..."
    )

    inspect_metadata()

    meter_files = (
        find_cleaned_electricity_files()
    )

    print_section(
        "CLEANED ELECTRICITY FILE DISCOVERY"
    )

    print(
        f"Electricity files found: "
        f"{len(meter_files):,}"
    )

    for path in meter_files:
        print(
            f"  {path}"
        )

    if not meter_files:
        raise FileNotFoundError(
            "No cleaned electricity meter file "
            "was found."
        )

    inspect_meter_file(
        meter_files[
            0
        ]
    )

    inspect_weather()

    print_section(
        "BDG2 INSPECTION COMPLETE"
    )

    print(
        "Next step: use the observed schemas "
        "to build the compact BDG2 preparation "
        "and ComStock compatibility pipeline."
    )


if __name__ == "__main__":
    main()