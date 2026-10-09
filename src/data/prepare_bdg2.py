from pathlib import Path

import numpy as np
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

ELECTRICITY_PATH = (
    BDG2_ROOT
    / "meters"
    / "cleaned"
    / "electricity_cleaned.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_PARQUET = (
    OUTPUT_DIR
    / "bdg2_electricity_hourly.parquet"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "bdg2_electricity_hourly_summary.csv"
)

BUILDING_SUMMARY_PATH = (
    OUTPUT_DIR
    / "bdg2_building_coverage.csv"
)


def normalize_metadata(
    metadata: pd.DataFrame,
) -> pd.DataFrame:

    required_columns = [
        "building_id",
        "site_id",
        "primaryspaceusage",
        "sqft",
        "timezone",
        "electricity",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in metadata.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required metadata columns: "
            f"{missing_columns}"
        )

    metadata = metadata[
        required_columns
    ].copy()

    metadata = metadata.rename(
        columns={
            "electricity":
                "electricity_available"
        }
    )

    metadata[
        "building_id"
    ] = (
        metadata[
            "building_id"
        ]
        .astype(
            "string"
        )
        .str.strip()
    )

    metadata[
        "site_id"
    ] = (
        metadata[
            "site_id"
        ]
        .astype(
            "string"
        )
        .str.strip()
    )

    metadata[
        "primaryspaceusage"
    ] = (
        metadata[
            "primaryspaceusage"
        ]
        .astype(
            "string"
        )
        .str.strip()
    )

    metadata[
        "timezone"
    ] = (
        metadata[
            "timezone"
        ]
        .astype(
            "string"
        )
        .str.strip()
    )

    metadata[
        "sqft"
    ] = pd.to_numeric(
        metadata[
            "sqft"
        ],
        errors="coerce",
    )

    return metadata


def load_meter_header() -> list[str]:

    header = pd.read_csv(
        ELECTRICITY_PATH,
        nrows=0,
    )

    meter_columns = [
        column
        for column in header.columns
        if column != "timestamp"
    ]

    return meter_columns


def build_metadata_subset(
    metadata: pd.DataFrame,
    meter_columns: list[str],
) -> pd.DataFrame:

    meter_set = set(
        meter_columns
    )

    subset = metadata[
        metadata[
            "building_id"
        ].isin(
            meter_set
        )
    ].copy()

    subset = subset[
        subset[
            "electricity_available"
        ].notna()
    ].copy()

    subset = subset.drop_duplicates(
        subset=[
            "building_id"
        ]
    )

    subset = subset.sort_values(
        "building_id"
    ).reset_index(
        drop=True
    )

    return subset


def read_and_reshape_meters(
    selected_buildings: list[str],
) -> pd.DataFrame:

    usecols = [
        "timestamp"
    ] + selected_buildings

    electricity = pd.read_csv(
        ELECTRICITY_PATH,
        usecols=usecols,
        low_memory=False,
    )

    electricity[
        "timestamp"
    ] = pd.to_datetime(
        electricity[
            "timestamp"
        ],
        errors="coerce",
    )

    invalid_timestamps = int(
        electricity[
            "timestamp"
        ].isna().sum()
    )

    if invalid_timestamps > 0:
        raise ValueError(
            f"Unparseable electricity timestamps: "
            f"{invalid_timestamps:,}"
        )

    electricity = electricity.melt(
        id_vars=[
            "timestamp"
        ],
        var_name="building_id",
        value_name="electricity",
    )

    electricity[
        "electricity"
    ] = pd.to_numeric(
        electricity[
            "electricity"
        ],
        errors="coerce",
    )

    return electricity


def build_building_coverage(
    data: pd.DataFrame,
) -> pd.DataFrame:

    coverage = (
        data
        .groupby(
            "building_id",
            observed=True,
        )
        .agg(
            total_rows=(
                "timestamp",
                "size",
            ),
            observed_rows=(
                "electricity",
                "count",
            ),
            minimum_timestamp=(
                "timestamp",
                "min",
            ),
            maximum_timestamp=(
                "timestamp",
                "max",
            ),
            minimum_electricity=(
                "electricity",
                "min",
            ),
            maximum_electricity=(
                "electricity",
                "max",
            ),
            mean_electricity=(
                "electricity",
                "mean",
            ),
        )
        .reset_index()
    )

    coverage[
        "missing_rows"
    ] = (
        coverage[
            "total_rows"
        ]
        - coverage[
            "observed_rows"
        ]
    )

    coverage[
        "missing_percentage"
    ] = (
        coverage[
            "missing_rows"
        ]
        / coverage[
            "total_rows"
        ]
        * 100.0
    )

    first_valid = (
        data[
            data[
                "electricity"
            ].notna()
        ]
        .groupby(
            "building_id",
            observed=True,
        )[
            "timestamp"
        ]
        .min()
        .rename(
            "first_valid_timestamp"
        )
    )

    last_valid = (
        data[
            data[
                "electricity"
            ].notna()
        ]
        .groupby(
            "building_id",
            observed=True,
        )[
            "timestamp"
        ]
        .max()
        .rename(
            "last_valid_timestamp"
        )
    )

    coverage = coverage.merge(
        first_valid,
        on="building_id",
        how="left",
    )

    coverage = coverage.merge(
        last_valid,
        on="building_id",
        how="left",
    )

    return coverage


def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading BDG2 metadata..."
    )

    metadata = pd.read_csv(
        METADATA_PATH,
        low_memory=False,
    )

    metadata = normalize_metadata(
        metadata
    )

    print(
        f"Metadata buildings: "
        f"{len(metadata):,}"
    )

    print(
        "Reading electricity meter header..."
    )

    meter_columns = (
        load_meter_header()
    )

    print(
        f"Electricity meter columns: "
        f"{len(meter_columns):,}"
    )

    metadata_subset = (
        build_metadata_subset(
            metadata,
            meter_columns,
        )
    )

    selected_buildings = (
        metadata_subset[
            "building_id"
        ]
        .astype(
            str
        )
        .tolist()
    )

    print(
        f"Matched electricity buildings: "
        f"{len(selected_buildings):,}"
    )

    unmatched_meter_columns = sorted(
        set(
            meter_columns
        )
        - set(
            selected_buildings
        )
    )

    unmatched_metadata_buildings = sorted(
        set(
            metadata[
                "building_id"
            ]
            .dropna()
            .astype(
                str
            )
        )
        - set(
            meter_columns
        )
    )

    print(
        f"Meter columns not retained: "
        f"{len(unmatched_meter_columns):,}"
    )

    print(
        f"Metadata buildings without "
        f"cleaned electricity column: "
        f"{len(unmatched_metadata_buildings):,}"
    )

    print(
        "Loading and reshaping cleaned "
        "electricity data..."
    )

    electricity = (
        read_and_reshape_meters(
            selected_buildings
        )
    )

    print(
        f"Long-format rows: "
        f"{len(electricity):,}"
    )

    data = electricity.merge(
        metadata_subset,
        on="building_id",
        how="left",
        validate="many_to_one",
    )

    duplicate_rows = int(
        data.duplicated(
            subset=[
                "building_id",
                "timestamp",
            ]
        ).sum()
    )

    if duplicate_rows > 0:
        raise ValueError(
            f"Duplicate building-hour rows: "
            f"{duplicate_rows:,}"
        )

    coverage = (
        build_building_coverage(
            data
        )
    )

    total_rows = len(
        data
    )

    missing_electricity = int(
        data[
            "electricity"
        ].isna().sum()
    )

    observed_rows = (
        total_rows
        - missing_electricity
    )

    summary = pd.DataFrame(
        [
            {
                "metadata_buildings":
                    len(
                        metadata
                    ),
                "meter_building_columns":
                    len(
                        meter_columns
                    ),
                "matched_buildings":
                    len(
                        selected_buildings
                    ),
                "rows_total":
                    total_rows,
                "rows_observed":
                    observed_rows,
                "rows_missing_electricity":
                    missing_electricity,
                "missing_electricity_percentage":
                    (
                        missing_electricity
                        / total_rows
                        * 100.0
                        if total_rows > 0
                        else np.nan
                    ),
                "minimum_timestamp":
                    data[
                        "timestamp"
                    ].min(),
                "maximum_timestamp":
                    data[
                        "timestamp"
                    ].max(),
                "duplicate_building_timestamps":
                    duplicate_rows,
                "sites":
                    data[
                        "site_id"
                    ].nunique(),
                "primary_space_usages":
                    data[
                        "primaryspaceusage"
                    ].nunique(
                        dropna=True
                    ),
                "missing_primaryspaceusage":
                    int(
                        data[
                            "primaryspaceusage"
                        ].isna()
                        .sum()
                    ),
                "missing_sqft":
                    int(
                        data[
                            "sqft"
                        ].isna()
                        .sum()
                    ),
            }
        ]
    )

    print(
        "Saving processed BDG2 "
        "electricity dataset..."
    )

    data.to_parquet(
        OUTPUT_PARQUET,
        index=False,
    )

    coverage.to_csv(
        BUILDING_SUMMARY_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print()

    print(
        "BDG2 preparation complete."
    )

    print()

    print(
        summary.to_string(
            index=False
        )
    )

    print()

    print(
        "Building coverage summary:"
    )

    print(
        coverage[
            [
                "observed_rows",
                "missing_percentage",
            ]
        ]
        .describe()
        .to_string()
    )

    print()

    print(
        f"Processed dataset: "
        f"{OUTPUT_PARQUET}"
    )

    print(
        f"Summary: "
        f"{SUMMARY_PATH}"
    )

    print(
        f"Building coverage: "
        f"{BUILDING_SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()