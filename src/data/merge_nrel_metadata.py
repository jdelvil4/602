from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

HOURLY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_modeling_hourly.parquet"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_resstock_buildings.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_modeling_base.parquet"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_modeling_base_summary.csv"
)

METADATA_COLUMNS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_modeling_metadata_columns.csv"
)


BUILDING_ID_COLUMN = "bldg_id"
TIMESTAMP_COLUMN = "timestamp"
TARGET_COLUMN = "electricity_total_hourly"
ROLE_COLUMN = "sample_role"

EXPECTED_BUILDINGS = 250
EXPECTED_DEVELOPMENT_BUILDINGS = 200
EXPECTED_HELD_OUT_BUILDINGS = 50
EXPECTED_ROWS = 2_190_000
EXPECTED_HOURS_PER_BUILDING = 8760


REQUIRED_METADATA_COLUMNS = [
    "in.geometry_building_type_recs",
    "in.sqft",
    "in.cec_climate_zone",
    "in.vintage",
]


OPTIONAL_METADATA_COLUMNS = [
    "in.county",
    "in.puma",
    "in.building_america_climate_zone",
    "in.geometry_floor_area",
    "in.heating_fuel",
    "in.hvac_heating_type_and_fuel",
    "in.hvac_cooling_type",
    "in.hvac_heating_type",
    "in.hvac_cooling_efficiency",
    "in.heating_setpoint",
    "in.cooling_setpoint",
    "in.insulation_wall",
    "in.insulation_roof",
    "in.windows",
    "in.occupants",
    "in.bedrooms",
    "in.geometry_stories",
]


def load_hourly_data() -> pd.DataFrame:

    if not HOURLY_PATH.exists():
        raise FileNotFoundError(
            f"Hourly modeling dataset not found: {HOURLY_PATH}"
        )

    hourly = pd.read_parquet(
        HOURLY_PATH
    )

    required_columns = {
        BUILDING_ID_COLUMN,
        ROLE_COLUMN,
        TIMESTAMP_COLUMN,
        TARGET_COLUMN,
    }

    missing_columns = (
        required_columns
        - set(hourly.columns)
    )

    if missing_columns:
        raise ValueError(
            "Hourly modeling dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    hourly[BUILDING_ID_COLUMN] = pd.to_numeric(
        hourly[BUILDING_ID_COLUMN],
        errors="coerce",
    )

    if hourly[BUILDING_ID_COLUMN].isna().any():
        raise ValueError(
            "Hourly modeling dataset contains invalid building IDs."
        )

    hourly[BUILDING_ID_COLUMN] = (
        hourly[BUILDING_ID_COLUMN]
        .astype("int64")
    )

    hourly[TIMESTAMP_COLUMN] = pd.to_datetime(
        hourly[TIMESTAMP_COLUMN]
    )

    hourly[ROLE_COLUMN] = (
        hourly[ROLE_COLUMN]
        .astype(str)
        .str.strip()
    )

    if len(hourly) != EXPECTED_ROWS:
        raise ValueError(
            f"Hourly dataset contains {len(hourly):,} rows; "
            f"expected {EXPECTED_ROWS:,}."
        )

    building_count = hourly[
        BUILDING_ID_COLUMN
    ].nunique()

    if building_count != EXPECTED_BUILDINGS:
        raise ValueError(
            f"Hourly dataset contains {building_count} buildings; "
            f"expected {EXPECTED_BUILDINGS}."
        )

    duplicate_count = int(
        hourly.duplicated(
            subset=[
                BUILDING_ID_COLUMN,
                TIMESTAMP_COLUMN,
            ]
        )
        .sum()
    )

    if duplicate_count != 0:
        raise ValueError(
            f"Hourly dataset contains {duplicate_count} "
            "duplicate building-timestamp rows."
        )

    return hourly


def load_metadata() -> pd.DataFrame:

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"California ResStock building metadata not found: "
            f"{METADATA_PATH}"
        )

    metadata = pd.read_csv(
        METADATA_PATH,
        low_memory=False,
    )

    if BUILDING_ID_COLUMN not in metadata.columns:
        raise ValueError(
            f"Metadata does not contain {BUILDING_ID_COLUMN}."
        )

    metadata[BUILDING_ID_COLUMN] = pd.to_numeric(
        metadata[BUILDING_ID_COLUMN],
        errors="coerce",
    )

    metadata = (
        metadata
        .dropna(
            subset=[
                BUILDING_ID_COLUMN,
            ]
        )
        .copy()
    )

    metadata[BUILDING_ID_COLUMN] = (
        metadata[BUILDING_ID_COLUMN]
        .astype("int64")
    )

    duplicate_count = int(
        metadata[
            BUILDING_ID_COLUMN
        ]
        .duplicated()
        .sum()
    )

    if duplicate_count != 0:
        raise ValueError(
            f"Metadata contains {duplicate_count} duplicate building IDs."
        )

    missing_required_metadata = (
        set(REQUIRED_METADATA_COLUMNS)
        - set(metadata.columns)
    )

    if missing_required_metadata:
        raise ValueError(
            "Metadata is missing required modeling columns: "
            f"{sorted(missing_required_metadata)}"
        )

    return metadata


def select_modeling_metadata(
    metadata: pd.DataFrame,
    modeling_building_ids: set[int],
) -> tuple[pd.DataFrame, list[str]]:

    available_optional_columns = [
        column
        for column in OPTIONAL_METADATA_COLUMNS
        if column in metadata.columns
    ]

    selected_columns = [
        BUILDING_ID_COLUMN,
        *REQUIRED_METADATA_COLUMNS,
        *available_optional_columns,
    ]

    selected_columns = list(
        dict.fromkeys(
            selected_columns
        )
    )

    selected_metadata = (
        metadata.loc[
            metadata[
                BUILDING_ID_COLUMN
            ].isin(
                modeling_building_ids
            ),
            selected_columns,
        ]
        .copy()
    )

    selected_building_ids = set(
        selected_metadata[
            BUILDING_ID_COLUMN
        ].tolist()
    )

    missing_buildings = (
        modeling_building_ids
        - selected_building_ids
    )

    if missing_buildings:
        raise ValueError(
            "Metadata could not be found for the following "
            f"modeling building IDs: {sorted(missing_buildings)}"
        )

    if len(selected_metadata) != EXPECTED_BUILDINGS:
        raise ValueError(
            f"Selected metadata contains {len(selected_metadata)} rows; "
            f"expected {EXPECTED_BUILDINGS}."
        )

    return (
        selected_metadata,
        selected_columns,
    )


def clean_selected_metadata(
    metadata: pd.DataFrame,
) -> pd.DataFrame:

    working = metadata.copy()

    working["in.sqft"] = pd.to_numeric(
        working["in.sqft"],
        errors="coerce",
    )

    working["floor_area_group"] = pd.cut(
        working["in.sqft"],
        bins=[
            0,
            1000,
            1500,
            2000,
            2500,
            3000,
            float("inf"),
        ],
        labels=[
            "<1000",
            "1000-1499",
            "1500-1999",
            "2000-2499",
            "2500-2999",
            "3000+",
        ],
        right=False,
        include_lowest=True,
    )

    working["floor_area_group"] = (
        working["floor_area_group"]
        .astype("string")
        .fillna("Unknown")
    )

    categorical_columns = [
        column
        for column in working.columns
        if column not in {
            BUILDING_ID_COLUMN,
            "in.sqft",
        }
    ]

    for column in categorical_columns:

        if column == "in.occupants":
            continue

        if pd.api.types.is_numeric_dtype(
            working[column]
        ):
            continue

        working[column] = (
            working[column]
            .astype("string")
            .fillna("Unknown")
            .str.strip()
            .replace(
                {
                    "": "Unknown",
                    "<NA>": "Unknown",
                    "nan": "Unknown",
                    "None": "Unknown",
                }
            )
        )

    return working


def merge_metadata(
    hourly: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:

    original_row_count = len(
        hourly
    )

    original_role_counts = (
        hourly[
            [
                BUILDING_ID_COLUMN,
                ROLE_COLUMN,
            ]
        ]
        .drop_duplicates()
        [ROLE_COLUMN]
        .value_counts()
        .to_dict()
    )

    merged = hourly.merge(
        metadata,
        on=BUILDING_ID_COLUMN,
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    if len(merged) != original_row_count:
        raise ValueError(
            f"Metadata merge changed row count from "
            f"{original_row_count:,} to {len(merged):,}."
        )

    merge_counts = (
        merged["_merge"]
        .value_counts()
    )

    unmatched_rows = int(
        merge_counts.get(
            "left_only",
            0,
        )
    )

    if unmatched_rows != 0:
        raise ValueError(
            f"Metadata merge left {unmatched_rows:,} "
            "hourly rows without matching building metadata."
        )

    merged = (
        merged
        .drop(
            columns=[
                "_merge",
            ]
        )
    )

    post_merge_role_counts = (
        merged[
            [
                BUILDING_ID_COLUMN,
                ROLE_COLUMN,
            ]
        ]
        .drop_duplicates()
        [ROLE_COLUMN]
        .value_counts()
        .to_dict()
    )

    if original_role_counts != post_merge_role_counts:
        raise ValueError(
            "Development/held-out building counts changed "
            "during metadata merge."
        )

    return merged


def validate_merged_data(
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
            f"Merged dataset contains {duplicate_count} "
            "duplicate building-timestamp rows."
        )

    core_missing_count = int(
        merged[
            [
                BUILDING_ID_COLUMN,
                ROLE_COLUMN,
                TIMESTAMP_COLUMN,
                TARGET_COLUMN,
            ]
        ]
        .isna()
        .sum()
        .sum()
    )

    if core_missing_count != 0:
        raise ValueError(
            f"Merged dataset contains {core_missing_count} "
            "missing values in core modeling fields."
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
            f"{EXPECTED_HOURS_PER_BUILDING:,} hourly rows.\n"
            f"{invalid_buildings.to_string()}"
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

    if development_buildings != EXPECTED_DEVELOPMENT_BUILDINGS:
        raise ValueError(
            f"Merged dataset contains {development_buildings} "
            "development buildings; expected "
            f"{EXPECTED_DEVELOPMENT_BUILDINGS}."
        )

    if held_out_buildings != EXPECTED_HELD_OUT_BUILDINGS:
        raise ValueError(
            f"Merged dataset contains {held_out_buildings} "
            "held-out buildings; expected "
            f"{EXPECTED_HELD_OUT_BUILDINGS}."
        )


def create_summary(
    merged: pd.DataFrame,
    metadata_columns: list[str],
) -> pd.DataFrame:

    metadata_feature_columns = [
        column
        for column in metadata_columns
        if column != BUILDING_ID_COLUMN
    ]

    if (
        "floor_area_group"
        in merged.columns
    ):
        metadata_feature_columns.append(
            "floor_area_group"
        )

    metadata_feature_columns = list(
        dict.fromkeys(
            metadata_feature_columns
        )
    )

    metadata_missing_cells = int(
        merged[
            metadata_feature_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    metadata_columns_with_missing = int(
        (
            merged[
                metadata_feature_columns
            ]
            .isna()
            .any()
        )
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
                "rows_total": len(
                    merged
                ),
                "minimum_timestamp": merged[
                    TIMESTAMP_COLUMN
                ].min(),
                "maximum_timestamp": merged[
                    TIMESTAMP_COLUMN
                ].max(),
                "core_missing_values": int(
                    merged[
                        [
                            BUILDING_ID_COLUMN,
                            ROLE_COLUMN,
                            TIMESTAMP_COLUMN,
                            TARGET_COLUMN,
                        ]
                    ]
                    .isna()
                    .sum()
                    .sum()
                ),
                "metadata_feature_count": len(
                    metadata_feature_columns
                ),
                "metadata_missing_cells": (
                    metadata_missing_cells
                ),
                "metadata_columns_with_missing": (
                    metadata_columns_with_missing
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


def create_metadata_column_report(
    merged: pd.DataFrame,
    metadata_columns: list[str],
) -> pd.DataFrame:

    feature_columns = [
        column
        for column in metadata_columns
        if column != BUILDING_ID_COLUMN
    ]

    if "floor_area_group" in merged.columns:
        feature_columns.append(
            "floor_area_group"
        )

    feature_columns = list(
        dict.fromkeys(
            feature_columns
        )
    )

    report_rows = []

    for column in feature_columns:

        missing_count = int(
            merged[
                column
            ]
            .isna()
            .sum()
        )

        missing_percentage = (
            missing_count
            / len(merged)
            * 100
        )

        unique_values = merged[
            column
        ].nunique(
            dropna=True
        )

        report_rows.append(
            {
                "column": column,
                "dtype": str(
                    merged[
                        column
                    ].dtype
                ),
                "unique_values": (
                    unique_values
                ),
                "missing_count": (
                    missing_count
                ),
                "missing_percentage": (
                    missing_percentage
                ),
            }
        )

    return pd.DataFrame(
        report_rows
    )


def main() -> None:

    print(
        "Loading NREL hourly modeling dataset..."
    )

    hourly = load_hourly_data()

    print(
        f"Hourly rows loaded: {len(hourly):,}"
    )

    print(
        f"Modeling buildings: "
        f"{hourly[BUILDING_ID_COLUMN].nunique():,}"
    )

    print()

    print(
        "Loading California ResStock building metadata..."
    )

    metadata = load_metadata()

    print(
        f"California metadata buildings available: "
        f"{len(metadata):,}"
    )

    modeling_building_ids = set(
        hourly[
            BUILDING_ID_COLUMN
        ]
        .unique()
        .tolist()
    )

    selected_metadata, metadata_columns = (
        select_modeling_metadata(
            metadata,
            modeling_building_ids,
        )
    )

    selected_metadata = clean_selected_metadata(
        selected_metadata
    )

    print()

    print(
        "Selected metadata columns:"
    )

    for column in selected_metadata.columns:
        if column != BUILDING_ID_COLUMN:
            print(
                f"  - {column}"
            )

    print()

    print(
        "Merging static building metadata..."
    )

    merged = merge_metadata(
        hourly,
        selected_metadata,
    )

    print(
        "Validating merged modeling dataset..."
    )

    validate_merged_data(
        merged
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Saving NREL modeling base dataset..."
    )

    merged.to_parquet(
        OUTPUT_PATH,
        index=False,
        compression="snappy",
    )

    summary = create_summary(
        merged,
        metadata_columns,
    )

    metadata_report = (
        create_metadata_column_report(
            merged,
            metadata_columns,
        )
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    metadata_report.to_csv(
        METADATA_COLUMNS_PATH,
        index=False,
    )

    print()

    print(
        "NREL metadata merge complete."
    )

    print()

    print(
        summary.to_string(
            index=False
        )
    )

    print()

    print(
        "Metadata column report:"
    )

    print(
        metadata_report.to_string(
            index=False
        )
    )

    print()

    print(
        f"Saved modeling base dataset to: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Saved summary to: "
        f"{SUMMARY_PATH}"
    )

    print(
        f"Saved metadata column report to: "
        f"{METADATA_COLUMNS_PATH}"
    )


if __name__ == "__main__":
    main()