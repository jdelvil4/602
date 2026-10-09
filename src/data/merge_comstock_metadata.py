from pathlib import Path
from time import perf_counter

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

HOURLY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comstock_ca_modeling_hourly.parquet"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comstock_ca_buildings.csv"
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
    / "comstock_ca_modeling_base.parquet"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comstock_ca_modeling_base_summary.csv"
)

COLUMN_LIST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comstock_ca_modeling_base_columns.csv"
)


BUILDING_ID_COLUMN = "bldg_id"
ROLE_COLUMN = "sample_role"
TIMESTAMP_COLUMN = "timestamp"
TARGET_COLUMN = "electricity_consumption"

SELECTED_METADATA_COLUMNS = [
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

AUXILIARY_METADATA_COLUMNS = [
    "in.county",
    "in.resstock_county_id",
    "in.state_abbreviation",
    "in.state_name",
]

EXPECTED_BUILDINGS = 250
EXPECTED_ROWS = 2_190_000
EXPECTED_ROWS_PER_BUILDING = 8_760


def load_hourly() -> pd.DataFrame:

    if not HOURLY_PATH.exists():
        raise FileNotFoundError(
            f"Hourly ComStock dataset not found: {HOURLY_PATH}"
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
            "Hourly ComStock dataset is missing "
            f"required columns: {sorted(missing_columns)}"
        )

    return hourly


def load_metadata() -> pd.DataFrame:

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"ComStock building metadata not found: {METADATA_PATH}"
        )

    metadata = pd.read_csv(
        METADATA_PATH
    )

    required_columns = {
        BUILDING_ID_COLUMN,
        *SELECTED_METADATA_COLUMNS,
        *AUXILIARY_METADATA_COLUMNS,
    }

    missing_columns = (
        required_columns
        - set(metadata.columns)
    )

    if missing_columns:
        raise ValueError(
            "ComStock metadata is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if metadata[
        BUILDING_ID_COLUMN
    ].duplicated().any():
        raise ValueError(
            "ComStock metadata contains duplicate building IDs."
        )

    return metadata


def load_sample_ids() -> set[int]:

    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"ComStock sample file not found: {SAMPLE_PATH}"
        )

    sample = pd.read_csv(
        SAMPLE_PATH,
        usecols=[
            BUILDING_ID_COLUMN,
        ],
    )

    sample_ids = set(
        sample[
            BUILDING_ID_COLUMN
        ]
        .astype(
            int
        )
        .tolist()
    )

    if len(
        sample_ids
    ) != EXPECTED_BUILDINGS:
        raise ValueError(
            f"Expected {EXPECTED_BUILDINGS} unique sample IDs "
            f"but found {len(sample_ids)}."
        )

    return sample_ids


def filter_metadata_to_sample(
    metadata: pd.DataFrame,
    sample_ids: set[int],
) -> pd.DataFrame:

    filtered = metadata.loc[
        metadata[
            BUILDING_ID_COLUMN
        ]
        .isin(
            sample_ids
        )
    ].copy()

    if len(
        filtered
    ) != EXPECTED_BUILDINGS:
        raise ValueError(
            f"Expected metadata for {EXPECTED_BUILDINGS} "
            f"sample buildings but found {len(filtered)}."
        )

    return filtered


def validate_hourly(
    hourly: pd.DataFrame,
    sample_ids: set[int],
) -> None:

    if len(
        hourly
    ) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} hourly rows "
            f"but found {len(hourly):,}."
        )

    hourly_ids = set(
        hourly[
            BUILDING_ID_COLUMN
        ]
        .astype(
            int
        )
        .unique()
        .tolist()
    )

    if hourly_ids != sample_ids:
        missing_from_hourly = sorted(
            sample_ids
            - hourly_ids
        )

        unexpected_hourly = sorted(
            hourly_ids
            - sample_ids
        )

        raise ValueError(
            "Hourly building IDs do not match sample IDs. "
            f"Missing: {missing_from_hourly}. "
            f"Unexpected: {unexpected_hourly}."
        )

    rows_per_building = (
        hourly
        .groupby(
            BUILDING_ID_COLUMN
        )
        .size()
    )

    invalid_counts = rows_per_building.loc[
        rows_per_building
        != EXPECTED_ROWS_PER_BUILDING
    ]

    if not invalid_counts.empty:
        raise ValueError(
            "Some buildings do not contain exactly "
            f"{EXPECTED_ROWS_PER_BUILDING:,} hourly rows."
        )


def build_metadata_table(
    metadata: pd.DataFrame,
) -> pd.DataFrame:

    columns = [
        BUILDING_ID_COLUMN,
        *SELECTED_METADATA_COLUMNS,
        *AUXILIARY_METADATA_COLUMNS,
    ]

    return (
        metadata[
            columns
        ]
        .copy()
    )


def merge_metadata(
    hourly: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:

    merged = hourly.merge(
        metadata,
        on=BUILDING_ID_COLUMN,
        how="left",
        validate="many_to_one",
    )

    return merged


def validate_merged(
    merged: pd.DataFrame,
) -> None:

    if len(
        merged
    ) != EXPECTED_ROWS:
        raise ValueError(
            f"Merged dataset contains {len(merged):,} rows; "
            f"expected {EXPECTED_ROWS:,}."
        )

    if merged[
        BUILDING_ID_COLUMN
    ].nunique() != EXPECTED_BUILDINGS:
        raise ValueError(
            "Merged dataset does not contain "
            f"{EXPECTED_BUILDINGS} unique buildings."
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
            f"Merged dataset contains {duplicate_count:,} "
            "duplicate building-timestamp rows."
        )

    selected_missing = (
        merged[
            SELECTED_METADATA_COLUMNS
        ]
        .isna()
        .sum()
    )

    if int(
        selected_missing.sum()
    ) != 0:
        print()
        print(
            "WARNING: selected metadata contains missing values:"
        )
        print(
            selected_missing.loc[
                selected_missing > 0
            ].to_string()
        )

    outcome_columns = [
        column
        for column in merged.columns
        if str(
            column
        ).startswith(
            "out."
        )
    ]

    if outcome_columns:
        raise ValueError(
            "Merged modeling base unexpectedly contains "
            f"out.* columns: {outcome_columns}"
        )


def create_summary(
    merged: pd.DataFrame,
) -> pd.DataFrame:

    summary_rows = [
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
            "metric": "development_buildings",
            "value": merged.loc[
                merged[
                    ROLE_COLUMN
                ]
                == "development",
                BUILDING_ID_COLUMN,
            ].nunique(),
        },
        {
            "metric": "held_out_buildings",
            "value": merged.loc[
                merged[
                    ROLE_COLUMN
                ]
                == "held_out",
                BUILDING_ID_COLUMN,
            ].nunique(),
        },
        {
            "metric": "rows_per_building",
            "value": int(
                merged.groupby(
                    BUILDING_ID_COLUMN
                )
                .size()
                .median()
            ),
        },
        {
            "metric": "first_timestamp",
            "value": str(
                merged[
                    TIMESTAMP_COLUMN
                ].min()
            ),
        },
        {
            "metric": "last_timestamp",
            "value": str(
                merged[
                    TIMESTAMP_COLUMN
                ].max()
            ),
        },
        {
            "metric": "modeling_metadata_columns",
            "value": len(
                SELECTED_METADATA_COLUMNS
            ),
        },
        {
            "metric": "auxiliary_metadata_columns",
            "value": len(
                AUXILIARY_METADATA_COLUMNS
            ),
        },
        {
            "metric": "total_output_columns",
            "value": len(
                merged.columns
            ),
        },
        {
            "metric": "outcome_metadata_columns",
            "value": len(
                [
                    column
                    for column in merged.columns
                    if str(
                        column
                    ).startswith(
                        "out."
                    )
                ]
            ),
        },
        {
            "metric": "missing_selected_metadata_values",
            "value": int(
                merged[
                    SELECTED_METADATA_COLUMNS
                ]
                .isna()
                .sum()
                .sum()
            ),
        },
    ]

    return pd.DataFrame(
        summary_rows
    )


def create_column_table(
    merged: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for column in merged.columns:

        if column in {
            BUILDING_ID_COLUMN,
            ROLE_COLUMN,
            TIMESTAMP_COLUMN,
            TARGET_COLUMN,
        }:
            category = "core"

        elif column in SELECTED_METADATA_COLUMNS:
            category = "modeling_metadata"

        elif column in AUXILIARY_METADATA_COLUMNS:
            category = "auxiliary_metadata"

        else:
            category = "other"

        rows.append(
            {
                "column": column,
                "dtype": str(
                    merged[
                        column
                    ].dtype
                ),
                "category": category,
                "missing_values": int(
                    merged[
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
        "Loading hourly ComStock dataset..."
    )

    hourly = load_hourly()

    print(
        f"Hourly rows: {len(hourly):,}"
    )

    print(
        f"Hourly buildings: "
        f"{hourly[BUILDING_ID_COLUMN].nunique():,}"
    )

    print()

    print(
        "Loading California ComStock metadata..."
    )

    metadata = load_metadata()

    print(
        f"Metadata buildings: {len(metadata):,}"
    )

    print()

    print(
        "Loading modeling-sample IDs..."
    )

    sample_ids = load_sample_ids()

    print(
        f"Sample IDs: {len(sample_ids):,}"
    )

    print()

    print(
        "Validating hourly panel..."
    )

    validate_hourly(
        hourly,
        sample_ids,
    )

    print(
        "Filtering metadata to sampled buildings..."
    )

    metadata = filter_metadata_to_sample(
        metadata,
        sample_ids,
    )

    modeling_metadata = build_metadata_table(
        metadata
    )

    print(
        f"Metadata columns retained: "
        f"{len(modeling_metadata.columns) - 1}"
    )

    print()

    print(
        "Merging metadata onto hourly panel..."
    )

    start = perf_counter()

    merged = merge_metadata(
        hourly,
        modeling_metadata,
    )

    validate_merged(
        merged
    )

    summary = create_summary(
        merged
    )

    column_table = create_column_table(
        merged
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Saving ComStock modeling base..."
    )

    merged.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    column_table.to_csv(
        COLUMN_LIST_PATH,
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
        "ComStock metadata merge complete."
    )

    print()

    print(
        f"Rows: {len(merged):,}"
    )

    print(
        f"Buildings: "
        f"{merged[BUILDING_ID_COLUMN].nunique():,}"
    )

    print(
        f"Total columns: "
        f"{len(merged.columns):,}"
    )

    print(
        f"Selected modeling metadata columns: "
        f"{len(SELECTED_METADATA_COLUMNS):,}"
    )

    print(
        f"Auxiliary metadata columns: "
        f"{len(AUXILIARY_METADATA_COLUMNS):,}"
    )

    print(
        f"Missing selected metadata values: "
        f"{int(merged[SELECTED_METADATA_COLUMNS].isna().sum().sum()):,}"
    )

    print(
        f"Duplicate building timestamps: "
        f"{int(merged.duplicated(subset=[BUILDING_ID_COLUMN, TIMESTAMP_COLUMN]).sum()):,}"
    )

    print(
        f"Output-derived metadata columns: "
        f"{len([column for column in merged.columns if str(column).startswith('out.')]):,}"
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
        f"Saved modeling base to: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Saved summary to: "
        f"{SUMMARY_PATH}"
    )

    print(
        f"Saved column table to: "
        f"{COLUMN_LIST_PATH}"
    )


if __name__ == "__main__":
    main()