from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROFILE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nrel"
    / "resstock_ca"
    / "upgrade=0"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_modeling_download_manifest.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_modeling_hourly.parquet"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_modeling_hourly_summary.csv"
)


TIMESTAMP_COLUMN = "timestamp"

ELECTRICITY_COLUMN = (
    "out.electricity.total.energy_consumption"
)

EXPECTED_RAW_ROWS = 35040
EXPECTED_HOURLY_ROWS = 8760
EXPECTED_SAMPLE_SIZE = 250

VALID_SAMPLE_ROLES = {
    "development",
    "held_out",
}


def load_manifest() -> pd.DataFrame:

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Modeling download manifest not found: {MANIFEST_PATH}"
        )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    required_columns = {
        "bldg_id",
        "sample_role",
        "filename",
        "status",
    }

    missing_columns = (
        required_columns
        - set(manifest.columns)
    )

    if missing_columns:
        raise ValueError(
            "Modeling download manifest is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    manifest["bldg_id"] = pd.to_numeric(
        manifest["bldg_id"],
        errors="coerce",
    )

    if manifest["bldg_id"].isna().any():
        raise ValueError(
            "Manifest contains invalid building IDs."
        )

    manifest["bldg_id"] = (
        manifest["bldg_id"]
        .astype("int64")
    )

    manifest["sample_role"] = (
        manifest["sample_role"]
        .astype("string")
        .str.strip()
    )

    invalid_roles = (
        set(
            manifest["sample_role"]
            .dropna()
            .unique()
        )
        - VALID_SAMPLE_ROLES
    )

    if invalid_roles:
        raise ValueError(
            "Manifest contains invalid sample roles: "
            f"{sorted(invalid_roles)}"
        )

    usable_statuses = {
        "downloaded",
        "already_exists",
    }

    manifest = (
        manifest.loc[
            manifest["status"].isin(
                usable_statuses
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    if len(manifest) != EXPECTED_SAMPLE_SIZE:
        raise ValueError(
            f"Usable manifest contains {len(manifest)} buildings; "
            f"expected {EXPECTED_SAMPLE_SIZE}."
        )

    duplicate_count = int(
        manifest[
            "bldg_id"
        ]
        .duplicated()
        .sum()
    )

    if duplicate_count != 0:
        raise ValueError(
            f"Duplicate building IDs found in manifest: {duplicate_count}"
        )

    development_count = int(
        (
            manifest[
                "sample_role"
            ]
            == "development"
        )
        .sum()
    )

    held_out_count = int(
        (
            manifest[
                "sample_role"
            ]
            == "held_out"
        )
        .sum()
    )

    if development_count != 200:
        raise ValueError(
            f"Development building count is {development_count}; "
            "expected 200."
        )

    if held_out_count != 50:
        raise ValueError(
            f"Held-out building count is {held_out_count}; "
            "expected 50."
        )

    return (
        manifest
        .sort_values(
            "bldg_id"
        )
        .reset_index(drop=True)
    )


def validate_raw_profile(
    df: pd.DataFrame,
    bldg_id: int,
) -> None:

    required_columns = {
        TIMESTAMP_COLUMN,
        ELECTRICITY_COLUMN,
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Building {bldg_id} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if len(df) != EXPECTED_RAW_ROWS:
        raise ValueError(
            f"Building {bldg_id} contains {len(df):,} raw rows; "
            f"expected {EXPECTED_RAW_ROWS:,}."
        )

    if df[
        TIMESTAMP_COLUMN
    ].isna().any():
        raise ValueError(
            f"Building {bldg_id} contains missing timestamps."
        )

    if df[
        ELECTRICITY_COLUMN
    ].isna().any():
        raise ValueError(
            f"Building {bldg_id} contains missing electricity values."
        )

    duplicate_count = int(
        df[
            TIMESTAMP_COLUMN
        ]
        .duplicated()
        .sum()
    )

    if duplicate_count != 0:
        raise ValueError(
            f"Building {bldg_id} contains "
            f"{duplicate_count} duplicate timestamps."
        )

    sorted_timestamps = (
        df[
            TIMESTAMP_COLUMN
        ]
        .sort_values()
        .reset_index(drop=True)
    )

    timestamp_differences = (
        sorted_timestamps
        .diff()
        .dropna()
    )

    expected_difference = pd.Timedelta(
        minutes=15
    )

    invalid_intervals = int(
        (
            timestamp_differences
            != expected_difference
        )
        .sum()
    )

    if invalid_intervals != 0:
        raise ValueError(
            f"Building {bldg_id} contains "
            f"{invalid_intervals} non-15-minute intervals."
        )


def aggregate_to_hourly(
    df: pd.DataFrame,
    bldg_id: int,
    sample_role: str,
) -> pd.DataFrame:

    working = (
        df.loc[
            :,
            [
                TIMESTAMP_COLUMN,
                ELECTRICITY_COLUMN,
            ],
        ]
        .copy()
    )

    working = (
        working
        .sort_values(
            TIMESTAMP_COLUMN
        )
        .set_index(
            TIMESTAMP_COLUMN
        )
    )

    hourly = (
        working
        .resample(
            "1h",
            closed="right",
            label="right",
        )
        .sum()
        .reset_index()
    )

    hourly = hourly.rename(
        columns={
            ELECTRICITY_COLUMN: (
                "electricity_total_hourly"
            )
        }
    )

    hourly.insert(
        0,
        "sample_role",
        sample_role,
    )

    hourly.insert(
        0,
        "bldg_id",
        bldg_id,
    )

    if len(hourly) != EXPECTED_HOURLY_ROWS:
        raise ValueError(
            f"Building {bldg_id} produced {len(hourly):,} hourly rows; "
            f"expected {EXPECTED_HOURLY_ROWS:,}."
        )

    if hourly[
        "electricity_total_hourly"
    ].isna().any():
        raise ValueError(
            f"Building {bldg_id} contains missing hourly electricity."
        )

    return hourly


def process_profile(
    path: Path,
    bldg_id: int,
    sample_role: str,
) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"Profile file not found: {path}"
        )

    df = pd.read_parquet(
        path,
        columns=[
            TIMESTAMP_COLUMN,
            ELECTRICITY_COLUMN,
        ],
    )

    df[
        TIMESTAMP_COLUMN
    ] = pd.to_datetime(
        df[
            TIMESTAMP_COLUMN
        ]
    )

    validate_raw_profile(
        df,
        bldg_id,
    )

    return aggregate_to_hourly(
        df,
        bldg_id,
        sample_role,
    )


def validate_pooled_dataset(
    pooled: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:

    expected_total_rows = (
        EXPECTED_SAMPLE_SIZE
        * EXPECTED_HOURLY_ROWS
    )

    if len(pooled) != expected_total_rows:
        raise ValueError(
            f"Pooled dataset contains {len(pooled):,} rows; "
            f"expected {expected_total_rows:,}."
        )

    building_count = pooled[
        "bldg_id"
    ].nunique()

    if building_count != EXPECTED_SAMPLE_SIZE:
        raise ValueError(
            f"Pooled dataset contains {building_count} unique buildings; "
            f"expected {EXPECTED_SAMPLE_SIZE}."
        )

    duplicate_count = int(
        pooled.duplicated(
            subset=[
                "bldg_id",
                TIMESTAMP_COLUMN,
            ]
        )
        .sum()
    )

    if duplicate_count != 0:
        raise ValueError(
            f"Pooled dataset contains {duplicate_count} "
            "duplicate building-timestamp rows."
        )

    missing_count = int(
        pooled
        .isna()
        .sum()
        .sum()
    )

    if missing_count != 0:
        raise ValueError(
            f"Pooled dataset contains {missing_count} missing values."
        )

    rows_per_building = (
        pooled.groupby(
            "bldg_id"
        )
        .size()
    )

    invalid_building_counts = (
        rows_per_building[
            rows_per_building
            != EXPECTED_HOURLY_ROWS
        ]
    )

    if not invalid_building_counts.empty:
        raise ValueError(
            "One or more buildings do not contain exactly "
            f"{EXPECTED_HOURLY_ROWS:,} hourly rows.\n"
            f"{invalid_building_counts.to_string()}"
        )

    pooled_roles = (
        pooled[
            [
                "bldg_id",
                "sample_role",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    manifest_roles = (
        manifest[
            [
                "bldg_id",
                "sample_role",
            ]
        ]
        .copy()
    )

    pooled_roles[
        "bldg_id"
    ] = (
        pooled_roles[
            "bldg_id"
        ]
        .astype("int64")
    )

    manifest_roles[
        "bldg_id"
    ] = (
        manifest_roles[
            "bldg_id"
        ]
        .astype("int64")
    )

    pooled_roles[
        "sample_role"
    ] = (
        pooled_roles[
            "sample_role"
        ]
        .astype(str)
        .str.strip()
    )

    manifest_roles[
        "sample_role"
    ] = (
        manifest_roles[
            "sample_role"
        ]
        .astype(str)
        .str.strip()
    )

    pooled_role_duplicates = int(
        pooled_roles[
            "bldg_id"
        ]
        .duplicated()
        .sum()
    )

    if pooled_role_duplicates != 0:
        raise ValueError(
            "One or more pooled buildings have multiple sample roles."
        )

    manifest_role_duplicates = int(
        manifest_roles[
            "bldg_id"
        ]
        .duplicated()
        .sum()
    )

    if manifest_role_duplicates != 0:
        raise ValueError(
            "One or more manifest buildings have multiple sample roles."
        )

    pooled_roles = (
        pooled_roles
        .sort_values(
            "bldg_id"
        )
        .reset_index(drop=True)
    )

    manifest_roles = (
        manifest_roles
        .sort_values(
            "bldg_id"
        )
        .reset_index(drop=True)
    )

    role_comparison = pooled_roles.merge(
        manifest_roles,
        on="bldg_id",
        how="outer",
        suffixes=(
            "_pooled",
            "_manifest",
        ),
        indicator=True,
    )

    role_mismatches = (
        role_comparison.loc[
            (
                role_comparison[
                    "_merge"
                ]
                != "both"
            )
            |
            (
                role_comparison[
                    "sample_role_pooled"
                ]
                != role_comparison[
                    "sample_role_manifest"
                ]
            )
        ]
        .copy()
    )

    if not role_mismatches.empty:
        raise ValueError(
            "Building sample roles in the pooled dataset "
            "do not match the modeling manifest.\n"
            f"{role_mismatches.to_string(index=False)}"
        )

    development_buildings = pooled.loc[
        pooled[
            "sample_role"
        ]
        == "development",
        "bldg_id",
    ].nunique()

    held_out_buildings = pooled.loc[
        pooled[
            "sample_role"
        ]
        == "held_out",
        "bldg_id",
    ].nunique()

    if development_buildings != 200:
        raise ValueError(
            f"Pooled dataset contains {development_buildings} "
            "development buildings; expected 200."
        )

    if held_out_buildings != 50:
        raise ValueError(
            f"Pooled dataset contains {held_out_buildings} "
            "held-out buildings; expected 50."
        )

def create_summary(
    pooled: pd.DataFrame,
) -> pd.DataFrame:

    development = (
        pooled[
            pooled[
                "sample_role"
            ]
            == "development"
        ]
    )

    held_out = (
        pooled[
            pooled[
                "sample_role"
            ]
            == "held_out"
        ]
    )

    summary = pd.DataFrame(
        [
            {
                "buildings_total": pooled[
                    "bldg_id"
                ].nunique(),
                "development_buildings": development[
                    "bldg_id"
                ].nunique(),
                "held_out_buildings": held_out[
                    "bldg_id"
                ].nunique(),
                "rows_total": len(
                    pooled
                ),
                "development_rows": len(
                    development
                ),
                "held_out_rows": len(
                    held_out
                ),
                "minimum_timestamp": pooled[
                    TIMESTAMP_COLUMN
                ].min(),
                "maximum_timestamp": pooled[
                    TIMESTAMP_COLUMN
                ].max(),
                "missing_values": int(
                    pooled.isna()
                    .sum()
                    .sum()
                ),
                "duplicate_building_timestamps": int(
                    pooled.duplicated(
                        subset=[
                            "bldg_id",
                            TIMESTAMP_COLUMN,
                        ]
                    )
                    .sum()
                ),
                "mean_hourly_electricity": pooled[
                    "electricity_total_hourly"
                ].mean(),
                "median_hourly_electricity": pooled[
                    "electricity_total_hourly"
                ].median(),
                "minimum_hourly_electricity": pooled[
                    "electricity_total_hourly"
                ].min(),
                "maximum_hourly_electricity": pooled[
                    "electricity_total_hourly"
                ].max(),
            }
        ]
    )

    return summary


def main() -> None:

    manifest = load_manifest()

    print(
        f"Modeling profiles to process: "
        f"{len(manifest):,}"
    )

    print()

    hourly_frames = []

    for position, row in enumerate(
        manifest.itertuples(
            index=False
        ),
        start=1,
    ):

        bldg_id = int(
            str(
                row.bldg_id
            )
        )

        sample_role = str(
            row.sample_role
        )

        filename = str(
            row.filename
        )

        path = (
            PROFILE_DIR
            / filename
        )

        print(
            f"[{position}/{len(manifest)}] "
            f"Processing building {bldg_id} "
            f"({sample_role})"
        )

        hourly = process_profile(
            path,
            bldg_id,
            sample_role,
        )

        hourly_frames.append(
            hourly
        )

    print()

    print(
        "Combining hourly profiles..."
    )

    pooled = pd.concat(
        hourly_frames,
        ignore_index=True,
    )

    pooled = (
        pooled
        .sort_values(
            [
                TIMESTAMP_COLUMN,
                "bldg_id",
            ]
        )
        .reset_index(drop=True)
    )

    print(
        "Validating pooled modeling dataset..."
    )

    validate_pooled_dataset(
        pooled,
        manifest,
    )

    summary = create_summary(
        pooled
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Saving pooled hourly dataset..."
    )

    pooled.to_parquet(
        OUTPUT_PATH,
        index=False,
        compression="snappy",
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print()

    print(
        "NREL modeling profile preparation complete."
    )

    print()

    print(
        summary.to_string(
            index=False
        )
    )

    print()

    print(
        f"Saved hourly modeling dataset to: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Saved summary to: "
        f"{SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()