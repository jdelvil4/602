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
    / "nrel_ca_pilot_download_manifest.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_pilot_hourly.parquet"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_pilot_hourly_summary.csv"
)


TIMESTAMP_COLUMN = "timestamp"

ELECTRICITY_COLUMN = (
    "out.electricity.total.energy_consumption"
)

EXPECTED_INTERVAL_MINUTES = 15
EXPECTED_RAW_ROWS = 35040
EXPECTED_HOURLY_ROWS = 8760


def load_manifest() -> pd.DataFrame:

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Pilot manifest not found: {MANIFEST_PATH}"
        )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    required_columns = {
        "bldg_id",
        "filename",
        "status",
    }

    missing_columns = (
        required_columns
        - set(manifest.columns)
    )

    if missing_columns:
        raise ValueError(
            "Pilot manifest is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    usable_statuses = {
        "downloaded",
        "already_exists",
    }

    manifest = (
        manifest.loc[
            manifest.loc[:, "status"].isin(
                usable_statuses
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    if manifest.empty:
        raise ValueError(
            "No usable pilot profiles were found in the manifest."
        )

    return manifest


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
            f"Building {bldg_id} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if len(df) != EXPECTED_RAW_ROWS:
        raise ValueError(
            f"Building {bldg_id} has {len(df):,} raw rows; "
            f"expected {EXPECTED_RAW_ROWS:,}."
        )

    if df.loc[:, TIMESTAMP_COLUMN].isna().any():
        raise ValueError(
            f"Building {bldg_id} has missing timestamps."
        )

    if df.loc[:, ELECTRICITY_COLUMN].isna().any():
        raise ValueError(
            f"Building {bldg_id} has missing total electricity values."
        )

    if df.loc[:, TIMESTAMP_COLUMN].duplicated().any():
        raise ValueError(
            f"Building {bldg_id} has duplicate timestamps."
        )

    timestamp_diff = (
        df.loc[:, TIMESTAMP_COLUMN]
        .sort_values()
        .diff()
        .dropna()
    )

    expected_diff = pd.Timedelta(
        minutes=EXPECTED_INTERVAL_MINUTES
    )

    invalid_intervals = int(
        (timestamp_diff != expected_diff)
        .sum()
    )

    if invalid_intervals != 0:
        raise ValueError(
            f"Building {bldg_id} contains "
            f"{invalid_intervals} non-15-minute timestamp gaps."
        )


def aggregate_to_hourly(
    df: pd.DataFrame,
    bldg_id: int,
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

    working.loc[:, TIMESTAMP_COLUMN] = (
        pd.to_datetime(
            working.loc[:, TIMESTAMP_COLUMN]
        )
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

    hourly.insert(
        0,
        "bldg_id",
        int(bldg_id),
    )

    hourly = hourly.rename(
        columns={
            ELECTRICITY_COLUMN: (
                "electricity_total_hourly"
            )
        }
    )

    if len(hourly) != EXPECTED_HOURLY_ROWS:
        raise ValueError(
            f"Building {bldg_id} produced "
            f"{len(hourly):,} hourly rows; "
            f"expected {EXPECTED_HOURLY_ROWS:,}."
        )

    if hourly.loc[
        :,
        "electricity_total_hourly",
    ].isna().any():
        raise ValueError(
            f"Building {bldg_id} has missing hourly electricity."
        )

    return hourly


def process_profile(
    path: Path,
    bldg_id: int,
) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"Profile file not found: {path}"
        )

    df = pd.read_parquet(
        path
    )

    if TIMESTAMP_COLUMN not in df.columns:
        raise ValueError(
            f"Building {bldg_id} does not contain "
            f"{TIMESTAMP_COLUMN}."
        )

    df.loc[:, TIMESTAMP_COLUMN] = (
        pd.to_datetime(
            df.loc[:, TIMESTAMP_COLUMN]
        )
    )

    validate_raw_profile(
        df,
        bldg_id,
    )

    hourly = aggregate_to_hourly(
        df,
        bldg_id,
    )

    return hourly


def create_summary(
    pooled_df: pd.DataFrame,
) -> pd.DataFrame:

    building_counts = (
        pooled_df
        .groupby(
            "bldg_id"
        )
        .size()
    )

    summary = pd.DataFrame(
        [
            {
                "buildings": pooled_df.loc[
                    :,
                    "bldg_id",
                ].nunique(),
                "rows": len(pooled_df),
                "minimum_timestamp": pooled_df.loc[
                    :,
                    TIMESTAMP_COLUMN,
                ].min(),
                "maximum_timestamp": pooled_df.loc[
                    :,
                    TIMESTAMP_COLUMN,
                ].max(),
                "minimum_rows_per_building": (
                    building_counts.min()
                ),
                "maximum_rows_per_building": (
                    building_counts.max()
                ),
                "mean_hourly_electricity": pooled_df.loc[
                    :,
                    "electricity_total_hourly",
                ].mean(),
                "median_hourly_electricity": pooled_df.loc[
                    :,
                    "electricity_total_hourly",
                ].median(),
                "minimum_hourly_electricity": pooled_df.loc[
                    :,
                    "electricity_total_hourly",
                ].min(),
                "maximum_hourly_electricity": pooled_df.loc[
                    :,
                    "electricity_total_hourly",
                ].max(),
                "missing_values": int(
                    pooled_df
                    .isna()
                    .sum()
                    .sum()
                ),
            }
        ]
    )

    return summary


def main() -> None:

    manifest = load_manifest()

    print(
        f"Pilot profiles to process: "
        f"{len(manifest):,}"
    )

    hourly_frames = []

    for position, (
        bldg_id_value,
        filename_value,
    ) in enumerate(
        manifest.loc[
            :,
            [
                "bldg_id",
                "filename",
            ],
        ].itertuples(
            index=False,
            name=None,
        ),
        start=1,
    ):

        bldg_id = int(
            str(
                bldg_id_value
            )
        )

        filename = str(
            filename_value
        )

        path = (
            PROFILE_DIR
            / filename
        )

        print(
            f"[{position}/{len(manifest)}] "
            f"Processing building {bldg_id}"
        )

        hourly = process_profile(
            path,
            bldg_id,
        )

        hourly_frames.append(
            hourly
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
            "Duplicate building-timestamp rows found "
            f"after pooling: {duplicate_count}"
        )

    expected_total_rows = (
        len(manifest)
        * EXPECTED_HOURLY_ROWS
    )

    if len(pooled) != expected_total_rows:
        raise ValueError(
            f"Pooled dataset contains {len(pooled):,} rows; "
            f"expected {expected_total_rows:,}."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pooled.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    summary = create_summary(
        pooled
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print()
    print(
        "NREL pilot hourly preparation complete."
    )

    print(
        f"Buildings: "
        f"{pooled.loc[:, 'bldg_id'].nunique():,}"
    )

    print(
        f"Hourly rows: "
        f"{len(pooled):,}"
    )

    print(
        f"Expected hourly rows: "
        f"{expected_total_rows:,}"
    )

    print(
        f"Timestamp range: "
        f"{pooled.loc[:, TIMESTAMP_COLUMN].min()} "
        f"to "
        f"{pooled.loc[:, TIMESTAMP_COLUMN].max()}"
    )

    print(
        f"Missing values: "
        f"{int(pooled.isna().sum().sum()):,}"
    )

    print()

    print(
        f"Saved hourly pilot dataset to: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Saved summary to: "
        f"{SUMMARY_PATH}"
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
            "Duplicate building-timestamp rows found "
            f"after pooling: {duplicate_count}"
        )

    expected_total_rows = (
        len(manifest)
        * EXPECTED_HOURLY_ROWS
    )

    if len(pooled) != expected_total_rows:
        raise ValueError(
            f"Pooled dataset contains {len(pooled):,} rows; "
            f"expected {expected_total_rows:,}."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pooled.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    summary = create_summary(
        pooled
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print()
    print(
        "NREL pilot hourly preparation complete."
    )

    print(
        f"Buildings: "
        f"{pooled.loc[:, 'bldg_id'].nunique():,}"
    )

    print(
        f"Hourly rows: "
        f"{len(pooled):,}"
    )

    print(
        f"Expected hourly rows: "
        f"{expected_total_rows:,}"
    )

    print(
        f"Timestamp range: "
        f"{pooled.loc[:, TIMESTAMP_COLUMN].min()} "
        f"to "
        f"{pooled.loc[:, TIMESTAMP_COLUMN].max()}"
    )

    print(
        f"Missing values: "
        f"{int(pooled.isna().sum().sum()):,}"
    )

    print()

    print(
        f"Saved hourly pilot dataset to: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Saved summary to: "
        f"{SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()