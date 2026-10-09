from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROFILE_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nrel"
    / "comstock_ca"
    / "upgrade=0"
)

SAMPLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comstock_ca_modeling_sample.csv"
)


def find_profile() -> Path:

    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"Modeling sample not found: {SAMPLE_PATH}"
        )

    sample = pd.read_csv(
        SAMPLE_PATH
    )

    first_bldg_id = int(
        sample[
            "bldg_id"
        ].iloc[0]
    )

    profile_path = (
        PROFILE_DIRECTORY
        / f"{first_bldg_id}-0.parquet"
    )

    if not profile_path.exists():
        raise FileNotFoundError(
            f"Profile not found: {profile_path}"
        )

    return profile_path


def inspect_parquet_schema(
    profile_path: Path,
) -> None:

    parquet_file = pq.ParquetFile(
        profile_path
    )

    metadata = parquet_file.metadata

    print(
        "Parquet metadata:"
    )

    print(
        f"Rows: {metadata.num_rows:,}"
    )

    print(
        f"Columns: {metadata.num_columns:,}"
    )

    print(
        f"Row groups: {metadata.num_row_groups:,}"
    )

    print()

    print(
        "Parquet schema:"
    )

    print(
        parquet_file.schema
    )


def inspect_dataframe(
    profile_path: Path,
) -> None:

    df = pd.read_parquet(
        profile_path
    )

    print()

    print(
        "DataFrame shape:"
    )

    print(
        df.shape
    )

    print()

    print(
        "Columns:"
    )

    for column in df.columns:

        print(
            f"  {column}"
        )

    print()

    print(
        "Dtypes:"
    )

    print(
        df.dtypes.to_string()
    )

    print()

    print(
        "First five rows:"
    )

    print(
        df.head().to_string()
    )

    print()

    print(
        "Missing values by column:"
    )

    missing = (
        df
        .isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    print(
        missing.to_string()
    )

    print()

    print(
        "Candidate timestamp columns:"
    )

    timestamp_candidates = []

    for column in df.columns:

        column_lower = (
            str(
                column
            )
            .lower()
        )

        if (
            "time" in column_lower
            or "date" in column_lower
        ):
            timestamp_candidates.append(
                column
            )

    if timestamp_candidates:

        for column in timestamp_candidates:

            print(
                f"  {column}"
            )

            parsed = pd.to_datetime(
                df[
                    column
                ],
                errors="coerce",
            )

            valid = parsed.dropna()

            if valid.empty:

                print(
                    "    No valid parsed timestamps."
                )

                continue

            print(
                f"    Min: {valid.min()}"
            )

            print(
                f"    Max: {valid.max()}"
            )

            print(
                f"    Valid timestamps: "
                f"{len(valid):,}"
            )

            if len(
                valid
            ) > 1:

                differences = (
                    valid
                    .sort_values()
                    .diff()
                    .dropna()
                )

                print(
                    "    Most common interval:"
                )

                print(
                    differences
                    .value_counts()
                    .head(
                        10
                    )
                    .to_string()
                )

    else:

        print(
            "  None detected."
        )

    print()

    print(
        "Candidate electricity columns:"
    )

    electricity_candidates = []

    for column in df.columns:

        column_lower = (
            str(
                column
            )
            .lower()
        )

        if (
            "electric" in column_lower
            or "energy" in column_lower
            or "load" in column_lower
        ):

            electricity_candidates.append(
                column
            )

    if electricity_candidates:

        for column in electricity_candidates:

            print(
                f"  {column}"
            )

    else:

        print(
            "  None detected."
        )


def main() -> None:

    profile_path = find_profile()

    print(
        f"Inspecting ComStock profile:"
    )

    print(
        profile_path
    )

    print()

    inspect_parquet_schema(
        profile_path
    )

    inspect_dataframe(
        profile_path
    )

    print()

    print(
        "ComStock profile inspection complete."
    )


if __name__ == "__main__":
    main()