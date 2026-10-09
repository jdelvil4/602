from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "comstock_metadata.parquet"
)

COLUMN_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comstock_metadata_columns.csv"
)

SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comstock_metadata_inspection_summary.csv"
)


SEARCH_TERMS = [
    "state",
    "building_type",
    "building type",
    "floor_area",
    "floor area",
    "sqft",
    "county",
    "climate",
    "cec",
    "vintage",
    "year_built",
    "year built",
    "hvac",
    "heating",
    "cooling",
    "fuel",
    "stories",
    "occup",
    "puma",
    "cz",
]


def load_metadata() -> pd.DataFrame:

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"ComStock metadata not found: "
            f"{METADATA_PATH}"
        )

    print(
        f"Loading ComStock metadata from: "
        f"{METADATA_PATH}"
    )

    df = pd.read_parquet(
        METADATA_PATH
    )

    return df


def create_column_inventory(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for column in df.columns:

        series = df[
            column
        ]

        rows.append(
            {
                "column": column,
                "dtype": str(
                    series.dtype
                ),
                "non_null_count": int(
                    series.notna().sum()
                ),
                "missing_count": int(
                    series.isna().sum()
                ),
                "missing_pct": float(
                    series.isna().mean()
                    * 100
                ),
                "unique_count": int(
                    series.nunique(
                        dropna=True
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def find_candidate_columns(
    df: pd.DataFrame,
) -> list[str]:

    candidates = []

    for column in df.columns:

        column_lower = str(
            column
        ).lower()

        if any(
            term in column_lower
            for term in SEARCH_TERMS
        ):
            candidates.append(
                column
            )

    return candidates


def print_candidate_column_details(
    df: pd.DataFrame,
    candidates: list[str],
) -> None:

    print()

    print(
        "Potentially useful ComStock metadata columns:"
    )

    print()

    for column in candidates:

        series = df[
            column
        ]

        unique_count = int(
            series.nunique(
                dropna=True
            )
        )

        missing_count = int(
            series.isna().sum()
        )

        print(
            f"{column}"
        )

        print(
            f"  dtype: "
            f"{series.dtype}"
        )

        print(
            f"  unique values: "
            f"{unique_count:,}"
        )

        print(
            f"  missing values: "
            f"{missing_count:,}"
        )

        if unique_count <= 30:

            value_counts = (
                series
                .value_counts(
                    dropna=False
                )
                .head(
                    30
                )
            )

            print(
                "  value counts:"
            )

            for (
                value,
                count,
            ) in value_counts.items():

                print(
                    f"    {value}: "
                    f"{count:,}"
                )

        else:

            non_null = (
                series
                .dropna()
            )

            if pd.api.types.is_numeric_dtype(
                non_null
            ):

                print(
                    f"  min: "
                    f"{non_null.min()}"
                )

                print(
                    f"  median: "
                    f"{non_null.median()}"
                )

                print(
                    f"  max: "
                    f"{non_null.max()}"
                )

            else:

                print(
                    "  top values:"
                )

                value_counts = (
                    series
                    .value_counts(
                        dropna=False
                    )
                    .head(
                        10
                    )
                )

                for (
                    value,
                    count,
                ) in value_counts.items():

                    print(
                        f"    {value}: "
                        f"{count:,}"
                    )

        print()


def detect_state_columns(
    df: pd.DataFrame,
) -> list[str]:

    state_columns = []

    for column in df.columns:

        column_lower = str(
            column
        ).lower()

        if "state" not in column_lower:
            continue

        series = (
            df[
                column
            ]
            .dropna()
            .astype(
                str
            )
        )

        values = set(
            series
            .str.strip()
            .str.upper()
            .unique()
        )

        if (
            "CA" in values
            or "CALIFORNIA" in values
        ):
            state_columns.append(
                column
            )

    return state_columns


def summarize_state_columns(
    df: pd.DataFrame,
    state_columns: list[str],
) -> pd.DataFrame:

    rows = []

    for column in state_columns:

        normalized = (
            df[
                column
            ]
            .astype(
                "string"
            )
            .str.strip()
            .str.upper()
        )

        california_mask = (
            normalized.isin(
                [
                    "CA",
                    "CALIFORNIA",
                ]
            )
        )

        rows.append(
            {
                "state_column": column,
                "california_rows": int(
                    california_mask.sum()
                ),
                "total_rows": int(
                    len(
                        df
                    )
                ),
                "california_pct": float(
                    california_mask.mean()
                    * 100
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main() -> None:

    metadata = load_metadata()

    print()

    print(
        "ComStock metadata loaded."
    )

    print(
        f"Rows: "
        f"{len(metadata):,}"
    )

    print(
        f"Columns: "
        f"{len(metadata.columns):,}"
    )

    print()

    print(
        f"Index name: "
        f"{metadata.index.name}"
    )

    print(
        f"Index dtype: "
        f"{metadata.index.dtype}"
    )

    print(
        f"Duplicate index values: "
        f"{metadata.index.duplicated().sum():,}"
    )

    column_inventory = (
        create_column_inventory(
            metadata
        )
    )

    candidates = (
        find_candidate_columns(
            metadata
        )
    )

    print_candidate_column_details(
        metadata,
        candidates,
    )

    print(
        "Searching for state columns that "
        "contain California..."
    )

    state_columns = (
        detect_state_columns(
            metadata
        )
    )

    if state_columns:

        print(
            f"Candidate state columns: "
            f"{state_columns}"
        )

        state_summary = (
            summarize_state_columns(
                metadata,
                state_columns,
            )
        )

        print()

        print(
            state_summary.to_string(
                index=False
            )
        )

    else:

        print(
            "No state column containing "
            "CA or California was detected."
        )

        state_summary = pd.DataFrame(
            columns=[
                "state_column",
                "california_rows",
                "total_rows",
                "california_pct",
            ]
        )

    COLUMN_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    column_inventory.to_csv(
        COLUMN_OUTPUT_PATH,
        index=False,
    )

    state_summary.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )

    print()

    print(
        "ComStock metadata inspection complete."
    )

    print()

    print(
        f"Saved column inventory to: "
        f"{COLUMN_OUTPUT_PATH}"
    )

    print(
        f"Saved state summary to: "
        f"{SUMMARY_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()