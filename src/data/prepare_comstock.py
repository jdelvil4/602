from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "comstock_metadata.parquet"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

CA_METADATA_PATH = (
    OUTPUT_DIRECTORY
    / "comstock_ca_metadata.parquet"
)

CA_BUILDINGS_PATH = (
    OUTPUT_DIRECTORY
    / "comstock_ca_buildings.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "comstock_ca_summary.csv"
)

BUILDING_TYPE_PATH = (
    OUTPUT_DIRECTORY
    / "comstock_ca_building_type_distribution.csv"
)

ASHRAE_ZONE_PATH = (
    OUTPUT_DIRECTORY
    / "comstock_ca_ashrae_zone_distribution.csv"
)

STRATA_PATH = (
    OUTPUT_DIRECTORY
    / "comstock_ca_building_type_ashrae_strata.csv"
)

HEATING_FUEL_PATH = (
    OUTPUT_DIRECTORY
    / "comstock_ca_heating_fuel_distribution.csv"
)

HVAC_PATH = (
    OUTPUT_DIRECTORY
    / "comstock_ca_hvac_distribution.csv"
)

NUMERIC_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "comstock_ca_numeric_summary.csv"
)


BUILDING_ID_COLUMN = "bldg_id"

STATE_COLUMN = "in.state_abbreviation"
STATE_NAME_COLUMN = "in.state_name"

COUNTY_COLUMN = "in.county"
COUNTY_NAME_COLUMN = "in.resstock_county_id"

BUILDING_TYPE_COLUMN = "in.building_type"

SQFT_COLUMN = "in.sqft"

STORIES_COLUMN = "in.number_of_stories"

HVAC_COLUMN = "in.hvac_system_type"

HEATING_FUEL_COLUMN = "in.heating_fuel"

SERVICE_WATER_HEATING_FUEL_COLUMN = (
    "in.service_water_heating_fuel"
)

ASHRAE_ZONE_COLUMN = (
    "in.climate_zone_ashrae_2004"
)

BUILDING_AMERICA_ZONE_COLUMN = (
    "in.climate_zone_building_america"
)

OCCUPANT_DENSITY_COLUMN = (
    "stat.occupant_density_ppl_per_m_2"
)

EXPECTED_CA_ROWS = 40048


COMPACT_COLUMNS = [
    BUILDING_ID_COLUMN,
    STATE_COLUMN,
    STATE_NAME_COLUMN,
    COUNTY_COLUMN,
    COUNTY_NAME_COLUMN,
    BUILDING_TYPE_COLUMN,
    SQFT_COLUMN,
    STORIES_COLUMN,
    HVAC_COLUMN,
    HEATING_FUEL_COLUMN,
    SERVICE_WATER_HEATING_FUEL_COLUMN,
    ASHRAE_ZONE_COLUMN,
    BUILDING_AMERICA_ZONE_COLUMN,
    OCCUPANT_DENSITY_COLUMN,
]


def load_metadata() -> pd.DataFrame:

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"ComStock metadata not found: "
            f"{INPUT_PATH}"
        )

    print(
        f"Loading ComStock metadata from: "
        f"{INPUT_PATH}"
    )

    df = pd.read_parquet(
        INPUT_PATH
    )

    if (
        BUILDING_ID_COLUMN
        not in df.columns
    ):

        if (
            df.index.name
            == BUILDING_ID_COLUMN
        ):

            df = (
                df
                .reset_index()
            )

        else:

            raise ValueError(
                "ComStock metadata does not contain "
                "bldg_id as either a column or index."
            )

    return df


def validate_source_metadata(
    df: pd.DataFrame,
) -> None:

    required_columns = set(
        COMPACT_COLUMNS
    )

    missing_columns = (
        required_columns
        - set(
            df.columns
        )
    )

    if missing_columns:

        raise ValueError(
            "ComStock metadata is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    duplicate_ids = int(
        df[
            BUILDING_ID_COLUMN
        ]
        .duplicated()
        .sum()
    )

    if duplicate_ids != 0:

        raise ValueError(
            f"ComStock metadata contains "
            f"{duplicate_ids:,} duplicate building IDs."
        )


def filter_california(
    df: pd.DataFrame,
) -> pd.DataFrame:

    state_values = (
        df[
            STATE_COLUMN
        ]
        .astype(
            "string"
        )
        .str.strip()
        .str.upper()
    )

    ca = (
        df.loc[
            state_values
            == "CA"
        ]
        .copy()
    )

    ca = (
        ca
        .sort_values(
            BUILDING_ID_COLUMN
        )
        .reset_index(
            drop=True
        )
    )

    return ca


def validate_california(
    ca: pd.DataFrame,
) -> None:

    print()

    print(
        "Validating California subset..."
    )

    if len(
        ca
    ) != EXPECTED_CA_ROWS:

        raise ValueError(
            f"Expected "
            f"{EXPECTED_CA_ROWS:,} California buildings "
            f"but found "
            f"{len(ca):,}."
        )

    duplicate_ids = int(
        ca[
            BUILDING_ID_COLUMN
        ]
        .duplicated()
        .sum()
    )

    if duplicate_ids != 0:

        raise ValueError(
            f"California subset contains "
            f"{duplicate_ids:,} duplicate building IDs."
        )

    state_values = set(
        ca[
            STATE_COLUMN
        ]
        .astype(
            str
        )
        .str.strip()
        .str.upper()
        .unique()
    )

    if state_values != {
        "CA"
    }:

        raise ValueError(
            "California subset contains unexpected "
            f"state values: "
            f"{sorted(state_values)}"
        )

    print(
        f"California buildings: "
        f"{len(ca):,}"
    )

    print(
        f"Duplicate building IDs: "
        f"{duplicate_ids:,}"
    )

    print(
        "State validation: CA only"
    )


def create_compact_building_index(
    ca: pd.DataFrame,
) -> pd.DataFrame:

    compact = (
        ca[
            COMPACT_COLUMNS
        ]
        .copy()
    )

    return compact


def create_distribution(
    df: pd.DataFrame,
    column: str,
) -> pd.DataFrame:

    output = (
        df[
            column
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            column
        )
        .reset_index(
            name="buildings"
        )
    )

    output[
        "pct_of_ca_sample"
    ] = (
        output[
            "buildings"
        ]
        / len(
            df
        )
        * 100
    )

    return output


def create_strata_table(
    df: pd.DataFrame,
) -> pd.DataFrame:

    strata = (
        df
        .groupby(
            [
                BUILDING_TYPE_COLUMN,
                ASHRAE_ZONE_COLUMN,
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="buildings"
        )
    )

    strata[
        "pct_of_ca_sample"
    ] = (
        strata[
            "buildings"
        ]
        / len(
            df
        )
        * 100
    )

    strata = (
        strata
        .sort_values(
            [
                BUILDING_TYPE_COLUMN,
                "buildings",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    return strata


def create_numeric_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:

    numeric_columns = [
        SQFT_COLUMN,
        STORIES_COLUMN,
        OCCUPANT_DENSITY_COLUMN,
    ]

    rows = []

    for column in numeric_columns:

        values = (
            pd.to_numeric(
                df[
                    column
                ],
                errors="coerce",
            )
            .dropna()
            .to_numpy(
                dtype=float
            )
        )

        if len(
            values
        ) == 0:

            continue

        rows.append(
            {
                "feature": column,
                "n": int(
                    len(
                        values
                    )
                ),
                "missing": int(
                    df[
                        column
                    ]
                    .isna()
                    .sum()
                ),
                "mean": float(
                    np.mean(
                        values
                    )
                ),
                "median": float(
                    np.median(
                        values
                    )
                ),
                "std": float(
                    np.std(
                        values,
                        ddof=0,
                    )
                ),
                "min": float(
                    np.min(
                        values
                    )
                ),
                "p25": float(
                    np.percentile(
                        values,
                        25,
                    )
                ),
                "p75": float(
                    np.percentile(
                        values,
                        75,
                    )
                ),
                "max": float(
                    np.max(
                        values
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def create_summary(
    source: pd.DataFrame,
    ca: pd.DataFrame,
    compact: pd.DataFrame,
    strata: pd.DataFrame,
) -> pd.DataFrame:

    output_columns = [
        column
        for column in source.columns
        if str(
            column
        ).startswith(
            "out."
        )
    ]

    compact_output_columns = [
        column
        for column in compact.columns
        if str(
            column
        ).startswith(
            "out."
        )
    ]

    rows = [
        {
            "metric": "source_rows",
            "value": len(
                source
            ),
        },
        {
            "metric": "source_columns",
            "value": len(
                source.columns
            ),
        },
        {
            "metric": "california_rows",
            "value": len(
                ca
            ),
        },
        {
            "metric": "california_pct_of_source",
            "value": (
                len(
                    ca
                )
                / len(
                    source
                )
                * 100
            ),
        },
        {
            "metric": "unique_building_types",
            "value": ca[
                BUILDING_TYPE_COLUMN
            ].nunique(
                dropna=True
            ),
        },
        {
            "metric": "unique_ashrae_zones",
            "value": ca[
                ASHRAE_ZONE_COLUMN
            ].nunique(
                dropna=True
            ),
        },
        {
            "metric": "unique_counties",
            "value": ca[
                COUNTY_COLUMN
            ].nunique(
                dropna=True
            ),
        },
        {
            "metric": "building_type_ashrae_strata",
            "value": len(
                strata
            ),
        },
        {
            "metric": "source_out_columns",
            "value": len(
                output_columns
            ),
        },
        {
            "metric": "compact_out_columns",
            "value": len(
                compact_output_columns
            ),
        },
        {
            "metric": "compact_columns",
            "value": len(
                compact.columns
            ),
        },
    ]

    return pd.DataFrame(
        rows
    )


def print_distribution(
    title: str,
    df: pd.DataFrame,
) -> None:

    print()

    print(
        title
    )

    print(
        df.to_string(
            index=False
        )
    )


def main() -> None:

    metadata = load_metadata()

    print()

    print(
        f"Source rows: "
        f"{len(metadata):,}"
    )

    print(
        f"Source columns: "
        f"{len(metadata.columns):,}"
    )

    print()

    print(
        "Validating source metadata..."
    )

    validate_source_metadata(
        metadata
    )

    print(
        "Filtering ComStock metadata to California..."
    )

    ca = filter_california(
        metadata
    )

    validate_california(
        ca
    )

    print()

    print(
        "Creating compact California "
        "commercial-building index..."
    )

    compact = (
        create_compact_building_index(
            ca
        )
    )

    leakage_columns = [
        column
        for column in compact.columns
        if str(
            column
        ).startswith(
            "out."
        )
    ]

    if leakage_columns:

        raise ValueError(
            "Compact building index unexpectedly "
            "contains output columns: "
            f"{leakage_columns}"
        )

    print(
        f"Compact columns: "
        f"{len(compact.columns)}"
    )

    print(
        "Output-derived metadata columns in "
        "compact index: 0"
    )

    print()

    print(
        "Creating California metadata summaries..."
    )

    building_type_distribution = (
        create_distribution(
            ca,
            BUILDING_TYPE_COLUMN,
        )
    )

    ashrae_distribution = (
        create_distribution(
            ca,
            ASHRAE_ZONE_COLUMN,
        )
    )

    heating_fuel_distribution = (
        create_distribution(
            ca,
            HEATING_FUEL_COLUMN,
        )
    )

    hvac_distribution = (
        create_distribution(
            ca,
            HVAC_COLUMN,
        )
    )

    strata = create_strata_table(
        ca
    )

    numeric_summary = (
        create_numeric_summary(
            ca
        )
    )

    summary = create_summary(
        metadata,
        ca,
        compact,
        strata,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()

    print(
        "Saving California ComStock metadata..."
    )

    ca.to_parquet(
        CA_METADATA_PATH,
        index=False,
        compression="snappy",
    )

    compact.to_csv(
        CA_BUILDINGS_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    building_type_distribution.to_csv(
        BUILDING_TYPE_PATH,
        index=False,
    )

    ashrae_distribution.to_csv(
        ASHRAE_ZONE_PATH,
        index=False,
    )

    strata.to_csv(
        STRATA_PATH,
        index=False,
    )

    heating_fuel_distribution.to_csv(
        HEATING_FUEL_PATH,
        index=False,
    )

    hvac_distribution.to_csv(
        HVAC_PATH,
        index=False,
    )

    numeric_summary.to_csv(
        NUMERIC_SUMMARY_PATH,
        index=False,
    )

    print_distribution(
        "California building-type distribution:",
        building_type_distribution,
    )

    print_distribution(
        "California ASHRAE climate-zone distribution:",
        ashrae_distribution,
    )

    print()

    print(
        "Building type x ASHRAE climate-zone strata:"
    )

    print(
        f"Total strata: "
        f"{len(strata):,}"
    )

    print(
        f"Smallest stratum: "
        f"{int(strata['buildings'].min()):,} buildings"
    )

    print(
        f"Largest stratum: "
        f"{int(strata['buildings'].max()):,} buildings"
    )

    print(
        f"Strata with fewer than 5 buildings: "
        f"{int((strata['buildings'] < 5).sum()):,}"
    )

    print(
        f"Strata with fewer than 10 buildings: "
        f"{int((strata['buildings'] < 10).sum()):,}"
    )

    print()

    print(
        "Numeric characteristics:"
    )

    print(
        numeric_summary.to_string(
            index=False
        )
    )

    print()

    print(
        "ComStock California preparation complete."
    )

    print()

    print(
        f"Saved full California metadata to: "
        f"{CA_METADATA_PATH}"
    )

    print(
        f"Saved compact building index to: "
        f"{CA_BUILDINGS_PATH}"
    )

    print(
        f"Saved summary to: "
        f"{SUMMARY_PATH}"
    )

    print(
        f"Saved building-type distribution to: "
        f"{BUILDING_TYPE_PATH}"
    )

    print(
        f"Saved ASHRAE zone distribution to: "
        f"{ASHRAE_ZONE_PATH}"
    )

    print(
        f"Saved candidate strata table to: "
        f"{STRATA_PATH}"
    )

    print(
        f"Saved heating-fuel distribution to: "
        f"{HEATING_FUEL_PATH}"
    )

    print(
        f"Saved HVAC distribution to: "
        f"{HVAC_PATH}"
    )

    print(
        f"Saved numeric summary to: "
        f"{NUMERIC_SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()