from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "resstock_metadata.parquet")

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed")

CA_METADATA_PATH = (
    OUTPUT_DIR
    / "nrel_ca_resstock_metadata.parquet")

CA_BUILDING_INDEX_PATH = (
    OUTPUT_DIR
    / "nrel_ca_resstock_buildings.csv")

CA_SUMMARY_PATH = (
    OUTPUT_DIR
    / "nrel_ca_resstock_summary.csv")


STATE_COLUMN = "in.state"
STATE_NAME_COLUMN = "in.state_name"
TARGET_STATE = "CA"


BUILDING_INDEX_COLUMNS = [
    "bldg_id",
    "in.state",
    "in.state_name",
    "in.county",
    "in.puma",
    "in.resstock_county_id",
    "in.resstock_puma_id",
    "in.cec_climate_zone",
    "in.building_america_climate_zone",
    "in.iso_rto_region",
    "in.reeds_balancing_area",
    "in.sqft",
    "in.bedrooms",
    "in.occupants",
    "in.geometry_building_type_recs",
    "in.geometry_floor_area",
    "in.geometry_stories",
    "in.vintage",
    "in.heating_fuel",
    "in.hvac_heating_type",
    "in.hvac_cooling_type",
    "in.weather_file_city",
    "in.weather_file_latitude",
    "in.weather_file_longitude",
    "out.electricity.total.energy_consumption",
    "out.electricity.total.energy_consumption_intensity",
    "qoi_report.peak_magnitude_timing_hour",
    "qoi_report.peak_magnitude_use_kw",
]


def load_metadata(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"NREL ResStock metadata not found: {path}")

    df = pd.read_parquet(
        path)

    if df.index.name == "bldg_id":
        df = (
            df
            .reset_index())

    elif "bldg_id" not in df.columns:
        raise ValueError(
            "Could not locate bldg_id in either "
            "the index or the columns.")

    return df


def validate_metadata(
    df: pd.DataFrame,
) -> None:

    required_columns = [
        "bldg_id",
        STATE_COLUMN,
        STATE_NAME_COLUMN,]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns]

    if missing_columns:
        raise ValueError(
            "Required NREL metadata columns are missing: "
            f"{missing_columns}")

    duplicate_ids = int(
        df.loc[:, "bldg_id"]
        .duplicated()
        .sum())

    if duplicate_ids != 0:
        raise ValueError(
            f"Duplicate bldg_id values found: {duplicate_ids}")


def filter_california(
    df: pd.DataFrame,
) -> pd.DataFrame:

    state_values = (
        df.loc[:, STATE_COLUMN]
        .astype(str)
        .str.strip()
        .str.upper())

    ca_df = (
        df.loc[
            state_values == TARGET_STATE
        ]
        .copy()
        .reset_index(drop=True))

    if ca_df.empty:
        raise ValueError(
            "California filter returned zero records.")

    return ca_df


def create_building_index(
    ca_df: pd.DataFrame,
) -> pd.DataFrame:

    available_columns = [
        column
        for column in BUILDING_INDEX_COLUMNS
        if column in ca_df.columns]

    building_index = (
        ca_df.filter(items=available_columns).copy())

    return building_index


def create_summary(
    full_df: pd.DataFrame, ca_df: pd.DataFrame,) -> pd.DataFrame:

    ca_missing_values = int(ca_df.isna().sum().sum())

    electricity_column = ("out.electricity.total.energy_consumption")

    if electricity_column in ca_df.columns:
        mean_electricity = float(
            ca_df.loc[:,electricity_column,].mean())

        median_electricity = float(
            ca_df.loc[:,electricity_column,].median())
    else:
        mean_electricity = float("nan")
        median_electricity = float("nan")

    sqft_column = "in.sqft"

    if sqft_column in ca_df.columns:
        mean_sqft = float(
            ca_df.loc[
                :,
                sqft_column,
            ]
            .mean()
        )
    else:
        mean_sqft = float("nan")

    summary = pd.DataFrame(
        [
            {
                "national_metadata_rows": len(full_df),
                "national_metadata_columns": len(full_df.columns),
                "california_rows": len(ca_df),
                "california_share_percent": (
                    len(ca_df)
                    / len(full_df)
                    * 100
                ),
                "california_unique_buildings": (
                    ca_df.loc[
                        :,
                        "bldg_id",
                    ]
                    .nunique()
                ),
                "california_missing_values": ca_missing_values,
                "mean_california_sqft": mean_sqft,
                "mean_annual_electricity_consumption": (
                    mean_electricity
                ),
                "median_annual_electricity_consumption": (
                    median_electricity
                ),
            }
        ]
    )

    return summary


def print_summary(
    full_df: pd.DataFrame,
    ca_df: pd.DataFrame,
    building_index: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:

    print(
        f"National ResStock rows: "
        f"{len(full_df):,}"
    )

    print(
        f"National ResStock columns: "
        f"{len(full_df.columns):,}"
    )

    print()

    print(
        f"California rows: "
        f"{len(ca_df):,}"
    )

    print(
        f"California unique buildings: "
        f"{ca_df.loc[:, 'bldg_id'].nunique():,}"
    )

    print(
        f"California share of metadata: "
        f"{len(ca_df) / len(full_df) * 100:.2f}%"
    )

    print()

    print(
        f"Building-index columns retained: "
        f"{len(building_index.columns)}"
    )

    print()

    print(
        "California state values:"
    )

    print(
        ca_df.loc[
            :,
            [
                STATE_COLUMN,
                STATE_NAME_COLUMN,
            ],
        ]
        .drop_duplicates()
        .to_string(
            index=False
        )
    )

    if "upgrade" in ca_df.columns:

        print()

        print(
            "Upgrade values:"
        )

        print(
            ca_df.loc[
                :,
                "upgrade",
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    print()

    print(
        "California summary:"
    )

    print(
        summary_df.to_string(
            index=False
        )
    )


def save_outputs(
    ca_df: pd.DataFrame,
    building_index: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ca_df.to_parquet(
        CA_METADATA_PATH,
        index=False,
    )

    building_index.to_csv(
        CA_BUILDING_INDEX_PATH,
        index=False,
    )

    summary_df.to_csv(
        CA_SUMMARY_PATH,
        index=False,
    )

    print()

    print(
        "Saved California metadata to:"
    )

    print(
        CA_METADATA_PATH
    )

    print()

    print(
        "Saved California building index to:"
    )

    print(
        CA_BUILDING_INDEX_PATH
    )

    print()

    print(
        "Saved California summary to:"
    )

    print(
        CA_SUMMARY_PATH
    )


def main() -> None:

    full_df = load_metadata(
        INPUT_PATH
    )

    validate_metadata(
        full_df
    )

    ca_df = filter_california(
        full_df
    )

    building_index = create_building_index(ca_df)

    summary_df = create_summary(full_df,ca_df,)

    print_summary(full_df, ca_df, building_index, summary_df)

    save_outputs(ca_df, building_index, summary_df)


if __name__ == "__main__":
    main()