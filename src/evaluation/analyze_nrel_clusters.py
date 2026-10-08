from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLUSTER_ASSIGNMENTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "nrel_load_shape_cluster_assignments.csv"
)

CLUSTER_PROFILES_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "nrel_load_shape_cluster_profiles.csv"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_modeling_sample.csv"
)

MODELING_BASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_modeling_base.parquet"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
)

CLUSTER_METADATA_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "nrel_cluster_metadata_summary.csv"
)

CLUSTER_BUILDING_TYPE_PATH = (
    OUTPUT_DIRECTORY
    / "nrel_cluster_by_building_type.csv"
)

CLUSTER_CLIMATE_ZONE_PATH = (
    OUTPUT_DIRECTORY
    / "nrel_cluster_by_climate_zone.csv"
)

CLUSTER_VINTAGE_PATH = (
    OUTPUT_DIRECTORY
    / "nrel_cluster_by_vintage.csv"
)

CLUSTER_HEATING_FUEL_PATH = (
    OUTPUT_DIRECTORY
    / "nrel_cluster_by_heating_fuel.csv"
)

CLUSTER_HVAC_COOLING_PATH = (
    OUTPUT_DIRECTORY
    / "nrel_cluster_by_hvac_cooling.csv"
)

CLUSTER_HVAC_HEATING_PATH = (
    OUTPUT_DIRECTORY
    / "nrel_cluster_by_hvac_heating.csv"
)

CLUSTER_CONTINUOUS_PATH = (
    OUTPUT_DIRECTORY
    / "nrel_cluster_continuous_characteristics.csv"
)

CLUSTER_PROFILE_LONG_PATH = (
    OUTPUT_DIRECTORY
    / "nrel_cluster_profiles_long.csv"
)


BUILDING_ID_COLUMN = "bldg_id"
CLUSTER_COLUMN = "cluster"
ROLE_COLUMN = "sample_role"

BUILDING_TYPE_COLUMN = (
    "in.geometry_building_type_recs"
)

CLIMATE_ZONE_COLUMN = (
    "in.cec_climate_zone"
)

VINTAGE_COLUMN = (
    "in.vintage"
)

HEATING_FUEL_COLUMN = (
    "in.heating_fuel"
)

HVAC_COOLING_COLUMN = (
    "in.hvac_cooling_type"
)

HVAC_HEATING_COLUMN = (
    "in.hvac_heating_type"
)

SQFT_COLUMN = (
    "in.sqft"
)

OCCUPANTS_COLUMN = (
    "in.occupants"
)

BEDROOMS_COLUMN = (
    "in.bedrooms"
)

STORIES_COLUMN = (
    "in.geometry_stories"
)


CATEGORICAL_COLUMNS = [
    BUILDING_TYPE_COLUMN,
    CLIMATE_ZONE_COLUMN,
    VINTAGE_COLUMN,
    HEATING_FUEL_COLUMN,
    HVAC_COOLING_COLUMN,
    HVAC_HEATING_COLUMN,
]

CONTINUOUS_COLUMNS = [
    SQFT_COLUMN,
    OCCUPANTS_COLUMN,
    BEDROOMS_COLUMN,
    STORIES_COLUMN,
]


def load_cluster_assignments() -> pd.DataFrame:

    if not CLUSTER_ASSIGNMENTS_PATH.exists():
        raise FileNotFoundError(
            f"Cluster assignments not found: "
            f"{CLUSTER_ASSIGNMENTS_PATH}"
        )

    assignments = pd.read_csv(
        CLUSTER_ASSIGNMENTS_PATH
    )

    required_columns = {
        BUILDING_ID_COLUMN,
        CLUSTER_COLUMN,
        ROLE_COLUMN,
    }

    missing_columns = (
        required_columns
        - set(assignments.columns)
    )

    if missing_columns:
        raise ValueError(
            "Cluster assignment file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if assignments[
        BUILDING_ID_COLUMN
    ].duplicated().any():
        raise ValueError(
            "Cluster assignment file contains duplicate building IDs."
        )

    return assignments


def load_metadata() -> pd.DataFrame:

    if not MODELING_BASE_PATH.exists():
        raise FileNotFoundError(
            f"Modeling base dataset not found: "
            f"{MODELING_BASE_PATH}"
        )

    required_columns = [
        BUILDING_ID_COLUMN,
        *CATEGORICAL_COLUMNS,
        *CONTINUOUS_COLUMNS,
    ]

    metadata = pd.read_parquet(
        MODELING_BASE_PATH,
        columns=required_columns,
    )

    metadata = (
        metadata
        .drop_duplicates(
            subset=[
                BUILDING_ID_COLUMN,
            ]
        )
        .copy()
    )

    if metadata[
        BUILDING_ID_COLUMN
    ].duplicated().any():
        raise ValueError(
            "Metadata contains duplicate building IDs."
        )

    return metadata


def merge_assignments_metadata(
    assignments: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:

    merged = assignments.merge(
        metadata,
        on=BUILDING_ID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    metadata_columns = (
        CATEGORICAL_COLUMNS
        + CONTINUOUS_COLUMNS
    )

    missing_metadata_count = int(
        merged[
            metadata_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    if missing_metadata_count != 0:
        raise ValueError(
            f"Metadata merge produced "
            f"{missing_metadata_count} missing values."
        )

    return merged


def create_cluster_metadata_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for cluster_id, group in df.groupby(
        CLUSTER_COLUMN,
        sort=True,
    ):

        rows.append(
            {
                "cluster": int(
                    str(
                        cluster_id
                    )
                ),
                "buildings": int(
                    len(
                        group
                    )
                ),
                "development_buildings": int(
                    (
                        group[
                            ROLE_COLUMN
                        ]
                        == "development"
                    )
                    .sum()
                ),
                "held_out_buildings": int(
                    (
                        group[
                            ROLE_COLUMN
                        ]
                        == "held_out"
                    )
                    .sum()
                ),
                "mean_sqft": float(
                    group[
                        SQFT_COLUMN
                    ].mean()
                ),
                "median_sqft": float(
                    group[
                        SQFT_COLUMN
                    ].median()
                ),
                "mean_occupants": float(
                    group[
                        OCCUPANTS_COLUMN
                    ].mean()
                ),
                "median_occupants": float(
                    group[
                        OCCUPANTS_COLUMN
                    ].median()
                ),
                "mean_bedrooms": float(
                    group[
                        BEDROOMS_COLUMN
                    ].mean()
                ),
                "median_bedrooms": float(
                    group[
                        BEDROOMS_COLUMN
                    ].median()
                ),
                "mean_stories": float(
                    group[
                        STORIES_COLUMN
                    ].mean()
                ),
                "median_stories": float(
                    group[
                        STORIES_COLUMN
                    ].median()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def create_categorical_distribution(
    df: pd.DataFrame,
    category_column: str,
) -> pd.DataFrame:

    counts = (
        df
        .groupby(
            [
                CLUSTER_COLUMN,
                category_column,
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="buildings"
        )
    )

    cluster_sizes = (
        df
        .groupby(
            CLUSTER_COLUMN
        )
        .size()
        .rename(
            "cluster_total"
        )
        .reset_index()
    )

    counts = counts.merge(
        cluster_sizes,
        on=CLUSTER_COLUMN,
        how="left",
        validate="many_to_one",
    )

    counts[
        "pct_of_cluster"
    ] = (
        counts[
            "buildings"
        ]
        / counts[
            "cluster_total"
        ]
        * 100
    )

    counts = counts.sort_values(
        [
            CLUSTER_COLUMN,
            "buildings",
        ],
        ascending=[
            True,
            False,
        ],
    )

    return counts


def create_continuous_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for cluster_id, group in df.groupby(
        CLUSTER_COLUMN,
        sort=True,
    ):

        for column in CONTINUOUS_COLUMNS:

            values = (
                group[
                    column
                ]
                .to_numpy(
                    dtype=float
                )
            )

            rows.append(
                {
                    "cluster": int(
                        str(
                            cluster_id
                        )
                    ),
                    "feature": column,
                    "n": int(
                        len(
                            values
                        )
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


def create_cluster_profile_long() -> pd.DataFrame:

    if not CLUSTER_PROFILES_PATH.exists():
        raise FileNotFoundError(
            f"Cluster profiles not found: "
            f"{CLUSTER_PROFILES_PATH}"
        )

    profiles = pd.read_csv(
        CLUSTER_PROFILES_PATH
    )

    required_hour_columns = [
        f"hour_{hour:02d}"
        for hour in range(
            24
        )
    ]

    required_columns = {
        CLUSTER_COLUMN,
        *required_hour_columns,
    }

    missing_columns = (
        required_columns
        - set(profiles.columns)
    )

    if missing_columns:
        raise ValueError(
            "Cluster profile file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    long_profiles = profiles.melt(
        id_vars=[
            CLUSTER_COLUMN,
        ],
        value_vars=required_hour_columns,
        var_name="hour_label",
        value_name="normalized_load",
    )

    long_profiles[
        "hour"
    ] = (
        long_profiles[
            "hour_label"
        ]
        .str.replace(
            "hour_",
            "",
            regex=False,
        )
        .astype(
            int
        )
    )

    long_profiles = (
        long_profiles[
            [
                CLUSTER_COLUMN,
                "hour",
                "normalized_load",
            ]
        ]
        .sort_values(
            [
                CLUSTER_COLUMN,
                "hour",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return long_profiles


def print_top_categories(
    df: pd.DataFrame,
    category_column: str,
    label: str,
) -> None:

    print()

    print(
        f"Dominant {label} by cluster:"
    )

    for cluster_id, group in df.groupby(
        CLUSTER_COLUMN,
        sort=True,
    ):

        category_counts = (
            group[
                category_column
            ]
            .value_counts(
                dropna=False
            )
        )

        top_category = str(
            category_counts.index[
                0
            ]
        )

        top_count = int(
            category_counts.iloc[
                0
            ]
        )

        pct = (
            top_count
            / len(
                group
            )
            * 100
        )

        print(
            f"  Cluster "
            f"{int(str(cluster_id))}: "
            f"{top_category} "
            f"({top_count}/{len(group)}, "
            f"{pct:.1f}%)"
        )


def print_continuous_summary(
    summary: pd.DataFrame,
) -> None:

    print()

    print(
        "Continuous-characteristic summary:"
    )

    display_columns = [
        "cluster",
        "buildings",
        "mean_sqft",
        "mean_occupants",
        "mean_bedrooms",
        "mean_stories",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )


def main() -> None:

    print(
        "Loading NREL load-shape cluster assignments..."
    )

    assignments = load_cluster_assignments()

    print(
        f"Clustered buildings: "
        f"{len(assignments):,}"
    )

    print(
        f"Clusters: "
        f"{assignments[CLUSTER_COLUMN].nunique()}"
    )

    print()

    print(
        "Loading NREL static building metadata..."
    )

    metadata = load_metadata()

    print(
        f"Metadata buildings: "
        f"{len(metadata):,}"
    )

    print()

    print(
        "Merging cluster assignments with building metadata..."
    )

    merged = merge_assignments_metadata(
        assignments,
        metadata,
    )

    print(
        f"Merged buildings: "
        f"{len(merged):,}"
    )

    print()

    print(
        "Creating overall cluster metadata summary..."
    )

    metadata_summary = (
        create_cluster_metadata_summary(
            merged
        )
    )

    print(
        "Creating categorical distributions..."
    )

    building_type_summary = (
        create_categorical_distribution(
            merged,
            BUILDING_TYPE_COLUMN,
        )
    )

    climate_zone_summary = (
        create_categorical_distribution(
            merged,
            CLIMATE_ZONE_COLUMN,
        )
    )

    vintage_summary = (
        create_categorical_distribution(
            merged,
            VINTAGE_COLUMN,
        )
    )

    heating_fuel_summary = (
        create_categorical_distribution(
            merged,
            HEATING_FUEL_COLUMN,
        )
    )

    hvac_cooling_summary = (
        create_categorical_distribution(
            merged,
            HVAC_COOLING_COLUMN,
        )
    )

    hvac_heating_summary = (
        create_categorical_distribution(
            merged,
            HVAC_HEATING_COLUMN,
        )
    )

    print(
        "Creating continuous-characteristic summaries..."
    )

    continuous_summary = (
        create_continuous_summary(
            merged
        )
    )

    print(
        "Creating long-format cluster profiles..."
    )

    profile_long = (
        create_cluster_profile_long()
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_summary.to_csv(
        CLUSTER_METADATA_SUMMARY_PATH,
        index=False,
    )

    building_type_summary.to_csv(
        CLUSTER_BUILDING_TYPE_PATH,
        index=False,
    )

    climate_zone_summary.to_csv(
        CLUSTER_CLIMATE_ZONE_PATH,
        index=False,
    )

    vintage_summary.to_csv(
        CLUSTER_VINTAGE_PATH,
        index=False,
    )

    heating_fuel_summary.to_csv(
        CLUSTER_HEATING_FUEL_PATH,
        index=False,
    )

    hvac_cooling_summary.to_csv(
        CLUSTER_HVAC_COOLING_PATH,
        index=False,
    )

    hvac_heating_summary.to_csv(
        CLUSTER_HVAC_HEATING_PATH,
        index=False,
    )

    continuous_summary.to_csv(
        CLUSTER_CONTINUOUS_PATH,
        index=False,
    )

    profile_long.to_csv(
        CLUSTER_PROFILE_LONG_PATH,
        index=False,
    )

    print_continuous_summary(
        metadata_summary
    )

    print_top_categories(
        merged,
        BUILDING_TYPE_COLUMN,
        "building type",
    )

    print_top_categories(
        merged,
        CLIMATE_ZONE_COLUMN,
        "CEC climate zone",
    )

    print_top_categories(
        merged,
        VINTAGE_COLUMN,
        "vintage",
    )

    print_top_categories(
        merged,
        HEATING_FUEL_COLUMN,
        "heating fuel",
    )

    print_top_categories(
        merged,
        HVAC_COOLING_COLUMN,
        "HVAC cooling type",
    )

    print_top_categories(
        merged,
        HVAC_HEATING_COLUMN,
        "HVAC heating type",
    )

    print()

    print(
        "NREL cluster interpretation analysis complete."
    )

    print()

    print(
        f"Saved cluster metadata summary to: "
        f"{CLUSTER_METADATA_SUMMARY_PATH}"
    )

    print(
        f"Saved building-type distribution to: "
        f"{CLUSTER_BUILDING_TYPE_PATH}"
    )

    print(
        f"Saved climate-zone distribution to: "
        f"{CLUSTER_CLIMATE_ZONE_PATH}"
    )

    print(
        f"Saved vintage distribution to: "
        f"{CLUSTER_VINTAGE_PATH}"
    )

    print(
        f"Saved heating-fuel distribution to: "
        f"{CLUSTER_HEATING_FUEL_PATH}"
    )

    print(
        f"Saved HVAC cooling distribution to: "
        f"{CLUSTER_HVAC_COOLING_PATH}"
    )

    print(
        f"Saved HVAC heating distribution to: "
        f"{CLUSTER_HVAC_HEATING_PATH}"
    )

    print(
        f"Saved continuous characteristics to: "
        f"{CLUSTER_CONTINUOUS_PATH}"
    )

    print(
        f"Saved long-format cluster profiles to: "
        f"{CLUSTER_PROFILE_LONG_PATH}"
    )


if __name__ == "__main__":
    main()