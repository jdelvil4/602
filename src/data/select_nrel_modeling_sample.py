from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_resstock_buildings.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_modeling_sample.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_modeling_sample_summary.csv"
)


TOTAL_SAMPLE_SIZE = 250
DEVELOPMENT_BUILDINGS = 200
HELD_OUT_BUILDINGS = 50
RANDOM_STATE = 42


BUILDING_ID_COLUMN = "bldg_id"

RAW_STRATIFICATION_COLUMNS = [
    "in.geometry_building_type_recs",
    "in.sqft",
    "in.cec_climate_zone",
    "in.vintage",
]

STRATIFICATION_COLUMNS = [
    "in.geometry_building_type_recs",
    "in.cec_climate_zone",
]


def load_building_index() -> pd.DataFrame:

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"California building index not found: {INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH
    )

    required_columns = {
        BUILDING_ID_COLUMN,
        *RAW_STRATIFICATION_COLUMNS,
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Building index is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return df


def clean_metadata(
    df: pd.DataFrame,
) -> pd.DataFrame:

    working = (
        df.loc[
            :,
            [
                BUILDING_ID_COLUMN,
                *RAW_STRATIFICATION_COLUMNS,
            ],
        ]
        .copy()
    )

    working[BUILDING_ID_COLUMN] = pd.to_numeric(
        working[BUILDING_ID_COLUMN],
        errors="coerce",
    )

    working["in.sqft"] = pd.to_numeric(
        working["in.sqft"],
        errors="coerce",
    )

    working = (
        working
        .dropna(
            subset=[
                BUILDING_ID_COLUMN,
            ]
        )
        .copy()
    )

    working[BUILDING_ID_COLUMN] = (
        working[BUILDING_ID_COLUMN]
        .astype("int64")
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
        "in.geometry_building_type_recs",
        "in.cec_climate_zone",
        "in.vintage",
    ]

    for column in categorical_columns:

        cleaned_column = (
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

        working[column] = cleaned_column

    duplicate_count = int(
        working[
            BUILDING_ID_COLUMN
        ]
        .duplicated()
        .sum()
    )

    if duplicate_count != 0:
        raise ValueError(
            f"Duplicate building IDs found: {duplicate_count}"
        )

    return (
        working
        .sort_values(
            BUILDING_ID_COLUMN
        )
        .reset_index(drop=True)
    )


def create_stratum(
    df: pd.DataFrame,
) -> pd.DataFrame:

    working = df.copy()

    working.loc[
        :,
        "stratum",
    ] = (
        working.loc[
            :,
            STRATIFICATION_COLUMNS,
        ]
        .astype("string")
        .agg(
            " | ".join,
            axis=1,
        )
    )

    return working


def calculate_stratum_targets(
    df: pd.DataFrame,
    total_sample_size: int,
) -> pd.DataFrame:

    counts = (
        df.loc[
            :,
            "stratum",
        ]
        .value_counts()
        .rename_axis(
            "stratum"
        )
        .reset_index(
            name="population_count"
        )
    )

    counts.loc[
        :,
        "population_share",
    ] = (
        counts.loc[
            :,
            "population_count",
        ]
        / len(df)
    )

    counts.loc[
        :,
        "raw_target",
    ] = (
        counts.loc[
            :,
            "population_share",
        ]
        * total_sample_size
    )

    counts.loc[
        :,
        "sample_target",
    ] = (
        np.floor(
            counts.loc[
                :,
                "raw_target",
            ]
        )
        .astype(int)
    )

    counts.loc[
        :,
        "fractional_remainder",
    ] = (
        counts.loc[
            :,
            "raw_target",
        ]
        - counts.loc[
            :,
            "sample_target",
        ]
    )

    return counts

def adjust_targets(
    target_df: pd.DataFrame,
    total_sample_size: int,
) -> pd.DataFrame:

    working = (
        target_df
        .copy()
    )

    current_total = int(
        working.loc[
            :,
            "sample_target",
        ]
        .sum()
    )

    remaining = (
        total_sample_size
        - current_total
    )

    if remaining < 0:
        raise ValueError(
            "Initial proportional allocation exceeds "
            "the requested sample size."
        )

    if remaining > 0:

        eligible = (
            working.loc[
                working.loc[
                    :,
                    "sample_target",
                ]
                < working.loc[
                    :,
                    "population_count",
                ]
            ]
            .sort_values(
                [
                    "fractional_remainder",
                    "population_count",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
        )

        if len(eligible) < remaining:
            raise ValueError(
                "Not enough eligible strata remain "
                "to complete sample allocation."
            )

        selected_indices = (
            eligible
            .head(
                remaining
            )
            .index
        )

        working.loc[
            selected_indices,
            "sample_target",
        ] += 1

    final_total = int(
        working.loc[
            :,
            "sample_target",
        ]
        .sum()
    )

    if final_total != total_sample_size:
        raise ValueError(
            f"Final allocation is {final_total}; "
            f"expected {total_sample_size}."
        )

    return working

def sample_buildings(
    df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:

    sampled_frames = []

    for row in target_df.itertuples(
        index=False
    ):

        stratum_value = str(
            row.stratum
        )

        sample_target = int(
            str(
                row.sample_target
            )
        )
        
        if sample_target == 0:
            continue

        stratum_df = (
            df.loc[
                df.loc[
                    :,
                    "stratum",
                ]
                == stratum_value
            ]
            .copy()
        )

        if len(stratum_df) < sample_target:
            raise ValueError(
                f"Stratum has only {len(stratum_df)} buildings "
                f"but target is {sample_target}: {stratum_value}"
            )

        sampled = (
            stratum_df
            .sample(
                n=sample_target,
                random_state=(
                    RANDOM_STATE
                    + len(sampled_frames)
                ),
                replace=False,
            )
        )

        sampled_frames.append(
            sampled
        )

    sampled_df = pd.concat(
        sampled_frames,
        ignore_index=True,
    )

    sampled_df = (
        sampled_df
        .sort_values(
            BUILDING_ID_COLUMN
        )
        .reset_index(drop=True)
    )

    if len(sampled_df) != TOTAL_SAMPLE_SIZE:
        raise ValueError(
            f"Sample contains {len(sampled_df)} buildings; "
            f"expected {TOTAL_SAMPLE_SIZE}."
        )

    return sampled_df

def assign_sample_roles(
    sampled_df: pd.DataFrame,
) -> pd.DataFrame:

    working = sampled_df.copy()

    working.loc[
        :,
        "sample_role",
    ] = ""

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    held_out_indices = []

    grouped = working.groupby(
        "stratum",
        sort=True,
    )

    for _, group in grouped:

        group_indices = group.index.to_numpy()

        shuffled_indices = rng.permutation(
            group_indices
        )

        group_size = len(
            shuffled_indices
        )

        held_out_count = int(
            round(
                group_size
                * (
                    HELD_OUT_BUILDINGS
                    / TOTAL_SAMPLE_SIZE
                )
            )
        )

        held_out_count = min(
            held_out_count,
            group_size,
        )

        if held_out_count > 0:

            held_out_indices.extend(
                shuffled_indices[
                    :held_out_count
                ]
                .tolist()
            )

    held_out_indices = list(
        dict.fromkeys(
            held_out_indices
        )
    )

    current_held_out = len(
        held_out_indices
    )

    if current_held_out < HELD_OUT_BUILDINGS:

        remaining_indices = (
            working.loc[
                ~working.index.isin(
                    held_out_indices
                )
            ]
            .index
            .to_numpy()
        )

        additional_needed = (
            HELD_OUT_BUILDINGS
            - current_held_out
        )

        additional_indices = rng.choice(
            remaining_indices,
            size=additional_needed,
            replace=False,
        )

        held_out_indices.extend(
            additional_indices.tolist()
        )

    elif current_held_out > HELD_OUT_BUILDINGS:

        held_out_indices = (
            rng.choice(
                np.array(
                    held_out_indices
                ),
                size=HELD_OUT_BUILDINGS,
                replace=False,
            )
            .tolist()
        )

    working.loc[
        :,
        "sample_role",
    ] = "development"

    working.loc[
        held_out_indices,
        "sample_role",
    ] = "held_out"

    development_count = int(
        (
            working.loc[
                :,
                "sample_role",
            ]
            == "development"
        )
        .sum()
    )

    held_out_count = int(
        (
            working.loc[
                :,
                "sample_role",
            ]
            == "held_out"
        )
        .sum()
    )

    if development_count != DEVELOPMENT_BUILDINGS:
        raise ValueError(
            f"Development count is {development_count}; "
            f"expected {DEVELOPMENT_BUILDINGS}."
        )

    if held_out_count != HELD_OUT_BUILDINGS:
        raise ValueError(
            f"Held-out count is {held_out_count}; "
            f"expected {HELD_OUT_BUILDINGS}."
        )

    return (
        working
        .sort_values(
            BUILDING_ID_COLUMN
        )
        .reset_index(drop=True)
    )

def create_summary(
    population_df: pd.DataFrame,
    sample_df: pd.DataFrame,
) -> pd.DataFrame:

    summary_rows = []

    summary_rows.append(
        {
            "metric": "population_buildings",
            "value": len(
                population_df
            ),
        }
    )

    summary_rows.append(
        {
            "metric": "sampled_buildings",
            "value": len(
                sample_df
            ),
        }
    )

    summary_rows.append(
        {
            "metric": "development_buildings",
            "value": int(
                (
                    sample_df.loc[
                        :,
                        "sample_role",
                    ]
                    == "development"
                )
                .sum()
            ),
        }
    )

    summary_rows.append(
        {
            "metric": "held_out_buildings",
            "value": int(
                (
                    sample_df.loc[
                        :,
                        "sample_role",
                    ]
                    == "held_out"
                )
                .sum()
            ),
        }
    )

    summary_rows.append(
        {
            "metric": "unique_strata_population",
            "value": population_df.loc[
                :,
                "stratum",
            ].nunique(),
        }
    )

    summary_rows.append(
        {
            "metric": "unique_strata_sample",
            "value": sample_df.loc[
                :,
                "stratum",
            ].nunique(),
        }
    )

    return pd.DataFrame(
        summary_rows
    )


def print_diagnostics(
    population_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:

    print(
        "NREL modeling sample selection complete."
    )

    print()

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()

    print(
        "Sample role counts:"
    )

    print(
        sample_df[
            "sample_role"
        ]
        .value_counts()
        .to_string()
    )

    print()

    print(
        "Building type distribution:"
    )

    building_type_comparison = pd.concat(
        [
            population_df[
                "in.geometry_building_type_recs"
            ]
            .value_counts(
                normalize=True
            )
            .rename(
                "population_share"
            ),
            sample_df[
                "in.geometry_building_type_recs"
            ]
            .value_counts(
                normalize=True
            )
            .rename(
                "sample_share"
            ),
        ],
        axis=1,
    ).fillna(0)

    print(
        building_type_comparison.to_string()
    )

    print()

    print(
        "Climate-zone distribution:"
    )

    climate_comparison = pd.concat(
        [
            population_df[
                "in.cec_climate_zone"
            ]
            .value_counts(
                normalize=True
            )
            .rename(
                "population_share"
            ),
            sample_df[
                "in.cec_climate_zone"
            ]
            .value_counts(
                normalize=True
            )
            .rename(
                "sample_share"
            ),
        ],
        axis=1,
    ).fillna(0)

    print(
        climate_comparison.to_string()
    )

    print()

    print(
        "Vintage distribution:"
    )

    vintage_comparison = pd.concat(
        [
            population_df[
                "in.vintage"
            ]
            .value_counts(
                normalize=True
            )
            .rename(
                "population_share"
            ),
            sample_df[
                "in.vintage"
            ]
            .value_counts(
                normalize=True
            )
            .rename(
                "sample_share"
            ),
        ],
        axis=1,
    ).fillna(0)

    print(
        vintage_comparison.to_string()
    )

    print()

    print(
        "Floor-area distribution:"
    )

    floor_area_comparison = pd.concat(
        [
            population_df[
                "floor_area_group"
            ]
            .value_counts(
                normalize=True
            )
            .rename(
                "population_share"
            ),
            sample_df[
                "floor_area_group"
            ]
            .value_counts(
                normalize=True
            )
            .rename(
                "sample_share"
            ),
        ],
        axis=1,
    ).fillna(0)

    print(
        floor_area_comparison.to_string()
    )

    print()

    print(
        "Building type distribution by sample role:"
    )

    role_building_type = pd.crosstab(
        sample_df[
            "in.geometry_building_type_recs"
        ],
        sample_df[
            "sample_role"
        ],
        normalize="columns",
    )

    print(
        role_building_type.to_string()
    )

    print()

    print(
        "Climate-zone distribution by sample role:"
    )

    role_climate = pd.crosstab(
        sample_df[
            "in.cec_climate_zone"
        ],
        sample_df[
            "sample_role"
        ],
        normalize="columns",
    )

    print(
        role_climate.to_string()
    )

def main() -> None:

    population = load_building_index()

    population = clean_metadata(
        population
    )

    population = create_stratum(
        population
    )

    targets = calculate_stratum_targets(
        population,
        TOTAL_SAMPLE_SIZE,
    )

    targets = adjust_targets(
        targets,
        TOTAL_SAMPLE_SIZE,
    )

    sampled = sample_buildings(
        population,
        targets,
    )

    sampled = assign_sample_roles(
        sampled
    )

    summary = create_summary(
        population,
        sampled,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sampled.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print_diagnostics(
        population,
        sampled,
        summary,
    )

    print()

    print(
        f"Saved modeling sample to: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Saved summary to: "
        f"{SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()