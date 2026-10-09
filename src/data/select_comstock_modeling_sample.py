from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comstock_ca_buildings.csv"
)

SAMPLE_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comstock_ca_modeling_sample.csv"
)

SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comstock_ca_modeling_sample_summary.csv"
)

BUILDING_TYPE_COMPARISON_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comstock_ca_sample_building_type_comparison.csv"
)

CLIMATE_ZONE_COMPARISON_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comstock_ca_sample_climate_zone_comparison.csv"
)


BUILDING_ID_COLUMN = "bldg_id"
BUILDING_TYPE_COLUMN = "in.building_type"
CLIMATE_ZONE_COLUMN = "in.climate_zone_ashrae_2004"
SQFT_COLUMN = "in.sqft"
STORIES_COLUMN = "in.number_of_stories"

RANDOM_STATE = 42

TOTAL_SAMPLE_SIZE = 250
DEVELOPMENT_SIZE = 200
HELD_OUT_SIZE = 50

MINIMUM_PER_BUILDING_TYPE = 2


def load_buildings() -> pd.DataFrame:

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"California ComStock building index not found: "
            f"{INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH
    )

    required_columns = {
        BUILDING_ID_COLUMN,
        BUILDING_TYPE_COLUMN,
        CLIMATE_ZONE_COLUMN,
        SQFT_COLUMN,
        STORIES_COLUMN,
    }

    missing_columns = (
        required_columns
        - set(
            df.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "ComStock building index is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if df[
        BUILDING_ID_COLUMN
    ].duplicated().any():
        raise ValueError(
            "California ComStock building index "
            "contains duplicate building IDs."
        )

    return df


def allocate_building_type_sample_sizes(
    df: pd.DataFrame,
) -> pd.DataFrame:

    counts = (
        df[
            BUILDING_TYPE_COLUMN
        ]
        .value_counts()
        .rename_axis(
            BUILDING_TYPE_COLUMN
        )
        .reset_index(
            name="population_buildings"
        )
    )

    counts[
        "population_fraction"
    ] = (
        counts[
            "population_buildings"
        ]
        / len(
            df
        )
    )

    counts[
        "raw_allocation"
    ] = (
        counts[
            "population_fraction"
        ]
        * TOTAL_SAMPLE_SIZE
    )

    counts[
        "sample_buildings"
    ] = np.floor(
        counts[
            "raw_allocation"
        ]
    ).astype(
        int
    )

    counts[
        "sample_buildings"
    ] = counts[
        "sample_buildings"
    ].clip(
        lower=MINIMUM_PER_BUILDING_TYPE
    )

    counts[
        "sample_buildings"
    ] = np.minimum(
        counts[
            "sample_buildings"
        ],
        counts[
            "population_buildings"
        ],
    )

    current_total = int(
        counts[
            "sample_buildings"
        ].sum()
    )

    if current_total < TOTAL_SAMPLE_SIZE:

        counts[
            "remainder"
        ] = (
            counts[
                "raw_allocation"
            ]
            - np.floor(
                counts[
                    "raw_allocation"
                ]
            )
        )

        while current_total < TOTAL_SAMPLE_SIZE:

            eligible = counts.loc[
                counts[
                    "sample_buildings"
                ]
                < counts[
                    "population_buildings"
                ]
            ].copy()

            if eligible.empty:
                raise ValueError(
                    "Unable to allocate requested "
                    "ComStock sample size."
                )

            eligible = eligible.sort_values(
                [
                    "remainder",
                    "population_buildings",
                    BUILDING_TYPE_COLUMN,
                ],
                ascending=[
                    False,
                    False,
                    True,
                ],
            )

            index_to_increment = (
                eligible.index[
                    0
                ]
            )

            counts.loc[
                index_to_increment,
                "sample_buildings",
            ] += 1

            current_total += 1

    elif current_total > TOTAL_SAMPLE_SIZE:

        counts[
            "over_allocation"
        ] = (
            counts[
                "sample_buildings"
            ]
            - counts[
                "raw_allocation"
            ]
        )

        while current_total > TOTAL_SAMPLE_SIZE:

            eligible = counts.loc[
                counts[
                    "sample_buildings"
                ]
                > MINIMUM_PER_BUILDING_TYPE
            ].copy()

            if eligible.empty:
                raise ValueError(
                    "Unable to reduce allocation "
                    "to requested sample size while "
                    "preserving minimum type coverage."
                )

            eligible = eligible.sort_values(
                [
                    "over_allocation",
                    "population_buildings",
                    BUILDING_TYPE_COLUMN,
                ],
                ascending=[
                    False,
                    True,
                    True,
                ],
            )

            index_to_decrement = (
                eligible.index[
                    0
                ]
            )

            counts.loc[
                index_to_decrement,
                "sample_buildings",
            ] -= 1

            current_total -= 1

    if int(
        counts[
            "sample_buildings"
        ].sum()
    ) != TOTAL_SAMPLE_SIZE:
        raise ValueError(
            "Final building-type allocation "
            "does not equal requested sample size."
        )

    return counts


def allocate_climate_zone_counts(
    type_df: pd.DataFrame,
    requested_size: int,
) -> pd.DataFrame:

    zone_counts = (
        type_df[
            CLIMATE_ZONE_COLUMN
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            CLIMATE_ZONE_COLUMN
        )
        .reset_index(
            name="population_buildings"
        )
    )

    zone_counts[
        "population_fraction"
    ] = (
        zone_counts[
            "population_buildings"
        ]
        / len(
            type_df
        )
    )

    zone_counts[
        "raw_allocation"
    ] = (
        zone_counts[
            "population_fraction"
        ]
        * requested_size
    )

    zone_counts[
        "sample_buildings"
    ] = np.floor(
        zone_counts[
            "raw_allocation"
        ]
    ).astype(
        int
    )

    zone_counts[
        "sample_buildings"
    ] = np.minimum(
        zone_counts[
            "sample_buildings"
        ],
        zone_counts[
            "population_buildings"
        ],
    )

    allocated = int(
        zone_counts[
            "sample_buildings"
        ].sum()
    )

    zone_counts[
        "remainder"
    ] = (
        zone_counts[
            "raw_allocation"
        ]
        - np.floor(
            zone_counts[
                "raw_allocation"
            ]
        )
    )

    while allocated < requested_size:

        eligible = zone_counts.loc[
            zone_counts[
                "sample_buildings"
            ]
            < zone_counts[
                "population_buildings"
            ]
        ].copy()

        if eligible.empty:
            raise ValueError(
                "Unable to complete climate-zone allocation."
            )

        eligible = eligible.sort_values(
            [
                "remainder",
                "population_buildings",
            ],
            ascending=[
                False,
                False,
            ],
        )

        index_to_increment = (
            eligible.index[
                0
            ]
        )

        zone_counts.loc[
            index_to_increment,
            "sample_buildings",
        ] += 1

        allocated += 1

    return zone_counts


def sample_buildings(
    df: pd.DataFrame,
    type_allocations: pd.DataFrame,
) -> pd.DataFrame:

    sampled_frames = []

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    for _, allocation_row in (
        type_allocations.iterrows()
    ):

        building_type = allocation_row[
            BUILDING_TYPE_COLUMN
        ]

        requested_size = int(
            allocation_row[
                "sample_buildings"
            ]
        )

        type_df = (
            df.loc[
                df[
                    BUILDING_TYPE_COLUMN
                ]
                == building_type
            ]
            .copy()
        )

        zone_allocations = (
            allocate_climate_zone_counts(
                type_df,
                requested_size,
            )
        )

        for _, zone_row in (
            zone_allocations.iterrows()
        ):

            climate_zone = zone_row[
                CLIMATE_ZONE_COLUMN
            ]

            zone_sample_size = int(
                zone_row[
                    "sample_buildings"
                ]
            )

            if zone_sample_size == 0:
                continue

            if pd.isna(
                climate_zone
            ):

                zone_df = type_df.loc[
                    type_df[
                        CLIMATE_ZONE_COLUMN
                    ].isna()
                ]

            else:

                zone_df = type_df.loc[
                    type_df[
                        CLIMATE_ZONE_COLUMN
                    ]
                    == climate_zone
                ]

            random_state = int(
                rng.randint(
                    0,
                    2**31 - 1,
                )
            )

            sampled_zone = (
                zone_df
                .sample(
                    n=zone_sample_size,
                    replace=False,
                    random_state=random_state,
                )
                .copy()
            )

            sampled_frames.append(
                sampled_zone
            )

    sample = pd.concat(
        sampled_frames,
        ignore_index=True,
    )

    if len(
        sample
    ) != TOTAL_SAMPLE_SIZE:

        raise ValueError(
            f"Expected "
            f"{TOTAL_SAMPLE_SIZE} sampled buildings "
            f"but received "
            f"{len(sample)}."
        )

    if sample[
        BUILDING_ID_COLUMN
    ].duplicated().any():

        raise ValueError(
            "Sample contains duplicate building IDs."
        )

    return sample


def assign_roles(
    sample: pd.DataFrame,
) -> pd.DataFrame:

    working = (
        sample
        .copy()
        .reset_index(
            drop=True
        )
    )

    rng = np.random.RandomState(
        RANDOM_STATE
        + 1
    )

    held_out_indices = []

    for _, type_group in working.groupby(
        BUILDING_TYPE_COLUMN,
        sort=True,
    ):

        type_size = len(
            type_group
        )

        raw_held_out = (
            type_size
            * HELD_OUT_SIZE
            / TOTAL_SAMPLE_SIZE
        )

        held_out_count = int(
            np.floor(
                raw_held_out
            )
        )

        if (
            type_size >= 2
            and held_out_count == 0
        ):
            held_out_count = 1

        held_out_count = min(
            held_out_count,
            type_size - 1,
        )

        if held_out_count <= 0:
            continue

        chosen = rng.choice(
            type_group.index.to_numpy(),
            size=held_out_count,
            replace=False,
        )

        held_out_indices.extend(
            chosen.tolist()
        )

    held_out_indices = list(
        dict.fromkeys(
            held_out_indices
        )
    )

    if len(
        held_out_indices
    ) < HELD_OUT_SIZE:

        remaining_candidates = (
            working.loc[
                ~working.index.isin(
                    held_out_indices
                )
            ]
            .index
            .to_numpy()
        )

        additional_count = (
            HELD_OUT_SIZE
            - len(
                held_out_indices
            )
        )

        additional = rng.choice(
            remaining_candidates,
            size=additional_count,
            replace=False,
        )

        held_out_indices.extend(
            additional.tolist()
        )

    elif len(
        held_out_indices
    ) > HELD_OUT_SIZE:

        held_out_indices = (
            rng.choice(
                np.asarray(
                    held_out_indices
                ),
                size=HELD_OUT_SIZE,
                replace=False,
            )
            .tolist()
        )

    working[
        ROLE_COLUMN
    ] = "development"

    working.loc[
        held_out_indices,
        ROLE_COLUMN,
    ] = "held_out"

    development_count = int(
        (
            working[
                ROLE_COLUMN
            ]
            == "development"
        )
        .sum()
    )

    held_out_count = int(
        (
            working[
                ROLE_COLUMN
            ]
            == "held_out"
        )
        .sum()
    )

    if development_count != DEVELOPMENT_SIZE:
        raise ValueError(
            f"Expected "
            f"{DEVELOPMENT_SIZE} development buildings "
            f"but found "
            f"{development_count}."
        )

    if held_out_count != HELD_OUT_SIZE:
        raise ValueError(
            f"Expected "
            f"{HELD_OUT_SIZE} held-out buildings "
            f"but found "
            f"{held_out_count}."
        )

    return working


ROLE_COLUMN = "sample_role"


def create_summary(
    population: pd.DataFrame,
    sample: pd.DataFrame,
) -> pd.DataFrame:

    rows = [
        {
            "metric": "population_buildings",
            "value": len(
                population
            ),
        },
        {
            "metric": "sample_buildings",
            "value": len(
                sample
            ),
        },
        {
            "metric": "development_buildings",
            "value": int(
                (
                    sample[
                        ROLE_COLUMN
                    ]
                    == "development"
                )
                .sum()
            ),
        },
        {
            "metric": "held_out_buildings",
            "value": int(
                (
                    sample[
                        ROLE_COLUMN
                    ]
                    == "held_out"
                )
                .sum()
            ),
        },
        {
            "metric": "population_building_types",
            "value": population[
                BUILDING_TYPE_COLUMN
            ].nunique(),
        },
        {
            "metric": "sample_building_types",
            "value": sample[
                BUILDING_TYPE_COLUMN
            ].nunique(),
        },
        {
            "metric": "population_climate_zones",
            "value": population[
                CLIMATE_ZONE_COLUMN
            ].nunique(),
        },
        {
            "metric": "sample_climate_zones",
            "value": sample[
                CLIMATE_ZONE_COLUMN
            ].nunique(),
        },
        {
            "metric": "sample_mean_sqft",
            "value": sample[
                SQFT_COLUMN
            ].mean(),
        },
        {
            "metric": "sample_median_sqft",
            "value": sample[
                SQFT_COLUMN
            ].median(),
        },
        {
            "metric": "sample_mean_stories",
            "value": sample[
                STORIES_COLUMN
            ].mean(),
        },
    ]

    return pd.DataFrame(
        rows
    )


def create_distribution_comparison(
    population: pd.DataFrame,
    sample: pd.DataFrame,
    column: str,
) -> pd.DataFrame:

    population_counts = (
        population[
            column
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            column
        )
        .reset_index(
            name="population_buildings"
        )
    )

    sample_counts = (
        sample[
            column
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            column
        )
        .reset_index(
            name="sample_buildings"
        )
    )

    comparison = population_counts.merge(
        sample_counts,
        on=column,
        how="outer",
    )

    comparison[
        "population_buildings"
    ] = (
        comparison[
            "population_buildings"
        ]
        .fillna(
            0
        )
        .astype(
            int
        )
    )

    comparison[
        "sample_buildings"
    ] = (
        comparison[
            "sample_buildings"
        ]
        .fillna(
            0
        )
        .astype(
            int
        )
    )

    comparison[
        "population_pct"
    ] = (
        comparison[
            "population_buildings"
        ]
        / len(
            population
        )
        * 100
    )

    comparison[
        "sample_pct"
    ] = (
        comparison[
            "sample_buildings"
        ]
        / len(
            sample
        )
        * 100
    )

    comparison[
        "percentage_point_difference"
    ] = (
        comparison[
            "sample_pct"
        ]
        - comparison[
            "population_pct"
        ]
    )

    return comparison


def main() -> None:

    print(
        "Loading California ComStock building index..."
    )

    buildings = load_buildings()

    print(
        f"Population buildings: "
        f"{len(buildings):,}"
    )

    print(
        f"Building types: "
        f"{buildings[BUILDING_TYPE_COLUMN].nunique():,}"
    )

    print(
        f"ASHRAE climate zones: "
        f"{buildings[CLIMATE_ZONE_COLUMN].nunique():,}"
    )

    print()

    print(
        "Allocating sample sizes by commercial building type..."
    )

    type_allocations = (
        allocate_building_type_sample_sizes(
            buildings
        )
    )

    print(
        type_allocations[
            [
                BUILDING_TYPE_COLUMN,
                "population_buildings",
                "sample_buildings",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "Sampling buildings with proportional "
        "climate-zone balancing within each type..."
    )

    sample = sample_buildings(
        buildings,
        type_allocations,
    )

    print(
        f"Sampled buildings: "
        f"{len(sample):,}"
    )

    print()

    print(
        "Assigning development and held-out roles..."
    )

    sample = assign_roles(
        sample
    )

    development_count = int(
        (
            sample[
                ROLE_COLUMN
            ]
            == "development"
        )
        .sum()
    )

    held_out_count = int(
        (
            sample[
                ROLE_COLUMN
            ]
            == "held_out"
        )
        .sum()
    )

    print(
        f"Development buildings: "
        f"{development_count:,}"
    )

    print(
        f"Held-out buildings: "
        f"{held_out_count:,}"
    )

    summary = create_summary(
        buildings,
        sample,
    )

    type_comparison = (
        create_distribution_comparison(
            buildings,
            sample,
            BUILDING_TYPE_COLUMN,
        )
    )

    zone_comparison = (
        create_distribution_comparison(
            buildings,
            sample,
            CLIMATE_ZONE_COLUMN,
        )
    )

    SAMPLE_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample.to_csv(
        SAMPLE_OUTPUT_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )

    type_comparison.to_csv(
        BUILDING_TYPE_COMPARISON_PATH,
        index=False,
    )

    zone_comparison.to_csv(
        CLIMATE_ZONE_COMPARISON_PATH,
        index=False,
    )

    print()

    print(
        "Building-type representation:"
    )

    print(
        type_comparison.to_string(
            index=False
        )
    )

    print()

    print(
        "Climate-zone representation:"
    )

    print(
        zone_comparison.to_string(
            index=False
        )
    )

    print()

    print(
        "ComStock modeling-sample selection complete."
    )

    print()

    print(
        f"Saved sample to: "
        f"{SAMPLE_OUTPUT_PATH}"
    )

    print(
        f"Saved summary to: "
        f"{SUMMARY_OUTPUT_PATH}"
    )

    print(
        f"Saved building-type comparison to: "
        f"{BUILDING_TYPE_COMPARISON_PATH}"
    )

    print(
        f"Saved climate-zone comparison to: "
        f"{CLIMATE_ZONE_COMPARISON_PATH}"
    )


if __name__ == "__main__":
    main()