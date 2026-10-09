from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bdg2_yearly_coverage.csv"
)

OUTPUT_SAMPLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bdg2_modeling_sample.csv"
)

OUTPUT_SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bdg2_modeling_sample_summary.csv"
)

YEAR = 2017
MIN_COVERAGE = 95.0
TOTAL_SAMPLE = 250
DEVELOPMENT_SAMPLE = 200
HELD_OUT_SAMPLE = 50
RANDOM_STATE = 42


def allocate_proportionally(
    counts: pd.Series,
    total: int,
) -> pd.Series:

    proportions = (
        counts
        / counts.sum()
    )

    exact = (
        proportions
        * total
    )

    allocation = (
        np.floor(
            exact
        )
        .astype(
            int
        )
    )

    allocation = pd.Series(
        allocation,
        index=counts.index,
    )

    remaining = (
        total
        - int(
            allocation.sum()
        )
    )

    if remaining > 0:

        remainders = (
            exact
            - allocation
        ).sort_values(
            ascending=False
        )

        for category in (
            remainders
            .index[
                :remaining
            ]
        ):
            allocation.loc[
                category
            ] += 1

    allocation = pd.concat(
        [
            allocation,
            counts,
        ],
        axis=1,
    )

    allocation.columns = [
        "allocated",
        "available",
    ]

    allocation[
        "allocated"
    ] = allocation[
        [
            "allocated",
            "available",
        ]
    ].min(
        axis=1
    )

    while (
        int(
            allocation[
                "allocated"
            ].sum()
        )
        < total
    ):

        remaining_capacity = (
            allocation[
                "available"
            ]
            - allocation[
                "allocated"
            ]
        )

        eligible = (
            remaining_capacity[
                remaining_capacity > 0
            ]
        )

        if eligible.empty:
            raise ValueError(
                "Unable to allocate requested sample."
            )

        category = (
            eligible
            .sort_values(
                ascending=False
            )
            .index[
                0
            ]
        )

        allocation.loc[
            category,
            "allocated",
        ] += 1

    return allocation[
        "allocated"
    ].astype(
        int
    )


def sample_by_category(
    data: pd.DataFrame,
    allocations: pd.Series,
    random_state: int,
) -> pd.DataFrame:

    sampled_frames: list[
        pd.DataFrame
    ] = []

    for category, sample_size in (
        allocations.items()
    ):

        if sample_size <= 0:
            continue

        group = data[
            data[
                "primaryspaceusage"
            ] == category
        ]

        sampled = group.sample(
            n=int(
                sample_size
            ),
            random_state=random_state,
        )

        sampled_frames.append(
            sampled
        )

    result = pd.concat(
        sampled_frames,
        ignore_index=True,
    )

    return result


def assign_modeling_split(
    sample: pd.DataFrame,
) -> pd.DataFrame:

    category_counts = (
        sample[
            "primaryspaceusage"
        ]
        .value_counts()
        .sort_index()
    )

    held_out_allocations = (
        allocate_proportionally(
            category_counts,
            HELD_OUT_SAMPLE,
        )
    )

    held_out = (
        sample_by_category(
            sample,
            held_out_allocations,
            RANDOM_STATE + 1,
        )
    )

    held_out_ids = set(
        held_out[
            "building_id"
        ]
    )

    result = sample.copy()

    result[
        "sample_group"
    ] = "development"

    result.loc[
        result[
            "building_id"
        ].isin(
            held_out_ids
        ),
        "sample_group",
    ] = "held_out"

    return result


def main() -> None:

    print(
        "Loading BDG2 yearly coverage..."
    )

    coverage = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    coverage[
        "year"
    ] = pd.to_numeric(
        coverage[
            "year"
        ],
        errors="coerce",
    )

    coverage[
        "coverage_percentage"
    ] = pd.to_numeric(
        coverage[
            "coverage_percentage"
        ],
        errors="coerce",
    )

    eligible = coverage[
        (
            coverage[
                "year"
            ] == YEAR
        )
        & (
            coverage[
                "coverage_percentage"
            ] >= MIN_COVERAGE
        )
        & (
            coverage[
                "primaryspaceusage"
            ].notna()
        )
    ].copy()

    print(
        f"Eligible 2017 buildings with "
        f">={MIN_COVERAGE:.0f}% coverage: "
        f"{len(eligible):,}"
    )

    if len(eligible) < TOTAL_SAMPLE:
        raise ValueError(
            f"Only {len(eligible):,} buildings "
            f"meet selection criteria."
        )

    category_counts = (
        eligible[
            "primaryspaceusage"
        ]
        .value_counts()
        .sort_index()
    )

    allocations = (
        allocate_proportionally(
            category_counts,
            TOTAL_SAMPLE,
        )
    )

    print()
    print(
        "Sample allocation by "
        "primary space usage:"
    )

    allocation_report = pd.DataFrame(
        {
            "available":
                category_counts,
            "sampled":
                allocations,
        }
    )

    print(
        allocation_report.to_string()
    )

    sample = sample_by_category(
        eligible,
        allocations,
        RANDOM_STATE,
    )

    sample = assign_modeling_split(
        sample
    )

    sample = sample.sort_values(
        [
            "sample_group",
            "primaryspaceusage",
            "building_id",
        ]
    ).reset_index(
        drop=True
    )

    development_count = int(
        (
            sample[
                "sample_group"
            ]
            == "development"
        ).sum()
    )

    held_out_count = int(
        (
            sample[
                "sample_group"
            ]
            == "held_out"
        ).sum()
    )

    if (
        development_count
        != DEVELOPMENT_SAMPLE
    ):
        raise ValueError(
            f"Expected {DEVELOPMENT_SAMPLE} "
            f"development buildings, "
            f"found {development_count}."
        )

    if (
        held_out_count
        != HELD_OUT_SAMPLE
    ):
        raise ValueError(
            f"Expected {HELD_OUT_SAMPLE} "
            f"held-out buildings, "
            f"found {held_out_count}."
        )

    if (
        sample[
            "building_id"
        ].duplicated().any()
    ):
        raise ValueError(
            "Duplicate building IDs found."
        )

    summary = (
        sample
        .groupby(
            [
                "sample_group",
                "primaryspaceusage",
            ],
            observed=True,
        )
        .agg(
            buildings=(
                "building_id",
                "nunique",
            ),
            mean_coverage_percentage=(
                "coverage_percentage",
                "mean",
            ),
            median_coverage_percentage=(
                "coverage_percentage",
                "median",
            ),
            median_sqft=(
                "sqft",
                "median",
            ),
        )
        .reset_index()
    )

    OUTPUT_SAMPLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample.to_csv(
        OUTPUT_SAMPLE_PATH,
        index=False,
    )

    summary.to_csv(
        OUTPUT_SUMMARY_PATH,
        index=False,
    )

    print()
    print(
        "BDG2 modeling sample selected."
    )

    print(
        f"Total buildings: "
        f"{len(sample):,}"
    )

    print(
        f"Development buildings: "
        f"{development_count:,}"
    )

    print(
        f"Held-out buildings: "
        f"{held_out_count:,}"
    )

    print(
        f"Mean coverage: "
        f"{sample['coverage_percentage'].mean():.3f}%"
    )

    print(
        f"Minimum coverage: "
        f"{sample['coverage_percentage'].min():.3f}%"
    )

    print()

    print(
        "Selected building types:"
    )

    print(
        sample[
            "primaryspaceusage"
        ]
        .value_counts()
        .to_string()
    )

    print()

    print(
        "Development / held-out distribution:"
    )

    print(
        pd.crosstab(
            sample[
                "primaryspaceusage"
            ],
            sample[
                "sample_group"
            ],
        ).to_string()
    )

    print()

    print(
        f"Saved sample to: "
        f"{OUTPUT_SAMPLE_PATH}"
    )

    print(
        f"Saved summary to: "
        f"{OUTPUT_SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()