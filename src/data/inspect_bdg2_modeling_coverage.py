from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bdg2_electricity_hourly.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bdg2_yearly_coverage.csv"
)


def main() -> None:

    print(
        "Loading processed BDG2 electricity data..."
    )

    data = pd.read_parquet(
        INPUT_PATH,
        columns=[
            "building_id",
            "timestamp",
            "electricity",
            "site_id",
            "primaryspaceusage",
            "sqft",
            "timezone",
        ],
    )

    data[
        "year"
    ] = data[
        "timestamp"
    ].dt.year

    yearly = (
        data
        .groupby(
            [
                "building_id",
                "year",
            ],
            observed=True,
        )
        .agg(
            site_id=(
                "site_id",
                "first",
            ),
            primaryspaceusage=(
                "primaryspaceusage",
                "first",
            ),
            sqft=(
                "sqft",
                "first",
            ),
            total_hours=(
                "timestamp",
                "size",
            ),
            observed_hours=(
                "electricity",
                "count",
            ),
            first_timestamp=(
                "timestamp",
                "min",
            ),
            last_timestamp=(
                "timestamp",
                "max",
            ),
        )
        .reset_index()
    )

    yearly[
        "missing_hours"
    ] = (
        yearly[
            "total_hours"
        ]
        - yearly[
            "observed_hours"
        ]
    )

    yearly[
        "coverage_percentage"
    ] = (
        yearly[
            "observed_hours"
        ]
        / yearly[
            "total_hours"
        ]
        * 100.0
    )

    yearly.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "BDG2 yearly coverage:"
    )

    for year in sorted(
        yearly[
            "year"
        ].unique()
    ):

        year_data = yearly[
            yearly[
                "year"
            ] == year
        ]

        print()
        print(
            f"Year: {year}"
        )

        print(
            f"Buildings represented: "
            f"{year_data['building_id'].nunique():,}"
        )

        for threshold in [
            90,
            95,
            98,
            99,
            100,
        ]:

            count = int(
                (
                    year_data[
                        "coverage_percentage"
                    ]
                    >= threshold
                ).sum()
            )

            print(
                f"Buildings >= "
                f"{threshold}% coverage: "
                f"{count:,}"
            )

        print()

        print(
            "Coverage distribution:"
        )

        print(
            year_data[
                "coverage_percentage"
            ]
            .describe()
            .to_string()
        )

        print()

        print(
            "Primary space usage among "
            "buildings with >=95% coverage:"
        )

        high_coverage = year_data[
            year_data[
                "coverage_percentage"
            ]
            >= 95.0
        ]

        print(
            high_coverage[
                "primaryspaceusage"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    print()
    print(
        f"Saved yearly coverage to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()