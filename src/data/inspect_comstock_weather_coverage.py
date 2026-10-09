from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

COMSTOCK_BASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comstock_ca_modeling_base.parquet"
)

WEATHER_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nrel"
    / "weather"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comstock_ca_weather_coverage.csv"
)


COUNTY_COLUMN = "in.county"
BUILDING_ID_COLUMN = "bldg_id"


def load_comstock_counties() -> pd.DataFrame:

    if not COMSTOCK_BASE_PATH.exists():
        raise FileNotFoundError(
            f"ComStock modeling base not found: "
            f"{COMSTOCK_BASE_PATH}"
        )

    df = pd.read_parquet(
        COMSTOCK_BASE_PATH,
        columns=[
            BUILDING_ID_COLUMN,
            COUNTY_COLUMN,
        ],
    )

    building_counties = (
        df[
            [
                BUILDING_ID_COLUMN,
                COUNTY_COLUMN,
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    if building_counties[
        BUILDING_ID_COLUMN
    ].duplicated().any():
        raise ValueError(
            "A ComStock building is associated "
            "with more than one county."
        )

    if building_counties[
        COUNTY_COLUMN
    ].isna().any():
        missing_count = int(
            building_counties[
                COUNTY_COLUMN
            ]
            .isna()
            .sum()
        )

        raise ValueError(
            f"ComStock sample contains "
            f"{missing_count:,} buildings "
            "with missing county values."
        )

    building_counties[
        COUNTY_COLUMN
    ] = building_counties[
        COUNTY_COLUMN
    ].astype(
        "string"
    )

    return building_counties


def find_weather_files() -> dict[str, Path]:

    if not WEATHER_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Weather directory not found: "
            f"{WEATHER_DIRECTORY}"
        )

    weather_files: dict[str, Path] = {}

    for path in WEATHER_DIRECTORY.rglob(
        "*_2018.csv"
    ):

        county_code = (
            path.stem
            .replace(
                "_2018",
                "",
            )
        )

        weather_files[
            county_code
        ] = path

    return weather_files


def get_weather_file(
    county_value: object,
    weather_files: dict[str, Path],
) -> str:

    county = str(
        county_value
    )

    weather_path = weather_files.get(
        county
    )

    if weather_path is None:
        return ""

    return str(
        weather_path
    )


def main() -> None:

    print(
        "Loading ComStock sample counties..."
    )

    building_counties = load_comstock_counties()

    county_counts = (
        building_counties[
            COUNTY_COLUMN
        ]
        .value_counts()
        .rename_axis(
            COUNTY_COLUMN
        )
        .reset_index(
            name="sample_buildings"
        )
    )

    print(
        f"Sample buildings: "
        f"{len(building_counties):,}"
    )

    print(
        f"Unique ComStock counties: "
        f"{len(county_counts):,}"
    )

    print()

    print(
        "Searching existing weather directory:"
    )

    print(
        WEATHER_DIRECTORY
    )

    weather_files = find_weather_files()

    print(
        f"Existing 2018 weather files found: "
        f"{len(weather_files):,}"
    )

    print()

    county_counts[
        COUNTY_COLUMN
    ] = county_counts[
        COUNTY_COLUMN
    ].astype(
        "string"
    )

    weather_keys = list(
        weather_files.keys()
    )

    county_counts[
        "weather_file_found"
    ] = county_counts[
        COUNTY_COLUMN
    ].isin(
        weather_keys
    )

    county_counts[
        "weather_file"
    ] = county_counts[
        COUNTY_COLUMN
    ].map(
        lambda value: get_weather_file(
            value,
            weather_files,
        )
    )

    covered = int(
        county_counts[
            "weather_file_found"
        ].sum()
    )

    missing = county_counts.loc[
        ~county_counts[
            "weather_file_found"
        ]
    ].copy()

    covered_buildings = int(
        county_counts.loc[
            county_counts[
                "weather_file_found"
            ],
            "sample_buildings",
        ].sum()
    )

    missing_buildings = int(
        county_counts.loc[
            ~county_counts[
                "weather_file_found"
            ],
            "sample_buildings",
        ].sum()
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    county_counts.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "ComStock weather coverage:"
    )

    print(
        f"  Counties covered: "
        f"{covered:,} / {len(county_counts):,}"
    )

    print(
        f"  Buildings covered: "
        f"{covered_buildings:,} / "
        f"{len(building_counties):,}"
    )

    print(
        f"  Buildings missing weather coverage: "
        f"{missing_buildings:,}"
    )

    if not missing.empty:

        print()

        print(
            "Counties missing existing weather files:"
        )

        print(
            missing[
                [
                    COUNTY_COLUMN,
                    "sample_buildings",
                ]
            ]
            .to_string(
                index=False
            )
        )

    else:

        print()

        print(
            "All ComStock sample counties already "
            "have local weather files."
        )

    print()

    print(
        "County coverage table:"
    )

    print(
        county_counts.to_string(
            index=False
        )
    )

    print()

    print(
        f"Saved coverage table to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()