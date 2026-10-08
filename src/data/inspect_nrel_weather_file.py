from io import BytesIO
from pathlib import Path

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELING_BASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_modeling_base.parquet"
)

BUCKET = "oedi-data-lake"

WEATHER_PREFIX = (
    "nrel-pds-building-stock/"
    "end-use-load-profiles-for-us-building-stock/"
    "2021/"
    "resstock_amy2018_release_1/"
    "weather/"
    "amy2018/"
)


def create_s3_client():

    return boto3.client(
        "s3",
        region_name="us-west-2",
        config=Config(
            signature_version=UNSIGNED
        ),
    )


def load_modeling_counties() -> list[str]:

    if not MODELING_BASE_PATH.exists():
        raise FileNotFoundError(
            f"Modeling base dataset not found: "
            f"{MODELING_BASE_PATH}"
        )

    df = pd.read_parquet(
        MODELING_BASE_PATH,
        columns=[
            "in.county",
        ],
    )

    counties = (
        df[
            "in.county"
        ]
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if not counties:
        raise ValueError(
            "No county identifiers were found "
            "in the modeling base dataset."
        )

    return counties


def build_weather_key(
    county_id: str,
) -> str:

    filename = (
        f"{county_id}_2018.csv"
    )

    return (
        WEATHER_PREFIX
        + filename
    )


def load_weather_file(
    s3,
    key: str,
) -> pd.DataFrame:

    response = s3.get_object(
        Bucket=BUCKET,
        Key=key,
    )

    file_bytes = response[
        "Body"
    ].read()

    df = pd.read_csv(
        BytesIO(
            file_bytes
        )
    )

    return df


def main() -> None:

    counties = load_modeling_counties()

    print(
        f"Unique counties in modeling sample: "
        f"{len(counties):,}"
    )

    print()

    print(
        "County identifiers:"
    )

    for county in counties:

        print(
            f"  {county}"
        )

    print()

    county_id = counties[0]

    key = build_weather_key(
        county_id
    )

    print(
        f"Inspecting county: "
        f"{county_id}"
    )

    print()

    print(
        f"S3 key:\n{key}"
    )

    print()

    s3 = create_s3_client()

    try:

        df = load_weather_file(
            s3,
            key,
        )

    except Exception as exc:

        raise RuntimeError(
            f"Could not load weather file for "
            f"county {county_id}: {exc}"
        ) from exc

    print(
        f"Shape: {df.shape}"
    )

    print()

    print(
        "Columns:"
    )

    print(
        df.columns.tolist()
    )

    print()

    print(
        "Data types:"
    )

    print(
        df.dtypes.to_string()
    )

    print()

    print(
        "First rows:"
    )

    print(
        df.head(
            10
        ).to_string()
    )

    print()

    print(
        "Last rows:"
    )

    print(
        df.tail(
            10
        ).to_string()
    )

    print()

    print(
        "Missing values by column:"
    )

    print(
        df.isna()
        .sum()
        .to_string()
    )


if __name__ == "__main__":
    main()