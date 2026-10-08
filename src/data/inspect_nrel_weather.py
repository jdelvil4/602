from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config


BUCKET = "oedi-data-lake"

WEATHER_PREFIX = (
    "nrel-pds-building-stock/"
    "end-use-load-profiles-for-us-building-stock/"
    "2021/"
    "resstock_amy2018_release_1/"
    "weather/"
    "amy2018/"
)

MAX_OBJECTS_TO_SHOW = 100


def create_s3_client():

    return boto3.client(
        "s3",
        region_name="us-west-2",
        config=Config(
            signature_version=UNSIGNED
        ),
    )


def list_prefix(
    s3,
    prefix: str,
) -> None:

    print(
        f"Inspecting prefix:\n{prefix}"
    )

    print()

    response = s3.list_objects_v2(
        Bucket=BUCKET,
        Prefix=prefix,
        Delimiter="/",
        MaxKeys=1000,
    )

    common_prefixes = response.get(
        "CommonPrefixes",
        [],
    )

    objects = response.get(
        "Contents",
        [],
    )

    print(
        "Child prefixes:"
    )

    if common_prefixes:

        for item in common_prefixes:

            print(
                f"  {item['Prefix']}"
            )

    else:

        print(
            "  None"
        )

    print()

    print(
        "Objects:"
    )

    direct_objects = [
        item
        for item in objects
        if item["Key"] != prefix
    ]

    if direct_objects:

        for item in direct_objects[
            :MAX_OBJECTS_TO_SHOW
        ]:

            print(
                f"  {item['Key']} "
                f"({item['Size']:,} bytes)"
            )

    else:

        print(
            "  None"
        )

    print()

    print(
        f"Returned child prefixes: "
        f"{len(common_prefixes):,}"
    )

    print(
        f"Returned direct objects: "
        f"{len(direct_objects):,}"
    )

    print(
        f"Response truncated: "
        f"{response.get('IsTruncated', False)}"
    )


def main() -> None:

    s3 = create_s3_client()

    list_prefix(
        s3,
        WEATHER_PREFIX,
    )


if __name__ == "__main__":
    main()