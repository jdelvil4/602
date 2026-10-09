from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config


BUCKET = "oedi-data-lake"

PREFIXES = [
    (
        "nrel-pds-building-stock/"
        "end-use-load-profiles-for-us-building-stock/"
        "2021/comstock_amy2018_release_1/"
        "timeseries_individual_buildings/"
    ),
    (
        "nrel-pds-building-stock/"
        "end-use-load-profiles-for-us-building-stock/"
        "2021/comstock_amy2018_release_1/"
        "timeseries_individual_buildings/"
        "by_state/"
    ),
    (
        "nrel-pds-building-stock/"
        "end-use-load-profiles-for-us-building-stock/"
        "2021/comstock_amy2018_release_1/"
        "timeseries_individual_buildings/"
        "by_state/"
        "upgrade=0/"
    ),
    (
        "nrel-pds-building-stock/"
        "end-use-load-profiles-for-us-building-stock/"
        "2021/comstock_amy2018_release_1/"
        "timeseries_individual_buildings/"
        "by_state/"
        "upgrade=0/"
        "state=CA/"
    ),
]


def create_s3_client():

    return boto3.client(
        "s3",
        region_name="us-west-2",
        config=Config(
            signature_version=UNSIGNED
        ),
    )


def inspect_prefix(
    s3,
    prefix: str,
) -> None:

    print()
    print(
        "=" * 80
    )

    print(
        f"Inspecting prefix:"
    )

    print(
        prefix
    )

    print(
        "=" * 80
    )

    response = s3.list_objects_v2(
        Bucket=BUCKET,
        Prefix=prefix,
        Delimiter="/",
        MaxKeys=50,
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
        f"Common prefixes returned: "
        f"{len(common_prefixes)}"
    )

    for item in common_prefixes[:20]:

        print(
            f"  DIR  {item['Prefix']}"
        )

    print()

    print(
        f"Objects returned: "
        f"{len(objects)}"
    )

    for item in objects[:20]:

        print(
            f"  FILE {item['Key']}"
        )

    print()

    print(
        f"Is truncated: "
        f"{response.get('IsTruncated', False)}"
    )


def inspect_california_files(
    s3,
) -> None:

    ca_prefix = (
        "nrel-pds-building-stock/"
        "end-use-load-profiles-for-us-building-stock/"
        "2021/comstock_amy2018_release_1/"
        "timeseries_individual_buildings/"
        "by_state/"
        "upgrade=0/"
        "state=CA/"
    )

    print()
    print(
        "=" * 80
    )

    print(
        "Inspecting first California timeseries objects..."
    )

    print(
        "=" * 80
    )

    response = s3.list_objects_v2(
        Bucket=BUCKET,
        Prefix=ca_prefix,
        MaxKeys=20,
    )

    objects = response.get(
        "Contents",
        [],
    )

    if not objects:

        print(
            "No objects found under expected "
            "California prefix."
        )

        return

    for item in objects:

        size_mb = (
            item[
                "Size"
            ]
            / (
                1024 ** 2
            )
        )

        print(
            f"{item['Key']}"
        )

        print(
            f"  Size: "
            f"{size_mb:.3f} MB"
        )


def main() -> None:

    print(
        "Creating anonymous S3 client..."
    )

    s3 = create_s3_client()

    for prefix in PREFIXES:

        inspect_prefix(
            s3,
            prefix,
        )

    inspect_california_files(
        s3
    )

    print()

    print(
        "ComStock timeseries inspection complete."
    )


if __name__ == "__main__":
    main()