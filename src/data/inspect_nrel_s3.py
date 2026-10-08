from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config


BUCKET = "oedi-data-lake"

PREFIX = ("nrel-pds-building-stock/"
    "end-use-load-profiles-for-us-building-stock/"
    "2021/"
    "resstock_amy2018_release_1/")


def main() -> None:

    s3 = boto3.client("s3", region_name="us-west-2", config=Config(signature_version=UNSIGNED),)

    paginator = s3.get_paginator("list_objects_v2")

    pages = paginator.paginate(
        Bucket=BUCKET,
        Prefix=PREFIX,
        Delimiter="/",
    )

    print(f"Inspecting S3 prefix:\n{PREFIX}\n")

    found_prefixes = []

    for page in pages:

        for item in page.get("CommonPrefixes",[],):
            prefix = item["Prefix"]
            found_prefixes.append(prefix)

    if not found_prefixes:
        print("No child prefixes were found.")
        return

    print("Available child prefixes:")

    for prefix in found_prefixes:
        print(prefix)


if __name__ == "__main__":
    main()