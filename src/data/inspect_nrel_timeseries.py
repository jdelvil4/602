import boto3
from botocore import UNSIGNED
from botocore.config import Config


BUCKET = "oedi-data-lake"

PREFIX = ("nrel-pds-building-stock/"
    "end-use-load-profiles-for-us-building-stock/"
    "2021/"
    "resstock_amy2018_release_1/"
    "timeseries_individual_buildings/")


def main() -> None:

    s3 = boto3.client("s3", region_name="us-west-2", config=Config(signature_version=UNSIGNED),)

    print(f"Inspecting:\n{PREFIX}\n")

    response = s3.list_objects_v2(Bucket=BUCKET,Prefix=PREFIX,Delimiter="/",MaxKeys=100,)

    common_prefixes = response.get("CommonPrefixes",[],)

    if common_prefixes:
        print("Child prefixes:")

        for item in common_prefixes:print(item["Prefix"])

    contents = response.get("Contents", [],)

    if contents:
        print()
        print("Example objects:")

        for item in contents[:20]: print(item["Key"])

    if not common_prefixes and not contents:
        print("No objects or child prefixes found.")



if __name__ == "__main__":
    main()