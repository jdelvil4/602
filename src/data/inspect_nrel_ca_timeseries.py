import boto3
from botocore import UNSIGNED
from botocore.config import Config


BUCKET = "oedi-data-lake"

PREFIX = (
    "nrel-pds-building-stock/"
    "end-use-load-profiles-for-us-building-stock/"
    "2021/"
    "resstock_amy2018_release_1/"
    "timeseries_individual_buildings/"
    "by_state/"
    "upgrade=0/"
    "state=CA/"
)


def main() -> None:

    s3 = boto3.client("s3", region_name="us-west-2",config=Config(signature_version=UNSIGNED),)

    print(f"Inspecting California ResStock time series:\n"f"{PREFIX}\n")

    response = s3.list_objects_v2(Bucket=BUCKET,Prefix=PREFIX,Delimiter="/",MaxKeys=100,)

    child_prefixes = [item["Prefix"]
        for item in response.get("CommonPrefixes",[],)]

    objects = response.get("Contents",[],)

    if child_prefixes:

        print("Child prefixes:")

        for prefix in child_prefixes:
            print(prefix)

    if objects:

        print()

        print("Example objects:")

        for item in objects[:25]:

            print(f"{item['Key']} "f"({item['Size']:,} bytes)")

    if not child_prefixes and not objects:

        print("No child prefixes or objects found.")

    print()

    print(
        f"Objects returned in this request: "
        f"{len(objects):,}")

    print(f"Is response truncated: "
        f"{response.get('IsTruncated', False)}")


if __name__ == "__main__":
    main()