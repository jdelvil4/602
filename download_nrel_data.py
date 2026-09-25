import boto3
from botocore import UNSIGNED
from botocore.config import Config
from pathlib import Path

BUCKET = "oedi-data-lake"

s3 = boto3.client(
    "s3",
    region_name="us-west-2",
    config=Config(signature_version=UNSIGNED)
)

downloads = {
    "data/resstock_metadata.parquet":
        "nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/"
        "2021/resstock_amy2018_release_1/metadata/metadata.parquet",

    "data/comstock_metadata.parquet":
        "nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/"
        "2021/comstock_amy2018_release_1/metadata/metadata.parquet",

    "data/resstock_data_dictionary.tsv":
        "nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/"
        "2021/resstock_amy2018_release_1/data_dictionary.tsv",

    "data/comstock_data_dictionary.tsv":
        "nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/"
        "2021/comstock_amy2018_release_1/data_dictionary.tsv",
}

for local_path, s3_key in downloads.items():
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {s3_key}")
    s3.download_file(BUCKET, s3_key, local_path)

print("Downloads complete.")