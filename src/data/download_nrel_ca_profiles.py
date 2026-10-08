from pathlib import Path
import sys

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT),)


BUCKET = "oedi-data-lake"

PREFIX = ("nrel-pds-building-stock/"
    "end-use-load-profiles-for-us-building-stock/"
    "2021/"
    "resstock_amy2018_release_1/"
    "timeseries_individual_buildings/"
    "by_state/"
    "upgrade=0/"
    "state=CA/")

BUILDING_INDEX_PATH = (PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_resstock_buildings.csv")

OUTPUT_DIR = (PROJECT_ROOT
    / "data"
    / "raw"
    / "nrel"
    / "resstock_ca"
    / "upgrade=0")

MANIFEST_PATH = (PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_pilot_download_manifest.csv")


PILOT_BUILDING_COUNT = 25
RANDOM_STATE = 42


def load_building_ids() -> pd.Series:

    if not BUILDING_INDEX_PATH.exists():
        raise FileNotFoundError(f"Building index not found: {BUILDING_INDEX_PATH}")

    df = pd.read_csv(BUILDING_INDEX_PATH)

    if "bldg_id" not in df.columns:
        raise ValueError("The building index does not contain bldg_id.")

    building_ids = (pd.to_numeric(df.loc[:, "bldg_id"],errors="coerce",)
        .dropna()
        .astype("int64")
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True))

    if len(building_ids) < PILOT_BUILDING_COUNT:
        raise ValueError("Not enough California buildings are available " 
                f"for a {PILOT_BUILDING_COUNT}-building pilot.")

    return building_ids


def select_pilot_buildings(building_ids: pd.Series,) -> pd.Series:

    sampled = (building_ids.sample(n=PILOT_BUILDING_COUNT,
            random_state=RANDOM_STATE,replace=False,).sort_values().reset_index(drop=True))

    return sampled


def create_s3_client():

    return boto3.client("s3",region_name="us-west-2",config=Config(signature_version=UNSIGNED),)


def download_profiles(s3,building_ids: pd.Series,) -> pd.DataFrame:

    OUTPUT_DIR.mkdir(parents=True,exist_ok=True,)

    records = []

    for position, bldg_id in enumerate(building_ids,start=1,):

        filename = (f"{int(bldg_id)}-0.parquet")

        s3_key = (PREFIX + filename)

        local_path = (OUTPUT_DIR
            / filename)

        print(f"[{position}/{len(building_ids)}] "
            f"Building {int(bldg_id)}")

        if local_path.exists():

            print(f"  Already exists: {local_path.name}")

            records.append(
                {
                    "bldg_id": int(bldg_id),
                    "filename": filename,
                    "s3_key": s3_key,
                    "local_path": str(local_path),
                    "status": "already_exists",
                    "file_size_bytes": local_path.stat().st_size,
                }
            )

            continue

        try:

            response = s3.head_object(
                Bucket=BUCKET,
                Key=s3_key,
            )

            expected_size = int(
                response.get(
                    "ContentLength",
                    0,
                )
            )

        except ClientError as exc:

            error_code = (
                exc.response
                .get("Error",{},)
                .get("Code","Unknown",))

            print(f"  Remote file not found or unavailable: "
                f"{error_code}")

            records.append(
                {
                    "bldg_id": int(bldg_id),
                    "filename": filename,
                    "s3_key": s3_key,
                    "local_path": str(local_path),
                    "status": "remote_missing",
                    "file_size_bytes": 0,
                }
            )

            continue

        print(f"  Downloading {expected_size / 1_048_576:.2f} MB")

        try:

            s3.download_file(BUCKET,s3_key,str(local_path),)

            actual_size = (local_path.stat().st_size)

            if (expected_size > 0
                and actual_size != expected_size):
                raise ValueError(f"Downloaded size mismatch for {filename}: "
                    f"expected {expected_size}, "
                    f"received {actual_size}")

            print(f"  Saved: {local_path.name}")

            records.append(
                {
                    "bldg_id": int(bldg_id),
                    "filename": filename,
                    "s3_key": s3_key,
                    "local_path": str(local_path),
                    "status": "downloaded",
                    "file_size_bytes": actual_size,
                }
            )

        except Exception as exc:

            if local_path.exists():
                local_path.unlink()

            print(
                f"  Download failed: {exc}"
            )

            records.append(
                {
                    "bldg_id": int(bldg_id),
                    "filename": filename,
                    "s3_key": s3_key,
                    "local_path": str(local_path),
                    "status": "download_failed",
                    "file_size_bytes": 0,
                }
            )

    return pd.DataFrame(
        records
    )


def save_manifest(
    manifest_df: pd.DataFrame,
) -> None:

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_df.to_csv(
        MANIFEST_PATH,
        index=False,
    )


def print_summary(
    manifest_df: pd.DataFrame,
) -> None:

    print()
    print(
        "Pilot download summary"
    )
    print(
        "----------------------"
    )

    print(
        f"Requested buildings: "
        f"{len(manifest_df):,}"
    )

    status_counts = (
        manifest_df.loc[:, "status"]
        .value_counts(
            dropna=False
        )
    )

    print()

    print(
        status_counts.to_string()
    )

    successful_mask = (
        manifest_df.loc[:, "status"].isin(
            ["downloaded","already_exists",]))

    successful_df = (
        manifest_df.loc[successful_mask])

    total_bytes = int(
        successful_df.loc[:,"file_size_bytes",].sum())

    print()

    print(f"Usable downloaded profiles: "f"{len(successful_df):,}")

    print(
        f"Total local size: "
        f"{total_bytes / 1_048_576:.2f} MB"
    )

    print()

    print(f"Manifest saved to: "f"{MANIFEST_PATH}")

    print(
        f"Profiles saved to: "
        f"{OUTPUT_DIR}"
    )


def main() -> None:

    building_ids = load_building_ids()

    print(
        f"California buildings available: "
        f"{len(building_ids):,}"
    )

    pilot_ids = select_pilot_buildings(
        building_ids
    )

    print(
        f"Pilot buildings selected: "
        f"{len(pilot_ids):,}"
    )

    print(
        f"Random state: "
        f"{RANDOM_STATE}"
    )

    print()

    s3 = create_s3_client()

    manifest_df = download_profiles(s3,pilot_ids,)

    save_manifest(manifest_df)

    print_summary(manifest_df)


if __name__ == "__main__":
    main()