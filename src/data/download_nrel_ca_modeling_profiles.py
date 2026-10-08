from pathlib import Path

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BUCKET = "oedi-data-lake"

S3_PREFIX = (
    "nrel-pds-building-stock/"
    "end-use-load-profiles-for-us-building-stock/"
    "2021/"
    "resstock_amy2018_release_1/"
    "timeseries_individual_buildings/"
    "by_state/"
    "upgrade=0/"
    "state=CA/"
)

SAMPLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_modeling_sample.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nrel"
    / "resstock_ca"
    / "upgrade=0"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_modeling_download_manifest.csv"
)


EXPECTED_SAMPLE_SIZE = 250


def create_s3_client():

    return boto3.client(
        "s3",
        region_name="us-west-2",
        config=Config(
            signature_version=UNSIGNED
        ),
    )


def load_modeling_sample() -> pd.DataFrame:

    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"Modeling sample not found: {SAMPLE_PATH}"
        )

    sample = pd.read_csv(
        SAMPLE_PATH
    )

    required_columns = {
        "bldg_id",
        "sample_role",
    }

    missing_columns = (
        required_columns
        - set(sample.columns)
    )

    if missing_columns:
        raise ValueError(
            "Modeling sample is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    sample["bldg_id"] = pd.to_numeric(
        sample["bldg_id"],
        errors="coerce",
    )

    if sample["bldg_id"].isna().any():
        raise ValueError(
            "Modeling sample contains invalid building IDs."
        )

    sample["bldg_id"] = (
        sample["bldg_id"]
        .astype("int64")
    )

    duplicate_count = int(
        sample["bldg_id"]
        .duplicated()
        .sum()
    )

    if duplicate_count != 0:
        raise ValueError(
            f"Duplicate building IDs found: {duplicate_count}"
        )

    if len(sample) != EXPECTED_SAMPLE_SIZE:
        raise ValueError(
            f"Modeling sample contains {len(sample)} buildings; "
            f"expected {EXPECTED_SAMPLE_SIZE}."
        )

    valid_roles = {
        "development",
        "held_out",
    }

    invalid_roles = (
        set(
            sample["sample_role"]
            .unique()
        )
        - valid_roles
    )

    if invalid_roles:
        raise ValueError(
            "Unexpected sample roles found: "
            f"{sorted(invalid_roles)}"
        )

    return (
        sample
        .sort_values(
            "bldg_id"
        )
        .reset_index(drop=True)
    )


def get_remote_size(
    s3,
    key: str,
) -> int:

    response = s3.head_object(
        Bucket=BUCKET,
        Key=key,
    )

    return int(
        response["ContentLength"]
    )


def download_profile(
    s3,
    bldg_id: int,
    sample_role: str,
) -> dict:

    filename = (
        f"{bldg_id}-0.parquet"
    )

    key = (
        S3_PREFIX
        + filename
    )

    local_path = (
        OUTPUT_DIR
        / filename
    )

    result = {
        "bldg_id": bldg_id,
        "sample_role": sample_role,
        "filename": filename,
        "s3_key": key,
        "local_path": str(
            local_path
        ),
        "remote_size_bytes": None,
        "local_size_bytes": None,
        "status": None,
        "error": None,
    }

    try:

        remote_size = get_remote_size(
            s3,
            key,
        )

        result[
            "remote_size_bytes"
        ] = remote_size

        if local_path.exists():

            local_size = (
                local_path.stat().st_size
            )

            if local_size == remote_size:

                result[
                    "local_size_bytes"
                ] = local_size

                result[
                    "status"
                ] = "already_exists"

                return result

            local_path.unlink()

        s3.download_file(
            BUCKET,
            key,
            str(
                local_path
            ),
        )

        local_size = (
            local_path.stat().st_size
        )

        result[
            "local_size_bytes"
        ] = local_size

        if local_size != remote_size:

            result[
                "status"
            ] = "size_mismatch"

            result[
                "error"
            ] = (
                f"Expected {remote_size} bytes, "
                f"downloaded {local_size} bytes."
            )

            return result

        result[
            "status"
        ] = "downloaded"

        return result

    except ClientError as exc:

        result[
            "status"
        ] = "failed"

        result[
            "error"
        ] = str(
            exc
        )

        return result

    except Exception as exc:

        result[
            "status"
        ] = "failed"

        result[
            "error"
        ] = str(
            exc
        )

        return result


def save_manifest(
    results: list[dict],
) -> pd.DataFrame:

    manifest = pd.DataFrame(
        results
    )

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest.to_csv(
        MANIFEST_PATH,
        index=False,
    )

    return manifest


def print_summary(
    manifest: pd.DataFrame,
) -> None:

    usable_statuses = {
        "downloaded",
        "already_exists",
    }

    usable = manifest[
        manifest[
            "status"
        ].isin(
            usable_statuses
        )
    ]

    failed = manifest[
        ~manifest[
            "status"
        ].isin(
            usable_statuses
        )
    ]

    total_bytes = int(
        usable[
            "local_size_bytes"
        ]
        .fillna(0)
        .sum()
    )

    print()

    print(
        "NREL modeling profile download complete."
    )

    print()

    print(
        f"Requested buildings: {len(manifest):,}"
    )

    print(
        f"Usable profiles: {len(usable):,}"
    )

    print(
        f"Failed profiles: {len(failed):,}"
    )

    print()

    print(
        "Status counts:"
    )

    print(
        manifest[
            "status"
        ]
        .value_counts()
        .to_string()
    )

    print()

    print(
        "Usable profiles by sample role:"
    )

    print(
        usable[
            "sample_role"
        ]
        .value_counts()
        .to_string()
    )

    print()

    print(
        f"Total local size: "
        f"{total_bytes / 1_048_576:.2f} MB"
    )

    print()

    print(
        f"Profiles directory: "
        f"{OUTPUT_DIR}"
    )

    print(
        f"Manifest: "
        f"{MANIFEST_PATH}"
    )

    if not failed.empty:

        print()

        print(
            "Failed or invalid downloads:"
        )

        print(
            failed[
                [
                    "bldg_id",
                    "sample_role",
                    "status",
                    "error",
                ]
            ]
            .to_string(
                index=False
            )
        )


def main() -> None:

    sample = load_modeling_sample()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    s3 = create_s3_client()

    results = []

    print(
        f"Modeling buildings to acquire: "
        f"{len(sample):,}"
    )

    print()

    for position, row in enumerate(
        sample.itertuples(
            index=False
        ),
        start=1,
    ):

        bldg_id = int(
            str(
                row.bldg_id
            )
        )

        sample_role = str(
            row.sample_role
        )

        print(
            f"[{position}/{len(sample)}] "
            f"Building {bldg_id} "
            f"({sample_role})"
        )

        result = download_profile(
            s3,
            bldg_id,
            sample_role,
        )

        results.append(
            result
        )

        print(
            f"    {result['status']}"
        )

    manifest = save_manifest(
        results
    )

    print_summary(
        manifest
    )


if __name__ == "__main__":
    main()
    