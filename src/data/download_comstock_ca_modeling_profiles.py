from pathlib import Path
from time import perf_counter

import boto3
import pandas as pd
import pyarrow.parquet as pq
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SAMPLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comstock_ca_modeling_sample.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nrel"
    / "comstock_ca"
    / "upgrade=0"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comstock_ca_modeling_download_manifest.csv"
)


BUCKET = "oedi-data-lake"

PREFIX = (
    "nrel-pds-building-stock/"
    "end-use-load-profiles-for-us-building-stock/"
    "2021/comstock_amy2018_release_1/"
    "timeseries_individual_buildings/"
    "by_state/"
    "upgrade=0/"
    "state=CA/"
)


BUILDING_ID_COLUMN = "bldg_id"
ROLE_COLUMN = "sample_role"

EXPECTED_BUILDINGS = 250
EXPECTED_DEVELOPMENT = 200
EXPECTED_HELD_OUT = 50


def create_s3_client():

    return boto3.client(
        "s3",
        region_name="us-west-2",
        config=Config(
            signature_version=UNSIGNED,
            retries={
                "max_attempts": 10,
                "mode": "standard",
            },
        ),
    )


def load_sample() -> pd.DataFrame:

    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"ComStock modeling sample not found: "
            f"{SAMPLE_PATH}"
        )

    sample = pd.read_csv(
        SAMPLE_PATH
    )

    required_columns = {
        BUILDING_ID_COLUMN,
        ROLE_COLUMN,
    }

    missing_columns = (
        required_columns
        - set(sample.columns)
    )

    if missing_columns:
        raise ValueError(
            "ComStock modeling sample is missing "
            "required columns: "
            f"{sorted(missing_columns)}"
        )

    if sample[
        BUILDING_ID_COLUMN
    ].duplicated().any():
        raise ValueError(
            "ComStock modeling sample contains "
            "duplicate building IDs."
        )

    if len(sample) != EXPECTED_BUILDINGS:
        raise ValueError(
            f"Expected {EXPECTED_BUILDINGS} buildings "
            f"but found {len(sample)}."
        )

    development_count = int(
        (
            sample[
                ROLE_COLUMN
            ]
            == "development"
        )
        .sum()
    )

    held_out_count = int(
        (
            sample[
                ROLE_COLUMN
            ]
            == "held_out"
        )
        .sum()
    )

    if development_count != EXPECTED_DEVELOPMENT:
        raise ValueError(
            f"Expected {EXPECTED_DEVELOPMENT} "
            "development buildings but found "
            f"{development_count}."
        )

    if held_out_count != EXPECTED_HELD_OUT:
        raise ValueError(
            f"Expected {EXPECTED_HELD_OUT} "
            "held-out buildings but found "
            f"{held_out_count}."
        )

    return (
        sample
        .sort_values(
            BUILDING_ID_COLUMN
        )
        .reset_index(
            drop=True
        )
    )


def build_s3_key(
    bldg_id: int,
) -> str:

    return (
        f"{PREFIX}"
        f"{bldg_id}-0.parquet"
    )


def build_local_path(
    bldg_id: int,
) -> Path:

    return (
        OUTPUT_DIRECTORY
        / f"{bldg_id}-0.parquet"
    )


def check_remote_object(
    s3,
    key: str,
) -> dict:

    try:

        response = s3.head_object(
            Bucket=BUCKET,
            Key=key,
        )

        return {
            "exists": True,
            "size_bytes": int(
                response[
                    "ContentLength"
                ]
            ),
        }

    except ClientError as error:

        error_code = str(
            error.response[
                "Error"
            ].get(
                "Code",
                ""
            )
        )

        if error_code in {
            "404",
            "NoSuchKey",
            "NotFound",
        }:

            return {
                "exists": False,
                "size_bytes": 0,
            }

        raise


def validate_local_parquet(
    path: Path,
) -> dict:

    if not path.exists():

        return {
            "valid_parquet": False,
            "rows": 0,
            "columns": 0,
        }

    try:

        metadata = pq.read_metadata(
            path
        )

        return {
            "valid_parquet": True,
            "rows": int(
                metadata.num_rows
            ),
            "columns": int(
                metadata.num_columns
            ),
        }

    except Exception:

        return {
            "valid_parquet": False,
            "rows": 0,
            "columns": 0,
        }


def download_building(
    s3,
    bldg_id: int,
    sample_role: str,
) -> dict:

    key = build_s3_key(
        bldg_id
    )

    local_path = build_local_path(
        bldg_id
    )

    remote = check_remote_object(
        s3,
        key,
    )

    if not remote[
        "exists"
    ]:

        return {
            BUILDING_ID_COLUMN: bldg_id,
            ROLE_COLUMN: sample_role,
            "s3_key": key,
            "local_path": str(
                local_path
            ),
            "status": "missing_remote",
            "remote_size_bytes": 0,
            "local_size_bytes": 0,
            "rows": 0,
            "columns": 0,
        }

    expected_size = int(
        remote[
            "size_bytes"
        ]
    )

    if local_path.exists():

        existing_size = int(
            local_path.stat().st_size
        )

        existing_validation = (
            validate_local_parquet(
                local_path
            )
        )

        if (
            existing_size
            == expected_size
            and existing_validation[
                "valid_parquet"
            ]
        ):

            return {
                BUILDING_ID_COLUMN: bldg_id,
                ROLE_COLUMN: sample_role,
                "s3_key": key,
                "local_path": str(
                    local_path
                ),
                "status": "already_present",
                "remote_size_bytes": (
                    expected_size
                ),
                "local_size_bytes": (
                    existing_size
                ),
                "rows": existing_validation[
                    "rows"
                ],
                "columns": existing_validation[
                    "columns"
                ],
            }

    try:

        s3.download_file(
            BUCKET,
            key,
            str(
                local_path
            ),
        )

    except Exception as error:

        return {
            BUILDING_ID_COLUMN: bldg_id,
            ROLE_COLUMN: sample_role,
            "s3_key": key,
            "local_path": str(
                local_path
            ),
            "status": (
                f"download_error:"
                f"{type(error).__name__}"
            ),
            "remote_size_bytes": (
                expected_size
            ),
            "local_size_bytes": (
                int(
                    local_path.stat().st_size
                )
                if local_path.exists()
                else 0
            ),
            "rows": 0,
            "columns": 0,
        }

    validation = validate_local_parquet(
        local_path
    )

    local_size = (
        int(
            local_path.stat().st_size
        )
        if local_path.exists()
        else 0
    )

    if not validation[
        "valid_parquet"
    ]:

        status = "invalid_parquet"

    elif local_size != expected_size:

        status = "size_mismatch"

    else:

        status = "downloaded"

    return {
        BUILDING_ID_COLUMN: bldg_id,
        ROLE_COLUMN: sample_role,
        "s3_key": key,
        "local_path": str(
            local_path
        ),
        "status": status,
        "remote_size_bytes": (
            expected_size
        ),
        "local_size_bytes": (
            local_size
        ),
        "rows": validation[
            "rows"
        ],
        "columns": validation[
            "columns"
        ],
    }


def print_summary(
    manifest: pd.DataFrame,
    elapsed_seconds: float,
) -> None:

    successful_statuses = {
        "downloaded",
        "already_present",
    }

    usable_mask = manifest[
        "status"
    ].isin(
        successful_statuses
    )

    usable = manifest.loc[
        usable_mask
    ]

    failed = manifest.loc[
        ~usable_mask
    ]

    development_usable = int(
        (
            (
                usable[
                    ROLE_COLUMN
                ]
                == "development"
            )
        )
        .sum()
    )

    held_out_usable = int(
        (
            (
                usable[
                    ROLE_COLUMN
                ]
                == "held_out"
            )
        )
        .sum()
    )

    total_size_bytes = int(
        usable[
            "local_size_bytes"
        ].sum()
    )

    total_size_mb = (
        total_size_bytes
        / (
            1024 ** 2
        )
    )

    print()

    print(
        "ComStock modeling-profile download summary:"
    )

    print(
        f"  Requested buildings: "
        f"{len(manifest):,}"
    )

    print(
        f"  Usable buildings: "
        f"{len(usable):,}"
    )

    print(
        f"  Failed buildings: "
        f"{len(failed):,}"
    )

    print(
        f"  Development usable: "
        f"{development_usable:,}"
    )

    print(
        f"  Held-out usable: "
        f"{held_out_usable:,}"
    )

    print(
        f"  Total local size: "
        f"{total_size_mb:,.2f} MB"
    )

    print(
        f"  Elapsed time: "
        f"{elapsed_seconds:,.2f} seconds"
    )

    print()

    print(
        "Status counts:"
    )

    print(
        manifest[
            "status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    if not failed.empty:

        print()

        print(
            "Failed building IDs:"
        )

        print(
            failed[
                [
                    BUILDING_ID_COLUMN,
                    ROLE_COLUMN,
                    "status",
                ]
            ]
            .to_string(
                index=False
            )
        )


def main() -> None:

    print(
        "Loading ComStock modeling sample..."
    )

    sample = load_sample()

    print(
        f"Requested buildings: "
        f"{len(sample):,}"
    )

    print(
        f"Development buildings: "
        f"{int((sample[ROLE_COLUMN] == 'development').sum()):,}"
    )

    print(
        f"Held-out buildings: "
        f"{int((sample[ROLE_COLUMN] == 'held_out').sum()):,}"
    )

    print()

    print(
        f"Destination directory: "
        f"{OUTPUT_DIRECTORY}"
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()

    print(
        "Creating anonymous S3 client..."
    )

    s3 = create_s3_client()

    print()

    print(
        "Downloading ComStock California "
        "modeling profiles..."
    )

    start = perf_counter()

    results = []

    total = len(
        sample
    )

    for number, row_index in enumerate(
        sample.index,
        start=1,
    ):

        bldg_id_value = sample.at[
            row_index,
            BUILDING_ID_COLUMN,
        ]

        sample_role_value = sample.at[
            row_index,
            ROLE_COLUMN,
        ]

        bldg_id = int(
            str(
                bldg_id_value
            )
        )

        sample_role = str(
            sample_role_value
        )

        print(
            f"[{number:03d}/{total:03d}] "
            f"Building {bldg_id} "
            f"({sample_role})..."
        )

        result = download_building(
            s3,
            bldg_id,
            sample_role,
        )

        results.append(
            result
        )

        print(
            f"  {result['status']}"
        )

    elapsed_seconds = (
        perf_counter()
        - start
    )

    manifest = pd.DataFrame(
        results
    )

    manifest.to_csv(
        MANIFEST_PATH,
        index=False,
    )

    print_summary(
        manifest,
        elapsed_seconds,
    )

    usable_statuses = {
        "downloaded",
        "already_present",
    }

    usable_count = int(
        manifest[
            "status"
        ]
        .isin(
            usable_statuses
        )
        .sum()
    )

    if usable_count != EXPECTED_BUILDINGS:

        raise RuntimeError(
            f"Only {usable_count} of "
            f"{EXPECTED_BUILDINGS} ComStock "
            "modeling profiles are usable. "
            "Review the download manifest."
        )

    print()

    print(
        "All requested ComStock modeling "
        "profiles are available and valid."
    )

    print()

    print(
        f"Saved download manifest to: "
        f"{MANIFEST_PATH}"
    )
if __name__ == "__main__":
    main()