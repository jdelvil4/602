from pathlib import Path
from time import perf_counter

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError


PROJECT_ROOT = Path(__file__).resolve().parents[2]

COVERAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comstock_ca_weather_coverage.csv"
)

WEATHER_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nrel"
    / "weather"
    / "amy2018"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comstock_ca_missing_weather_download_manifest.csv"
)


BUCKET = "oedi-data-lake"

WEATHER_PREFIX = (
    "nrel-pds-building-stock/"
    "end-use-load-profiles-for-us-building-stock/"
    "2021/resstock_amy2018_release_1/"
    "weather/"
    "amy2018/"
)

COUNTY_COLUMN = "in.county"
FOUND_COLUMN = "weather_file_found"

EXPECTED_HOURLY_ROWS = 8760


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


def load_missing_counties() -> list[str]:

    if not COVERAGE_PATH.exists():
        raise FileNotFoundError(
            f"ComStock weather coverage table not found: "
            f"{COVERAGE_PATH}"
        )

    coverage = pd.read_csv(
        COVERAGE_PATH
    )

    required_columns = {
        COUNTY_COLUMN,
        FOUND_COLUMN,
    }

    missing_columns = (
        required_columns
        - set(
            coverage.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Coverage table is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    missing = coverage.loc[
        ~coverage[
            FOUND_COLUMN
        ].astype(
            bool
        ),
        COUNTY_COLUMN,
    ]

    counties = sorted(
        missing
        .astype(
            str
        )
        .tolist()
    )

    if not counties:
        print(
            "No missing ComStock weather counties were found."
        )

    return counties


def build_s3_key(
    county: str,
) -> str:

    return (
        f"{WEATHER_PREFIX}"
        f"{county}_2018.csv"
    )


def build_local_path(
    county: str,
) -> Path:

    return (
        WEATHER_DIRECTORY
        / f"{county}_2018.csv"
    )


def check_remote_file(
    s3,
    key: str,
) -> int:

    try:

        response = s3.head_object(
            Bucket=BUCKET,
            Key=key,
        )

        return int(
            response[
                "ContentLength"
            ]
        )

    except ClientError as error:

        code = str(
            error.response[
                "Error"
            ].get(
                "Code",
                ""
            )
        )

        if code in {
            "404",
            "NoSuchKey",
            "NotFound",
        }:
            return 0

        raise


def validate_weather_file(
    path: Path,
) -> tuple[bool, int, int]:

    if not path.exists():
        return (
            False,
            0,
            0,
        )

    try:

        weather = pd.read_csv(
            path
        )

    except Exception:

        return (
            False,
            0,
            0,
        )

    rows = len(
        weather
    )

    columns = len(
        weather.columns
    )

    valid = (
        rows == EXPECTED_HOURLY_ROWS
        and columns > 0
    )

    return (
        valid,
        rows,
        columns,
    )


def download_weather_file(
    s3,
    county: str,
) -> dict:

    key = build_s3_key(
        county
    )

    local_path = build_local_path(
        county
    )

    remote_size = check_remote_file(
        s3,
        key,
    )

    if remote_size == 0:

        return {
            COUNTY_COLUMN: county,
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

    if local_path.exists():

        valid, rows, columns = (
            validate_weather_file(
                local_path
            )
        )

        local_size = int(
            local_path.stat().st_size
        )

        if (
            valid
            and local_size == remote_size
        ):

            return {
                COUNTY_COLUMN: county,
                "s3_key": key,
                "local_path": str(
                    local_path
                ),
                "status": "already_present",
                "remote_size_bytes": remote_size,
                "local_size_bytes": local_size,
                "rows": rows,
                "columns": columns,
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
            COUNTY_COLUMN: county,
            "s3_key": key,
            "local_path": str(
                local_path
            ),
            "status": (
                f"download_error:"
                f"{type(error).__name__}"
            ),
            "remote_size_bytes": remote_size,
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

    valid, rows, columns = (
        validate_weather_file(
            local_path
        )
    )

    local_size = int(
        local_path.stat().st_size
    )

    if not valid:
        status = "invalid_weather_file"

    elif local_size != remote_size:
        status = "size_mismatch"

    else:
        status = "downloaded"

    return {
        COUNTY_COLUMN: county,
        "s3_key": key,
        "local_path": str(
            local_path
        ),
        "status": status,
        "remote_size_bytes": remote_size,
        "local_size_bytes": local_size,
        "rows": rows,
        "columns": columns,
    }


def main() -> None:

    print(
        "Loading missing ComStock weather counties..."
    )

    counties = load_missing_counties()

    print(
        f"Missing counties: "
        f"{len(counties):,}"
    )

    if not counties:
        return

    for county in counties:

        print(
            f"  {county}"
        )

    print()

    WEATHER_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Creating anonymous S3 client..."
    )

    s3 = create_s3_client()

    print()

    print(
        "Downloading missing AMY2018 county weather files..."
    )

    start = perf_counter()

    results = []

    total = len(
        counties
    )

    for number, county in enumerate(
        counties,
        start=1,
    ):

        print(
            f"[{number:02d}/{total:02d}] "
            f"{county}..."
        )

        result = download_weather_file(
            s3,
            county,
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

    successful_statuses = {
        "downloaded",
        "already_present",
    }

    usable_count = int(
        manifest[
            "status"
        ]
        .isin(
            successful_statuses
        )
        .sum()
    )

    failed = manifest.loc[
        ~manifest[
            "status"
        ].isin(
            successful_statuses
        )
    ]

    total_size_mb = (
        manifest.loc[
            manifest[
                "status"
            ].isin(
                successful_statuses
            ),
            "local_size_bytes",
        ].sum()
        / (
            1024 ** 2
        )
    )

    print()

    print(
        "ComStock missing-weather download summary:"
    )

    print(
        f"  Requested counties: "
        f"{len(manifest):,}"
    )

    print(
        f"  Usable counties: "
        f"{usable_count:,}"
    )

    print(
        f"  Failed counties: "
        f"{len(failed):,}"
    )

    print(
        f"  Downloaded size: "
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
            "Failed counties:"
        )

        print(
            failed[
                [
                    COUNTY_COLUMN,
                    "status",
                ]
            ]
            .to_string(
                index=False
            )
        )

    if usable_count != len(
        counties
    ):

        raise RuntimeError(
            f"Only {usable_count} of "
            f"{len(counties)} missing weather files "
            "are usable. Review the manifest."
        )

    print()

    print(
        "All missing ComStock weather files "
        "are available and valid."
    )

    print()

    print(
        f"Saved download manifest to: "
        f"{MANIFEST_PATH}"
    )


if __name__ == "__main__":
    main()