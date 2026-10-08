from pathlib import Path

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELING_BASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_modeling_base.parquet"
)

OUTPUT_DIR = (
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
    / "nrel_ca_weather_download_manifest.csv"
)

BUCKET = "oedi-data-lake"

WEATHER_PREFIX = (
    "nrel-pds-building-stock/"
    "end-use-load-profiles-for-us-building-stock/"
    "2021/"
    "resstock_amy2018_release_1/"
    "weather/"
    "amy2018/"
)

EXPECTED_COUNTIES = 41


def create_s3_client():

    return boto3.client(
        "s3",
        region_name="us-west-2",
        config=Config(
            signature_version=UNSIGNED
        ),
    )


def load_required_counties() -> list[str]:

    if not MODELING_BASE_PATH.exists():
        raise FileNotFoundError(
            f"Modeling base dataset not found: "
            f"{MODELING_BASE_PATH}"
        )

    df = pd.read_parquet(
        MODELING_BASE_PATH,
        columns=[
            "in.county",
        ],
    )

    counties = (
        df[
            "in.county"
        ]
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if len(counties) != EXPECTED_COUNTIES:
        raise ValueError(
            f"Found {len(counties)} unique counties; "
            f"expected {EXPECTED_COUNTIES}."
        )

    return counties


def build_weather_key(
    county_id: str,
) -> str:

    return (
        WEATHER_PREFIX
        + f"{county_id}_2018.csv"
    )


def download_county_weather(
    s3,
    county_id: str,
) -> dict:

    filename = (
        f"{county_id}_2018.csv"
    )

    key = build_weather_key(
        county_id
    )

    local_path = (
        OUTPUT_DIR
        / filename
    )

    result = {
        "county_id": county_id,
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

        head = s3.head_object(
            Bucket=BUCKET,
            Key=key,
        )

        remote_size = int(
            head[
                "ContentLength"
            ]
        )

        result[
            "remote_size_bytes"
        ] = remote_size

        if local_path.exists():

            local_size = (
                local_path
                .stat()
                .st_size
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
            local_path
            .stat()
            .st_size
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
                f"Expected {remote_size} bytes but "
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


def validate_downloaded_file(
    path: Path,
    county_id: str,
) -> dict:

    result = {
        "county_id": county_id,
        "row_count": None,
        "column_count": None,
        "minimum_timestamp": None,
        "maximum_timestamp": None,
        "missing_values": None,
        "duplicate_timestamps": None,
        "validation_status": None,
        "validation_error": None,
    }

    try:

        df = pd.read_csv(
            path
        )

        required_columns = {
            "date_time",
            "Dry Bulb Temperature [°C]",
            "Relative Humidity [%]",
            "Wind Speed [m/s]",
            "Wind Direction [Deg]",
            "Global Horizontal Radiation [W/m2]",
            "Direct Normal Radiation [W/m2]",
            "Diffuse Horizontal Radiation [W/m2]",
        }

        missing_columns = (
            required_columns
            - set(
                df.columns
            )
        )

        if missing_columns:
            raise ValueError(
                "Missing required weather columns: "
                f"{sorted(missing_columns)}"
            )

        df[
            "date_time"
        ] = pd.to_datetime(
            df[
                "date_time"
            ]
        )

        result[
            "row_count"
        ] = len(
            df
        )

        result[
            "column_count"
        ] = len(
            df.columns
        )

        result[
            "minimum_timestamp"
        ] = df[
            "date_time"
        ].min()

        result[
            "maximum_timestamp"
        ] = df[
            "date_time"
        ].max()

        result[
            "missing_values"
        ] = int(
            df
            .isna()
            .sum()
            .sum()
        )

        result[
            "duplicate_timestamps"
        ] = int(
            df[
                "date_time"
            ]
            .duplicated()
            .sum()
        )

        if len(df) != 8760:
            raise ValueError(
                f"Expected 8,760 rows but found "
                f"{len(df):,}."
            )

        if (
            result[
                "missing_values"
            ]
            != 0
        ):
            raise ValueError(
                f"Found "
                f"{result['missing_values']} "
                "missing values."
            )

        if (
            result[
                "duplicate_timestamps"
            ]
            != 0
        ):
            raise ValueError(
                f"Found "
                f"{result['duplicate_timestamps']} "
                "duplicate timestamps."
            )

        expected_minimum = pd.Timestamp(
            "2018-01-01 01:00:00"
        )

        expected_maximum = pd.Timestamp(
            "2019-01-01 00:00:00"
        )

        if (
            result[
                "minimum_timestamp"
            ]
            != expected_minimum
        ):
            raise ValueError(
                "Unexpected minimum timestamp: "
                f"{result['minimum_timestamp']}"
            )

        if (
            result[
                "maximum_timestamp"
            ]
            != expected_maximum
        ):
            raise ValueError(
                "Unexpected maximum timestamp: "
                f"{result['maximum_timestamp']}"
            )

        timestamp_differences = (
            df[
                "date_time"
            ]
            .sort_values()
            .diff()
            .dropna()
        )

        invalid_hourly_intervals = int(
            (
                timestamp_differences
                != pd.Timedelta(
                    hours=1
                )
            )
            .sum()
        )

        if invalid_hourly_intervals != 0:
            raise ValueError(
                f"Found {invalid_hourly_intervals} "
                "non-hourly timestamp intervals."
            )

        result[
            "validation_status"
        ] = "valid"

        return result

    except Exception as exc:

        result[
            "validation_status"
        ] = "invalid"

        result[
            "validation_error"
        ] = str(
            exc
        )

        return result


def main() -> None:

    counties = load_required_counties()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Required weather counties: "
        f"{len(counties):,}"
    )

    print()

    s3 = create_s3_client()

    results = []

    for position, county_id in enumerate(
        counties,
        start=1,
    ):

        print(
            f"[{position}/{len(counties)}] "
            f"Downloading weather for "
            f"{county_id}"
        )

        download_result = (
            download_county_weather(
                s3,
                county_id,
            )
        )

        print(
            f"    download status: "
            f"{download_result['status']}"
        )

        validation_result = {
            "county_id": county_id,
            "row_count": None,
            "column_count": None,
            "minimum_timestamp": None,
            "maximum_timestamp": None,
            "missing_values": None,
            "duplicate_timestamps": None,
            "validation_status": None,
            "validation_error": None,
        }

        if download_result[
            "status"
        ] in {
            "downloaded",
            "already_exists",
        }:

            local_path = (
                OUTPUT_DIR
                / download_result[
                    "filename"
                ]
            )

            validation_result = (
                validate_downloaded_file(
                    local_path,
                    county_id,
                )
            )

            print(
                f"    validation status: "
                f"{validation_result['validation_status']}"
            )

        result = {
            **download_result,
            **{
                key: value
                for key, value
                in validation_result.items()
                if key != "county_id"
            },
        }

        results.append(
            result
        )

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

    valid_count = int(
        (
            manifest[
                "validation_status"
            ]
            == "valid"
        )
        .sum()
    )

    invalid_count = int(
        (
            manifest[
                "validation_status"
            ]
            == "invalid"
        )
        .sum()
    )

    total_size_bytes = int(
        manifest[
            "local_size_bytes"
        ]
        .fillna(0)
        .sum()
    )

    print()

    print(
        "NREL county weather acquisition complete."
    )

    print()

    print(
        f"Requested counties: "
        f"{len(manifest):,}"
    )

    print(
        f"Valid weather files: "
        f"{valid_count:,}"
    )

    print(
        f"Invalid weather files: "
        f"{invalid_count:,}"
    )

    print(
        f"Total local size: "
        f"{total_size_bytes / 1_048_576:.2f} MB"
    )

    print()

    print(
        "Download status counts:"
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
        "Validation status counts:"
    )

    print(
        manifest[
            "validation_status"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    invalid_rows = manifest.loc[
        manifest[
            "validation_status"
        ]
        != "valid"
    ]

    if not invalid_rows.empty:

        print()

        print(
            "Invalid weather files:"
        )

        print(
            invalid_rows[
                [
                    "county_id",
                    "status",
                    "error",
                    "validation_status",
                    "validation_error",
                ]
            ]
            .to_string(
                index=False
            )
        )

    print()

    print(
        f"Weather directory: "
        f"{OUTPUT_DIR}"
    )

    print(
        f"Manifest: "
        f"{MANIFEST_PATH}"
    )


if __name__ == "__main__":
    main()