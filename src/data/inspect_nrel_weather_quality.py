from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

WEATHER_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nrel"
    / "weather"
    / "amy2018"
)

WEATHER_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_weather_download_manifest.csv"
)

WIND_DIRECTION_COLUMN = "Wind Direction [Deg]"
WIND_SPEED_COLUMN = "Wind Speed [m/s]"
TIMESTAMP_COLUMN = "date_time"

VALID_MIN_DIRECTION = 0.0
VALID_MAX_DIRECTION = 360.0


def load_manifest() -> pd.DataFrame:

    if not WEATHER_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Weather manifest not found: "
            f"{WEATHER_MANIFEST_PATH}"
        )

    manifest = pd.read_csv(
        WEATHER_MANIFEST_PATH
    )

    required_columns = {
        "county_id",
        "filename",
        "validation_status",
    }

    missing_columns = (
        required_columns
        - set(manifest.columns)
    )

    if missing_columns:
        raise ValueError(
            "Weather manifest is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    manifest = (
        manifest.loc[
            manifest[
                "validation_status"
            ]
            == "valid"
        ]
        .copy()
        .reset_index(drop=True)
    )

    return manifest


def inspect_county(
    county_id: str,
    filename: str,
) -> tuple[dict, pd.DataFrame]:

    path = (
        WEATHER_DIR
        / filename
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Weather file not found: {path}"
        )

    df = pd.read_csv(
        path,
        usecols=[
            TIMESTAMP_COLUMN,
            WIND_DIRECTION_COLUMN,
            WIND_SPEED_COLUMN,
        ],
    )

    df[
        TIMESTAMP_COLUMN
    ] = pd.to_datetime(
        df[
            TIMESTAMP_COLUMN
        ]
    )

    out_of_range = df.loc[
        (
            df[
                WIND_DIRECTION_COLUMN
            ]
            < VALID_MIN_DIRECTION
        )
        |
        (
            df[
                WIND_DIRECTION_COLUMN
            ]
            > VALID_MAX_DIRECTION
        )
    ].copy()

    summary = {
        "county_id": county_id,
        "rows": len(df),
        "minimum_wind_direction": df[
            WIND_DIRECTION_COLUMN
        ].min(),
        "maximum_wind_direction": df[
            WIND_DIRECTION_COLUMN
        ].max(),
        "out_of_range_count": len(
            out_of_range
        ),
        "out_of_range_percentage": (
            len(out_of_range)
            / len(df)
            * 100
        ),
    }

    if not out_of_range.empty:

        out_of_range.insert(
            0,
            "county_id",
            county_id,
        )

    return (
        summary,
        out_of_range,
    )


def main() -> None:

    manifest = load_manifest()

    print(
        f"Weather files to inspect: "
        f"{len(manifest):,}"
    )

    print()

    summaries = []
    anomaly_frames = []

    for position, row in enumerate(
        manifest.itertuples(
            index=False
        ),
        start=1,
    ):

        county_id = str(
            row.county_id
        )

        filename = str(
            row.filename
        )

        print(
            f"[{position}/{len(manifest)}] "
            f"Inspecting {county_id}"
        )

        summary, anomalies = inspect_county(
            county_id,
            filename,
        )

        summaries.append(
            summary
        )

        if not anomalies.empty:

            anomaly_frames.append(
                anomalies
            )

    summary_df = pd.DataFrame(
        summaries
    )

    if anomaly_frames:

        anomalies_df = pd.concat(
            anomaly_frames,
            ignore_index=True,
        )

    else:

        anomalies_df = pd.DataFrame(
            columns=[
                "county_id",
                TIMESTAMP_COLUMN,
                WIND_DIRECTION_COLUMN,
                WIND_SPEED_COLUMN,
            ]
        )

    print()

    print(
        "Wind-direction quality summary:"
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()

    total_rows = int(
        summary_df[
            "rows"
        ].sum()
    )

    total_anomalies = int(
        summary_df[
            "out_of_range_count"
        ].sum()
    )

    print(
        f"Total weather rows: "
        f"{total_rows:,}"
    )

    print(
        f"Out-of-range wind-direction rows: "
        f"{total_anomalies:,}"
    )

    print(
        f"Out-of-range percentage: "
        f"{total_anomalies / total_rows * 100:.6f}%"
    )

    if not anomalies_df.empty:

        print()

        print(
            "Out-of-range wind-direction value counts:"
        )

        print(
            anomalies_df[
                WIND_DIRECTION_COLUMN
            ]
            .value_counts()
            .sort_index()
            .to_string()
        )

        print()

        print(
            "Counties containing out-of-range values:"
        )

        print(
            anomalies_df[
                "county_id"
            ]
            .value_counts()
            .sort_index()
            .to_string()
        )

        print()

        print(
            "Wind speeds associated with out-of-range directions:"
        )

        print(
            anomalies_df[
                WIND_SPEED_COLUMN
            ]
            .describe()
            .to_string()
        )

        print()

        print(
            "First 30 anomalous observations:"
        )

        print(
            anomalies_df.head(
                30
            )
            .to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()