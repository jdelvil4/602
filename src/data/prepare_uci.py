from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "appliances+energy+prediction"
    / "energydata_complete.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "uci_hourly.csv"
)


ENERGY_COLUMNS = [
    "Appliances",
    "lights",
]

DROP_COLUMNS = [
    "rv1",
    "rv2",
]


def load_uci_data(path: Path) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"UCI source file not found: {path}"
        )

    df = pd.read_csv(path)

    print(f"Loaded {len(df):,} raw rows.")
    print(f"Raw columns: {len(df.columns)}")

    return df


def validate_raw_data(df: pd.DataFrame) -> None:
    required_columns = {
        "date",
        "Appliances",
        "lights",
        "rv1",
        "rv2",
    }

    missing_required = required_columns - set(df.columns)

    if missing_required:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_required))
        )

    duplicate_rows = df.duplicated().sum()
    missing_values = df.isna().sum().sum()

    print(f"Duplicate rows: {duplicate_rows:,}")
    print(f"Missing values: {missing_values:,}")


def clean_uci_data(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="raise",
    )

    df = df.drop(columns=DROP_COLUMNS)

    df = df.sort_values("date").reset_index(drop=True)

    return df


def resample_to_hourly(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df = df.set_index("date")

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    average_columns = [
        col
        for col in numeric_columns
        if col not in ENERGY_COLUMNS
    ]

    hourly_energy = (
        df[ENERGY_COLUMNS]
        .resample("1h")
        .sum()
    )

    hourly_average = (
        df[average_columns]
        .resample("1h")
        .mean()
    )

    observations_per_hour = (
        df["Appliances"]
        .resample("1h")
        .count()
        .rename("observations_in_hour")
    )

    hourly = pd.concat(
        [
            hourly_energy,
            hourly_average,
            observations_per_hour,
        ],
        axis=1,
    )

    incomplete_hours = (
        hourly["observations_in_hour"] != 6
    ).sum()

    print(
        "Incomplete hours identified: "
        f"{incomplete_hours:,}"
    )

    hourly = hourly[
        hourly["observations_in_hour"] == 6
    ].copy()

    hourly = hourly.drop(
        columns="observations_in_hour"
    )

    hourly = hourly.reset_index()

    return hourly


def validate_hourly_data(df: pd.DataFrame) -> None:

    print(f"Hourly rows: {len(df):,}")
    print(f"Hourly columns: {len(df.columns)}")

    duplicate_timestamps = df["date"].duplicated().sum()
    missing_values = df.isna().sum().sum()

    print(
        f"Duplicate hourly timestamps: "
        f"{duplicate_timestamps:,}"
    )

    print(
        f"Missing values after resampling: "
        f"{missing_values:,}"
    )

    if not df["date"].is_monotonic_increasing:
        raise ValueError(
            "Hourly timestamps are not chronological."
        )


def save_hourly_data(
    df: pd.DataFrame,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        path,
        index=False,
    )

    print(f"Saved processed data to: {path}")


def main() -> None:
    df = load_uci_data(INPUT_PATH)

    validate_raw_data(df)

    df = clean_uci_data(df)

    hourly = resample_to_hourly(df)

    validate_hourly_data(hourly)

    save_hourly_data(
        hourly,
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()