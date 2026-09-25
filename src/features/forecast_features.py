from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "uci_hourly.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "uci_forecast_features.csv"
)

TARGET_COLUMN = "Appliances"

HEATING_THRESHOLD_C = 18.0
COOLING_THRESHOLD_C = 22.0

def load_hourly_data(path: Path) -> pd.DataFrame:


    if not path.exists():
        raise FileNotFoundError(
            f"Hourly UCI dataset not found: {path}"
        )

    df = pd.read_csv(
        path,
        parse_dates=["date"],
    )

    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )

    print(f"Loaded {len(df):,} hourly rows.")

    return df


def add_calendar_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    df["dow_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["dow_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["month_sin"] = np.sin(
        2 * np.pi * (df["month"] - 1) / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * (df["month"] - 1) / 12
    )

    return df

def add_temperature_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    if "T_out" not in df.columns:
        raise ValueError(
            "Expected outdoor temperature column 'T_out'."
        )

    df["heating_indicator"] = (
        df["T_out"] < HEATING_THRESHOLD_C
    ).astype(int)

    df["cooling_indicator"] = (
        df["T_out"] > COOLING_THRESHOLD_C
    ).astype(int)

    return df

def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:

    df = df.copy()

    df["consumption_lag_1"] = (
        df[target_col].shift(1)
    )

    df["consumption_lag_24"] = (
        df[target_col].shift(24)
    )

    df["seasonal_naive_24h"] = (
        df[target_col].shift(23)
    )

    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:

    df = df.copy()

    previous_values = (
        df[target_col].shift(1)
    )

    df["consumption_mean_24h"] = (
        previous_values
        .rolling(
            window=24,
            min_periods=24,
        )
        .mean()
    )

    df["consumption_mean_7d"] = (
        previous_values
        .rolling(
            window=168,
            min_periods=168,
        )
        .mean()
    )

    return df

def add_forecast_target(
    df: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:

    df = df.copy()

    df["target_next_hour"] = (
        df[target_col].shift(-1)
    )

    return df

def remove_incomplete_feature_rows(
    df: pd.DataFrame,
) -> pd.DataFrame:


    required = [
        "consumption_lag_1",
        "consumption_lag_24",
        "seasonal_naive_24h",
        "consumption_mean_24h",
        "consumption_mean_7d",
        "target_next_hour",
    ]

    before = len(df)

    df = (
        df
        .dropna(subset=required)
        .reset_index(drop=True)
    )

    removed = before - len(df)

    print(
        "Rows removed because of "
        "lag/rolling/target requirements: "
        f"{removed:,}"
    )

    return df

def validate_features(
    df: pd.DataFrame,
) -> None:

    required_columns = [
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "heating_indicator",
        "cooling_indicator",
        "consumption_lag_1",
        "consumption_lag_24",
        "seasonal_naive_24h",
        "consumption_mean_24h",
        "consumption_mean_7d",
        "target_next_hour",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing engineered features: "
            f"{missing_columns}"
        )

    duplicate_timestamps = (
        df["date"]
        .duplicated()
        .sum()
    )

    print(
        f"Final feature rows: {len(df):,}"
    )

    print(
        f"Final feature columns: {len(df.columns):,}"
    )

    print(
        "Duplicate timestamps: "
        f"{duplicate_timestamps:,}"
    )

    if not df["date"].is_monotonic_increasing:
        raise ValueError(
            "Forecast feature dataset is not "
            "chronologically ordered."
        )

    if duplicate_timestamps != 0:
        raise ValueError(
            "Duplicate timestamps exist in the "
            "forecast feature dataset."
        )

    if (
        df[required_columns]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            "Missing values remain in required "
            "forecasting features."
        )


def save_features(
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

    print()
    print(
        f"Saved forecasting features to: {path}"
    )


def main() -> None:
    df = load_hourly_data(
        INPUT_PATH
    )

    df = add_calendar_features(
        df
    )

    df = add_temperature_features(
        df
    )

    df = add_lag_features(
        df,
        TARGET_COLUMN,
    )

    df = add_rolling_features(
        df,
        TARGET_COLUMN,
    )

    df = add_forecast_target(
        df,
        TARGET_COLUMN,
    )

    df = remove_incomplete_feature_rows(
        df
    )

    validate_features(
        df
    )

    save_features(
        df,
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()