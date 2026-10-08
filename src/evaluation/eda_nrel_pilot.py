from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_pilot_hourly.parquet"
)

OUTPUT_TABLE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
)

OUTPUT_FIGURE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
)

BUILDING_STATS_PATH = (
    OUTPUT_TABLE_DIR
    / "nrel_pilot_building_statistics.csv"
)

HOURLY_PROFILE_PATH = (
    OUTPUT_TABLE_DIR
    / "nrel_pilot_average_hourly_profile.csv"
)

EDA_SUMMARY_PATH = (
    OUTPUT_TABLE_DIR
    / "nrel_pilot_eda_summary.csv"
)

DISTRIBUTION_FIGURE_PATH = (
    OUTPUT_FIGURE_DIR
    / "nrel_pilot_hourly_consumption_distribution.png"
)

HOURLY_PROFILE_FIGURE_PATH = (
    OUTPUT_FIGURE_DIR
    / "nrel_pilot_average_hourly_profile.png"
)

BUILDING_MEAN_FIGURE_PATH = (
    OUTPUT_FIGURE_DIR
    / "nrel_pilot_building_mean_consumption.png"
)


TIMESTAMP_COLUMN = "timestamp"
BUILDING_COLUMN = "bldg_id"
TARGET_COLUMN = "electricity_total_hourly"


def load_data() -> pd.DataFrame:

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"NREL pilot hourly dataset not found: {INPUT_PATH}"
        )

    df = pd.read_parquet(
        INPUT_PATH
    )

    required_columns = {
        TIMESTAMP_COLUMN,
        BUILDING_COLUMN,
        TARGET_COLUMN,
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Required columns are missing: "
            f"{sorted(missing_columns)}"
        )

    df.loc[:, TIMESTAMP_COLUMN] = pd.to_datetime(
        df.loc[:, TIMESTAMP_COLUMN]
    )

    return df


def create_building_statistics(
    df: pd.DataFrame,
) -> pd.DataFrame:

    building_stats = (
        df.groupby(
            BUILDING_COLUMN
        )[TARGET_COLUMN]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std",
                "min",
                "max",
                "sum",
            ]
        )
        .reset_index()
    )

    building_stats = building_stats.rename(
        columns={
            "count": "hourly_rows",
            "mean": "mean_hourly_electricity",
            "median": "median_hourly_electricity",
            "std": "std_hourly_electricity",
            "min": "minimum_hourly_electricity",
            "max": "maximum_hourly_electricity",
            "sum": "annual_electricity",
        }
    )

    return building_stats


def create_hourly_profile(
    df: pd.DataFrame,
) -> pd.DataFrame:

    working = df.copy()

    working.loc[:, "hour"] = (
        working.loc[
            :,
            TIMESTAMP_COLUMN,
        ]
        .dt.hour
    )

    hourly_profile = (
        working.groupby(
            "hour"
        )[TARGET_COLUMN]
        .agg(
            [
                "mean",
                "median",
                "std",
            ]
        )
        .reset_index()
    )

    hourly_profile = hourly_profile.rename(
        columns={
            "mean": "mean_electricity",
            "median": "median_electricity",
            "std": "std_electricity",
        }
    )

    return hourly_profile


def create_eda_summary(
    df: pd.DataFrame,
    building_stats: pd.DataFrame,
) -> pd.DataFrame:

    negative_count = int(
        (
            df.loc[
                :,
                TARGET_COLUMN,
            ]
            < 0
        )
        .sum()
    )

    zero_count = int(
        (
            df.loc[
                :,
                TARGET_COLUMN,
            ]
            == 0
        )
        .sum()
    )

    duplicate_count = int(
        df.duplicated(
            subset=[
                BUILDING_COLUMN,
                TIMESTAMP_COLUMN,
            ]
        )
        .sum()
    )

    file_size_bytes = (
        INPUT_PATH.stat().st_size
    )

    summary = pd.DataFrame(
        [
            {
                "buildings": df.loc[
                    :,
                    BUILDING_COLUMN,
                ].nunique(),
                "rows": len(df),
                "minimum_timestamp": df.loc[
                    :,
                    TIMESTAMP_COLUMN,
                ].min(),
                "maximum_timestamp": df.loc[
                    :,
                    TIMESTAMP_COLUMN,
                ].max(),
                "missing_values": int(
                    df.isna().sum().sum()
                ),
                "duplicate_building_timestamps": (
                    duplicate_count
                ),
                "negative_electricity_rows": (
                    negative_count
                ),
                "zero_electricity_rows": (
                    zero_count
                ),
                "mean_hourly_electricity": df.loc[
                    :,
                    TARGET_COLUMN,
                ].mean(),
                "median_hourly_electricity": df.loc[
                    :,
                    TARGET_COLUMN,
                ].median(),
                "minimum_hourly_electricity": df.loc[
                    :,
                    TARGET_COLUMN,
                ].min(),
                "maximum_hourly_electricity": df.loc[
                    :,
                    TARGET_COLUMN,
                ].max(),
                "minimum_building_annual_electricity": (
                    building_stats.loc[
                        :,
                        "annual_electricity",
                    ]
                    .min()
                ),
                "maximum_building_annual_electricity": (
                    building_stats.loc[
                        :,
                        "annual_electricity",
                    ]
                    .max()
                ),
                "pilot_parquet_size_mb": (
                    file_size_bytes
                    / 1_048_576
                ),
            }
        ]
    )

    return summary


def plot_distribution(
    df: pd.DataFrame,
) -> None:

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        df.loc[
            :,
            TARGET_COLUMN,
        ],
        bins=80,
    )

    plt.xlabel(
        "Hourly Electricity Consumption"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "NREL Pilot Hourly Electricity Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        DISTRIBUTION_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_hourly_profile(
    hourly_profile: pd.DataFrame,
) -> None:

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        hourly_profile.loc[
            :,
            "hour",
        ],
        hourly_profile.loc[
            :,
            "mean_electricity",
        ],
        marker="o",
    )

    plt.xlabel(
        "Hour of Day"
    )

    plt.ylabel(
        "Mean Hourly Electricity Consumption"
    )

    plt.title(
        "NREL Pilot Average Hourly Load Profile"
    )

    plt.xticks(
        range(0, 24)
    )

    plt.tight_layout()

    plt.savefig(
        HOURLY_PROFILE_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_building_means(
    building_stats: pd.DataFrame,
) -> None:

    plot_df = (
        building_stats
        .sort_values(
            "mean_hourly_electricity",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    plt.figure(
        figsize=(11, 7)
    )

    plt.barh(
        plot_df.loc[
            :,
            BUILDING_COLUMN,
        ].astype(str),
        plot_df.loc[
            :,
            "mean_hourly_electricity",
        ],
    )

    plt.xlabel(
        "Mean Hourly Electricity Consumption"
    )

    plt.ylabel(
        "Building ID"
    )

    plt.title(
        "NREL Pilot Mean Electricity Consumption by Building"
    )

    plt.tight_layout()

    plt.savefig(
        BUILDING_MEAN_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def main() -> None:

    OUTPUT_TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data()

    building_stats = create_building_statistics(
        df
    )

    hourly_profile = create_hourly_profile(
        df
    )

    summary = create_eda_summary(
        df,
        building_stats,
    )

    building_stats.to_csv(
        BUILDING_STATS_PATH,
        index=False,
    )

    hourly_profile.to_csv(
        HOURLY_PROFILE_PATH,
        index=False,
    )

    summary.to_csv(
        EDA_SUMMARY_PATH,
        index=False,
    )

    plot_distribution(
        df
    )

    plot_hourly_profile(
        hourly_profile
    )

    plot_building_means(
        building_stats
    )

    print(
        "NREL pilot EDA complete."
    )

    print()

    print(
        summary.to_string(
            index=False
        )
    )

    print()

    print(
        "Building statistics:"
    )

    print(
        building_stats.to_string(
            index=False
        )
    )

    print()

    print(
        f"Saved tables to: "
        f"{OUTPUT_TABLE_DIR}"
    )

    print(
        f"Saved figures to: "
        f"{OUTPUT_FIGURE_DIR}"
    )


if __name__ == "__main__":
    main()