from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "uci_hourly.csv"
)

TABLE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
)

SUMMARY_PATH = (
    TABLE_DIR
    / "uci_eda_summary.csv"
)

DESCRIPTIVE_STATS_PATH = (
    TABLE_DIR
    / "uci_eda_descriptive_statistics.csv"
)

HOURLY_PROFILE_PATH = (
    TABLE_DIR
    / "uci_eda_hourly_profile.csv"
)

CORRELATION_PATH = (
    TABLE_DIR
    / "uci_eda_correlations.csv"
)

HISTOGRAM_PATH = (
    FIGURE_DIR
    / "uci_eda_appliances_histogram.png"
)

HOURLY_PROFILE_FIGURE_PATH = (
    FIGURE_DIR
    / "uci_eda_hourly_consumption_profile.png"
)

CORRELATION_FIGURE_PATH = (
    FIGURE_DIR
    / "uci_eda_correlation_heatmap.png"
)


SUMMARY_COLUMNS = [
    "Appliances",
    "lights",
    "T_out",
    "RH_out",
    "Windspeed",
    "Press_mm_hg",
    "Tdewpoint",
]


CORRELATION_COLUMNS = [
    "Appliances",
    "lights",
    "T_out",
    "RH_out",
    "Windspeed",
    "Press_mm_hg",
    "Tdewpoint",
]


def load_data(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"UCI hourly file not found: {path}"
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

    return df


def validate_columns(
    df: pd.DataFrame,
) -> None:

    required_columns = [
        "date",
        *SUMMARY_COLUMNS,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required EDA columns: "
            f"{missing_columns}"
        )


def create_summary_table(
    df: pd.DataFrame,
) -> pd.DataFrame:

    summary = pd.DataFrame(
        [
            {
                "row_count": len(df),
                "column_count": len(df.columns),
                "start_date": df.loc[:, "date"].min(),
                "end_date": df.loc[:, "date"].max(),
                "duplicate_timestamps": int(
                    df.loc[:, "date"]
                    .duplicated()
                    .sum()
                ),
                "total_missing_values": int(
                    df.isna()
                    .sum()
                    .sum()
                ),
                "missing_rows": int(
                    df.isna()
                    .any(axis=1)
                    .sum()
                ),
            }
        ]
    )

    return summary


def create_descriptive_statistics(
    df: pd.DataFrame,
) -> pd.DataFrame:

    selected = df.filter(
        items=SUMMARY_COLUMNS
    )

    descriptive = (
        selected
        .describe()
        .transpose()
        .reset_index()
        .rename(
            columns={
                "index": "variable"
            }
        )
    )

    descriptive.loc[
        :,
        "missing_count",
    ] = (
        selected
        .isna()
        .sum()
        .to_numpy()
    )

    descriptive.loc[
        :,
        "missing_percent",
    ] = (
        descriptive.loc[
            :,
            "missing_count",
        ]
        / len(df)
        * 100
    )

    return descriptive


def create_hourly_profile(
    df: pd.DataFrame,
) -> pd.DataFrame:

    profile_df = df.copy()

    profile_df.loc[
        :,
        "hour",
    ] = (
        profile_df.loc[
            :,
            "date",
        ]
        .dt.hour
    )

    hourly_profile = (
        profile_df
        .groupby(
            "hour",
            as_index=False,
        )
        .agg(
            mean_appliances=(
                "Appliances",
                "mean",
            ),
            median_appliances=(
                "Appliances",
                "median",
            ),
            mean_lights=(
                "lights",
                "mean",
            ),
            mean_outdoor_temperature=(
                "T_out",
                "mean",
            ),
        )
    )

    return hourly_profile


def create_correlation_table(
    df: pd.DataFrame,
) -> pd.DataFrame:

    selected = df.filter(
        items=CORRELATION_COLUMNS
    )

    correlations = selected.corr(
        numeric_only=True
    )

    return correlations


def save_histogram(
    df: pd.DataFrame,
) -> None:

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        df.loc[
            :,
            "Appliances",
        ],
        bins=30,
    )

    plt.xlabel(
        "Hourly Appliance Energy Consumption"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "Distribution of Hourly Appliance Energy Consumption"
    )

    plt.tight_layout()

    plt.savefig(
        HISTOGRAM_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_hourly_profile_plot(
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
            "mean_appliances",
        ],
        marker="o",
    )

    plt.xlabel(
        "Hour of Day"
    )

    plt.ylabel(
        "Mean Appliance Energy Consumption"
    )

    plt.title(
        "Average Hourly Appliance Energy Consumption"
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


def save_correlation_heatmap(
    correlation_df: pd.DataFrame,
) -> None:

    plt.figure(
        figsize=(9, 7)
    )

    image = plt.imshow(
        correlation_df.to_numpy(),
        aspect="auto",
        vmin=-1,
        vmax=1,
    )

    plt.colorbar(
        image,
        label="Correlation",
    )

    labels = correlation_df.columns.tolist()

    plt.xticks(
        range(len(labels)),
        labels,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(len(labels)),
        labels,
    )

    plt.title(
        "Selected UCI Variable Correlations"
    )

    plt.tight_layout()

    plt.savefig(
        CORRELATION_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def print_summary(
    df: pd.DataFrame,
    descriptive_df: pd.DataFrame,
    hourly_profile: pd.DataFrame,
    correlation_df: pd.DataFrame,
) -> None:

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns):,}"
    )

    print(
        f"Start date: "
        f"{df.loc[:, 'date'].min()}"
    )

    print(
        f"End date: "
        f"{df.loc[:, 'date'].max()}"
    )

    print(
        f"Duplicate timestamps: "
        f"{df.loc[:, 'date'].duplicated().sum():,}"
    )

    print(
        f"Total missing values: "
        f"{df.isna().sum().sum():,}"
    )

    print()

    print(
        "Descriptive statistics"
    )

    print(
        "======================"
    )

    print(
        descriptive_df.to_string(
            index=False
        )
    )

    print()

    print(
        "Hourly profile"
    )

    print(
        "=============="
    )

    print(
        hourly_profile.to_string(
            index=False
        )
    )

    print()

    print(
        "Correlations with Appliances"
    )

    print(
        "============================"
    )

    appliance_correlations = (
        correlation_df
        .loc[
            :,
            "Appliances",
        ]
        .sort_values(
            ascending=False
        )
    )

    print(
        appliance_correlations.to_string()
    )


def main() -> None:

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data(
        INPUT_PATH
    )

    validate_columns(
        df
    )

    summary_df = (
        create_summary_table(
            df
        )
    )

    descriptive_df = (
        create_descriptive_statistics(
            df
        )
    )

    hourly_profile_df = (
        create_hourly_profile(
            df
        )
    )

    correlation_df = (
        create_correlation_table(
            df
        )
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    descriptive_df.to_csv(
        DESCRIPTIVE_STATS_PATH,
        index=False,
    )

    hourly_profile_df.to_csv(
        HOURLY_PROFILE_PATH,
        index=False,
    )

    correlation_df.to_csv(
        CORRELATION_PATH,
    )

    save_histogram(
        df
    )

    save_hourly_profile_plot(
        hourly_profile_df
    )

    save_correlation_heatmap(
        correlation_df
    )

    print_summary(
        df,
        descriptive_df,
        hourly_profile_df,
        correlation_df,
    )

    print()

    print(
        "Saved tables:"
    )

    print(
        SUMMARY_PATH
    )

    print(
        DESCRIPTIVE_STATS_PATH
    )

    print(
        HOURLY_PROFILE_PATH
    )

    print(
        CORRELATION_PATH
    )

    print()

    print(
        "Saved figures:"
    )

    print(
        HISTOGRAM_PATH
    )

    print(
        HOURLY_PROFILE_FIGURE_PATH
    )

    print(
        CORRELATION_FIGURE_PATH
    )


if __name__ == "__main__":
    main()