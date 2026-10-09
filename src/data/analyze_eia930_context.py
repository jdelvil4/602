from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_PATH = (
    PROJECT_ROOT
    / "Region_CAL.xlsx"
)

SHEET_NAME = "Published Hourly Data"

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

SUMMARY_PATH = (
    OUTPUT_TABLE_DIR
    / "eia930_california_summary.csv"
)

ANNUAL_SUMMARY_PATH = (
    OUTPUT_TABLE_DIR
    / "eia930_california_annual_summary.csv"
)

MONTHLY_PROFILE_PATH = (
    OUTPUT_TABLE_DIR
    / "eia930_california_monthly_profile.csv"
)

HOURLY_PROFILE_PATH = (
    OUTPUT_TABLE_DIR
    / "eia930_california_hourly_profile.csv"
)

PEAK_HOURS_PATH = (
    OUTPUT_TABLE_DIR
    / "eia930_california_peak_demand_hours.csv"
)

GENERATION_MIX_PATH = (
    OUTPUT_TABLE_DIR
    / "eia930_california_generation_mix.csv"
)

FORECAST_ERROR_PATH = (
    OUTPUT_TABLE_DIR
    / "eia930_california_forecast_error_summary.csv"
)

HOURLY_DEMAND_FIGURE_PATH = (
    OUTPUT_FIGURE_DIR
    / "eia930_average_demand_by_hour.png"
)

MONTHLY_DEMAND_FIGURE_PATH = (
    OUTPUT_FIGURE_DIR
    / "eia930_average_demand_by_month.png"
)

GENERATION_MIX_FIGURE_PATH = (
    OUTPUT_FIGURE_DIR
    / "eia930_generation_mix.png"
)


REQUIRED_COLUMNS = [
    "Region",
    "UTC time",
    "Local date",
    "Hour",
    "Local time",
    "Time zone",
    "Demand forecast",
    "Demand",
    "Net generation",
    "Total interchange",
]

GENERATION_COLUMNS = {
    "Coal": "NG: COL",
    "Natural gas": "NG: NG",
    "Nuclear": "NG: NUC",
    "Petroleum": "NG: OIL",
    "Geothermal": "NG: GEO",
    "Hydro": "NG: WAT",
    "Pumped storage": "NG: PS",
    "Solar": "NG: SUN",
    "Solar with storage": "NG: SNB",
    "Wind": "NG: WND",
    "Wind with storage": "NG: WNB",
    "Battery storage": "NG: BAT",
    "Other storage": "NG: OES",
    "Unknown storage": "NG: UES",
    "Other": "NG: OTH",
    "Unknown": "NG: UNK",
}


def load_data() -> pd.DataFrame:

    print(
        "Loading California EIA-930 hourly data..."
    )

    columns_to_load = [
        *REQUIRED_COLUMNS,
        *GENERATION_COLUMNS.values(),
    ]

    data = pd.read_excel(
        SOURCE_PATH,
        sheet_name=SHEET_NAME,
        usecols=columns_to_load,
        engine="openpyxl",
    )

    print(
        f"Raw hourly rows: "
        f"{len(data):,}"
    )

    if not data[
        "Region"
    ].eq(
        "CAL"
    ).all():

        unexpected_regions = (
            data.loc[
                data[
                    "Region"
                ]
                != "CAL",
                "Region",
            ]
            .dropna()
            .unique()
            .tolist()
        )

        raise ValueError(
            "Unexpected non-CAL regions found: "
            f"{unexpected_regions}"
        )

    data[
        "UTC time"
    ] = pd.to_datetime(
        data[
            "UTC time"
        ],
        errors="coerce",
    )

    data[
        "Local time"
    ] = pd.to_datetime(
        data[
            "Local time"
        ],
        errors="coerce",
    )

    data[
        "Local date"
    ] = pd.to_datetime(
        data[
            "Local date"
        ],
        errors="coerce",
    )

    return data


def prepare_data(
    data: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    missing_demand = int(
        data[
            "Demand"
        ]
        .isna()
        .sum()
    )

    print(
        f"Rows with missing demand: "
        f"{missing_demand:,}"
    )

    complete = data.loc[
        data[
            "Demand"
        ].notna()
    ].copy()

    removed = (
        len(data)
        - len(complete)
    )

    print(
        f"Rows retained with observed demand: "
        f"{len(complete):,}"
    )

    print(
        f"Rows removed for missing demand: "
        f"{removed:,}"
    )

    complete[
        "year"
    ] = complete[
        "Local time"
    ].dt.year

    complete[
        "month"
    ] = complete[
        "Local time"
    ].dt.month

    complete[
        "hour_of_day"
    ] = complete[
        "Local time"
    ].dt.hour

    complete[
        "day_of_week"
    ] = complete[
        "Local time"
    ].dt.dayofweek

    complete[
        "is_weekend"
    ] = (
        complete[
            "day_of_week"
        ]
        >= 5
    )

    complete[
        "forecast_error"
    ] = (
        complete[
            "Demand"
        ]
        - complete[
            "Demand forecast"
        ]
    )

    complete[
        "absolute_forecast_error"
    ] = complete[
        "forecast_error"
    ].abs()

    nonzero_demand = (
        complete[
            "Demand"
        ]
        != 0
    )

    complete[
        "absolute_percentage_forecast_error"
    ] = np.nan

    complete.loc[
        nonzero_demand,
        "absolute_percentage_forecast_error",
    ] = (
        complete.loc[
            nonzero_demand,
            "absolute_forecast_error",
        ]
        / complete.loc[
            nonzero_demand,
            "Demand",
        ]
        * 100.0
    )

    return (
        complete,
        data,
    )


def create_overall_summary(
    complete: pd.DataFrame,
    raw: pd.DataFrame,
) -> pd.DataFrame:

    demand = complete[
        "Demand"
    ]

    summary = [
        {
            "metric":
                "raw_hourly_rows",
            "value":
                len(raw),
        },
        {
            "metric":
                "observed_demand_rows",
            "value":
                len(complete),
        },
        {
            "metric":
                "missing_demand_rows",
            "value":
                int(
                    raw[
                        "Demand"
                    ]
                    .isna()
                    .sum()
                ),
        },
        {
            "metric":
                "first_observed_utc",
            "value":
                complete[
                    "UTC time"
                ].min(),
        },
        {
            "metric":
                "last_observed_utc",
            "value":
                complete[
                    "UTC time"
                ].max(),
        },
        {
            "metric":
                "first_observed_local_time",
            "value":
                complete[
                    "Local time"
                ].min(),
        },
        {
            "metric":
                "last_observed_local_time",
            "value":
                complete[
                    "Local time"
                ].max(),
        },
        {
            "metric":
                "mean_hourly_demand_mwh",
            "value":
                demand.mean(),
        },
        {
            "metric":
                "median_hourly_demand_mwh",
            "value":
                demand.median(),
        },
        {
            "metric":
                "minimum_hourly_demand_mwh",
            "value":
                demand.min(),
        },
        {
            "metric":
                "maximum_hourly_demand_mwh",
            "value":
                demand.max(),
        },
        {
            "metric":
                "demand_p95_mwh",
            "value":
                demand.quantile(
                    0.95
                ),
        },
        {
            "metric":
                "demand_p99_mwh",
            "value":
                demand.quantile(
                    0.99
                ),
        },
        {
            "metric":
                "mean_net_generation_mwh",
            "value":
                complete[
                    "Net generation"
                ].mean(),
        },
        {
            "metric":
                "mean_total_interchange_mwh",
            "value":
                complete[
                    "Total interchange"
                ].mean(),
        },
        {
            "metric":
                "mean_absolute_forecast_error_mwh",
            "value":
                complete[
                    "absolute_forecast_error"
                ].mean(),
        },
        {
            "metric":
                "median_absolute_forecast_error_mwh",
            "value":
                complete[
                    "absolute_forecast_error"
                ].median(),
        },
        {
            "metric":
                "mean_absolute_percentage_forecast_error",
            "value":
                complete[
                    "absolute_percentage_forecast_error"
                ].mean(),
        },
    ]

    return pd.DataFrame(
        summary
    )


def create_annual_summary(
    complete: pd.DataFrame,
) -> pd.DataFrame:

    annual = (
        complete
        .groupby(
            "year",
            observed=True,
        )
        .agg(
            hours=(
                "Demand",
                "count",
            ),
            mean_demand_mwh=(
                "Demand",
                "mean",
            ),
            median_demand_mwh=(
                "Demand",
                "median",
            ),
            peak_demand_mwh=(
                "Demand",
                "max",
            ),
            mean_net_generation_mwh=(
                "Net generation",
                "mean",
            ),
            mean_total_interchange_mwh=(
                "Total interchange",
                "mean",
            ),
            mean_absolute_forecast_error_mwh=(
                "absolute_forecast_error",
                "mean",
            ),
        )
        .reset_index()
    )

    return annual


def create_monthly_profile(
    complete: pd.DataFrame,
) -> pd.DataFrame:

    monthly = (
        complete
        .groupby(
            "month",
            observed=True,
        )
        .agg(
            mean_demand_mwh=(
                "Demand",
                "mean",
            ),
            median_demand_mwh=(
                "Demand",
                "median",
            ),
            peak_demand_mwh=(
                "Demand",
                "max",
            ),
            mean_net_generation_mwh=(
                "Net generation",
                "mean",
            ),
            mean_total_interchange_mwh=(
                "Total interchange",
                "mean",
            ),
        )
        .reset_index()
    )

    return monthly


def create_hourly_profile(
    complete: pd.DataFrame,
) -> pd.DataFrame:

    hourly = (
        complete
        .groupby(
            "hour_of_day",
            observed=True,
        )
        .agg(
            mean_demand_mwh=(
                "Demand",
                "mean",
            ),
            median_demand_mwh=(
                "Demand",
                "median",
            ),
            mean_demand_forecast_mwh=(
                "Demand forecast",
                "mean",
            ),
            mean_net_generation_mwh=(
                "Net generation",
                "mean",
            ),
            mean_total_interchange_mwh=(
                "Total interchange",
                "mean",
            ),
        )
        .reset_index()
    )

    return hourly


def create_peak_hours(
    complete: pd.DataFrame,
) -> pd.DataFrame:

    columns = [
        "UTC time",
        "Local time",
        "Demand forecast",
        "Demand",
        "Net generation",
        "Total interchange",
        "forecast_error",
    ]

    peak_hours = (
        complete[
            columns
        ]
        .sort_values(
            "Demand",
            ascending=False,
        )
        .head(
            20
        )
        .reset_index(
            drop=True
        )
    )

    return peak_hours


def create_generation_mix(
    complete: pd.DataFrame,
) -> pd.DataFrame:

    rows: list[
        dict[str, object]
    ] = []

    total_positive_generation = 0.0

    source_totals: dict[
        str,
        float,
    ] = {}

    for source_name, column_name in (
        GENERATION_COLUMNS.items()
    ):

        source_series = (
            complete[
                column_name
            ]
            .fillna(
                0.0
            )
        )

        positive_total = float(
            source_series.clip(
                lower=0.0
            ).sum()
        )

        source_totals[
            source_name
        ] = positive_total

        total_positive_generation += (
            positive_total
        )

    for source_name, positive_total in (
        source_totals.items()
    ):

        if (
            total_positive_generation
            > 0
        ):
            share_percent = (
                positive_total
                / total_positive_generation
                * 100.0
            )
        else:
            share_percent = np.nan

        rows.append(
            {
                "generation_source":
                    source_name,
                "positive_generation_mwh":
                    positive_total,
                "share_of_positive_reported_generation_percent":
                    share_percent,
            }
        )

    result = pd.DataFrame(
        rows
    )

    result = result.sort_values(
        "positive_generation_mwh",
        ascending=False,
    ).reset_index(
        drop=True
    )

    return result


def create_forecast_error_summary(
    complete: pd.DataFrame,
) -> pd.DataFrame:

    valid = complete.loc[
        complete[
            "Demand forecast"
        ].notna()
    ].copy()

    if valid.empty:
        raise ValueError(
            "No rows contain both demand "
            "and demand forecast values."
        )

    actual = valid[
        "Demand"
    ]

    forecast = valid[
        "Demand forecast"
    ]

    error = (
        actual
        - forecast
    )

    mae = (
        error
        .abs()
        .mean()
    )

    rmse = float(
        np.sqrt(
            np.mean(
                np.square(
                    error
                )
            )
        )
    )

    mean_actual = (
        actual.mean()
    )

    if mean_actual != 0:
        cvrmse = (
            rmse
            / mean_actual
            * 100.0
        )
    else:
        cvrmse = np.nan

    nonzero = (
        actual
        != 0
    )

    if nonzero.any():
        mape = (
            np.abs(
                error.loc[
                    nonzero
                ]
                / actual.loc[
                    nonzero
                ]
            )
            .mean()
            * 100.0
        )
    else:
        mape = np.nan

    return pd.DataFrame(
        [
            {
                "rows":
                    len(valid),
                "MAE_mwh":
                    mae,
                "RMSE_mwh":
                    rmse,
                "CVRMSE_percent":
                    cvrmse,
                "MAPE_percent":
                    mape,
                "mean_forecast_error_mwh":
                    error.mean(),
            }
        ]
    )


def save_figures(
    hourly_profile: pd.DataFrame,
    monthly_profile: pd.DataFrame,
    generation_mix: pd.DataFrame,
) -> None:

    OUTPUT_FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(
            9,
            5,
        )
    )

    ax.plot(
        hourly_profile[
            "hour_of_day"
        ],
        hourly_profile[
            "mean_demand_mwh"
        ],
        marker="o",
    )

    ax.set_title(
        "California Average Hourly Electricity Demand"
    )

    ax.set_xlabel(
        "Local Hour"
    )

    ax.set_ylabel(
        "Average Demand (MWh)"
    )

    ax.set_xticks(
        range(
            0,
            24,
            2,
        )
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        HOURLY_DEMAND_FIGURE_PATH,
        dpi=200,
    )

    plt.close(
        fig
    )

    fig, ax = plt.subplots(
        figsize=(
            9,
            5,
        )
    )

    ax.plot(
        monthly_profile[
            "month"
        ],
        monthly_profile[
            "mean_demand_mwh"
        ],
        marker="o",
    )

    ax.set_title(
        "California Average Electricity Demand by Month"
    )

    ax.set_xlabel(
        "Month"
    )

    ax.set_ylabel(
        "Average Demand (MWh)"
    )

    ax.set_xticks(
        range(
            1,
            13,
        )
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        MONTHLY_DEMAND_FIGURE_PATH,
        dpi=200,
    )

    plt.close(
        fig
    )

    generation_plot = (
        generation_mix
        .head(
            8
        )
        .sort_values(
            "share_of_positive_reported_generation_percent"
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            9,
            5,
        )
    )

    ax.barh(
        generation_plot[
            "generation_source"
        ],
        generation_plot[
            "share_of_positive_reported_generation_percent"
        ],
    )

    ax.set_title(
        "California Reported Generation Mix"
    )

    ax.set_xlabel(
        "Share of Positive Reported Generation (%)"
    )

    ax.set_ylabel(
        "Generation Source"
    )

    fig.tight_layout()

    fig.savefig(
        GENERATION_MIX_FIGURE_PATH,
        dpi=200,
    )

    plt.close(
        fig
    )


def main() -> None:

    OUTPUT_TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_data()

    (
        complete,
        raw,
    ) = prepare_data(
        data
    )

    print()

    print(
        f"Observed data period: "
        f"{complete['Local time'].min()} "
        f"to "
        f"{complete['Local time'].max()}"
    )

    summary = (
        create_overall_summary(
            complete,
            raw,
        )
    )

    annual_summary = (
        create_annual_summary(
            complete
        )
    )

    monthly_profile = (
        create_monthly_profile(
            complete
        )
    )

    hourly_profile = (
        create_hourly_profile(
            complete
        )
    )

    peak_hours = (
        create_peak_hours(
            complete
        )
    )

    generation_mix = (
        create_generation_mix(
            complete
        )
    )

    forecast_error_summary = (
        create_forecast_error_summary(
            complete
        )
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    annual_summary.to_csv(
        ANNUAL_SUMMARY_PATH,
        index=False,
    )

    monthly_profile.to_csv(
        MONTHLY_PROFILE_PATH,
        index=False,
    )

    hourly_profile.to_csv(
        HOURLY_PROFILE_PATH,
        index=False,
    )

    peak_hours.to_csv(
        PEAK_HOURS_PATH,
        index=False,
    )

    generation_mix.to_csv(
        GENERATION_MIX_PATH,
        index=False,
    )

    forecast_error_summary.to_csv(
        FORECAST_ERROR_PATH,
        index=False,
    )

    save_figures(
        hourly_profile,
        monthly_profile,
        generation_mix,
    )

    print()
    print(
        "=" * 70
    )

    print(
        "EIA-930 California contextual analysis complete."
    )

    print()
    print(
        "Overall summary:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Demand forecast error:"
    )

    print(
        forecast_error_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Top generation sources:"
    )

    print(
        generation_mix.head(
            10
        ).to_string(
            index=False
        )
    )

    print()
    print(
        "Peak demand hours:"
    )

    print(
        peak_hours.head(
            10
        ).to_string(
            index=False
        )
    )

    print()
    print(
        f"Summary: "
        f"{SUMMARY_PATH}"
    )

    print(
        f"Annual summary: "
        f"{ANNUAL_SUMMARY_PATH}"
    )

    print(
        f"Monthly profile: "
        f"{MONTHLY_PROFILE_PATH}"
    )

    print(
        f"Hourly profile: "
        f"{HOURLY_PROFILE_PATH}"
    )

    print(
        f"Peak hours: "
        f"{PEAK_HOURS_PATH}"
    )

    print(
        f"Generation mix: "
        f"{GENERATION_MIX_PATH}"
    )

    print(
        f"Forecast error: "
        f"{FORECAST_ERROR_PATH}"
    )

    print(
        f"Hourly-demand figure: "
        f"{HOURLY_DEMAND_FIGURE_PATH}"
    )

    print(
        f"Monthly-demand figure: "
        f"{MONTHLY_DEMAND_FIGURE_PATH}"
    )

    print(
        f"Generation-mix figure: "
        f"{GENERATION_MIX_FIGURE_PATH}"
    )


if __name__ == "__main__":
    main()