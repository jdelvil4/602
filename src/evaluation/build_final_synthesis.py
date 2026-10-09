from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

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


RESSTOCK_PATH = (
    TABLE_DIR
    / "nrel_final_test_metrics.csv"
)

COMSTOCK_PATH = (
    TABLE_DIR
    / "comstock_final_test_metrics.csv"
)

BDG2_PATH = (
    TABLE_DIR
    / "bdg2_final_test_metrics.csv"
)

TRANSFER_PATH = (
    TABLE_DIR
    / "comstock_to_bdg2_transfer_metrics.csv"
)

EIA_PATH = (
    TABLE_DIR
    / "eia930_california_forecast_error_summary.csv"
)


MASTER_RESULTS_PATH = (
    TABLE_DIR
    / "final_model_performance_summary.csv"
)

IMPROVEMENT_PATH = (
    TABLE_DIR
    / "final_mae_improvement_vs_seasonal.csv"
)

TRANSFER_SUMMARY_PATH = (
    TABLE_DIR
    / "final_transfer_comparison.csv"
)

PROJECT_FINDINGS_PATH = (
    TABLE_DIR
    / "final_project_key_findings.csv"
)


IMPROVEMENT_FIGURE_PATH = (
    FIGURE_DIR
    / "final_mae_improvement_vs_seasonal.png"
)

TRANSFER_FIGURE_PATH = (
    FIGURE_DIR
    / "final_simulated_vs_real_transfer.png"
)


def require_file(
    path: Path,
) -> None:

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


def find_column(
    data: pd.DataFrame,
    candidates: list[str],
) -> str:

    for candidate in candidates:
        if candidate in data.columns:
            return candidate

    raise ValueError(
        "Could not identify required column. "
        f"Candidates: {candidates}. "
        f"Available columns: {data.columns.tolist()}"
    )


def standardize_final_metrics(
    data: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:

    dataset_column = None

    for candidate in [
        "dataset",
        "test_set",
        "split",
        "evaluation_set",
        "Test Set",
    ]:
        if candidate in data.columns:
            dataset_column = candidate
            break

    model_column = find_column(
        data,
        [
            "model",
            "Model",
        ],
    )

    mae_column = find_column(
        data,
        [
            "MAE",
            "mae",
        ],
    )

    cvrmse_column = find_column(
        data,
        [
            "CVRMSE_percent",
            "CVRMSE",
            "CVRMSE (%)",
            "cvrmse",
            "cvrmse_pct",
            "cvrmse_percent",
        ],
    )

    mape_column = find_column(
        data,
        [
            "MAPE_percent",
            "MAPE",
            "MAPE (%)",
            "mape",
            "mape_pct",
            "mape_percent",
        ],
    )

    standardized = pd.DataFrame(
        index=data.index
    )

    standardized[
        "source_dataset"
    ] = dataset_name

    if dataset_column is None:
        standardized[
            "evaluation_split"
        ] = "Final Test"
    else:
        standardized[
            "evaluation_split"
        ] = (
            data[
                dataset_column
            ]
            .astype(
                str
            )
            .str.replace(
                "_",
                " ",
                regex=False,
            )
            .str.strip()
            .str.title()
        )

    standardized[
        "model"
    ] = data[
        model_column
    ].astype(
        str
    )

    standardized[
        "MAE"
    ] = pd.to_numeric(
        data[
            mae_column
        ],
        errors="coerce",
    )

    standardized[
        "CVRMSE_percent"
    ] = pd.to_numeric(
        data[
            cvrmse_column
        ],
        errors="coerce",
    )

    standardized[
        "MAPE_percent"
    ] = pd.to_numeric(
        data[
            mape_column
        ],
        errors="coerce",
    )

    return standardized.reset_index(
        drop=True
    )


def build_master_results() -> pd.DataFrame:

    paths = [
        (
            RESSTOCK_PATH,
            "NREL ResStock",
        ),
        (
            COMSTOCK_PATH,
            "NREL ComStock",
        ),
        (
            BDG2_PATH,
            "BDG2",
        ),
        (
            TRANSFER_PATH,
            "ComStock-to-BDG2 Transfer",
        ),
    ]

    frames: list[
        pd.DataFrame
    ] = []

    for path, dataset_name in paths:

        require_file(
            path
        )

        data = pd.read_csv(
            path
        )

        standardized = (
            standardize_final_metrics(
                data,
                dataset_name,
            )
        )

        frames.append(
            standardized
        )

    master = pd.concat(
        frames,
        ignore_index=True,
    )

    return master


def identify_seasonal_row(
    group: pd.DataFrame,
) -> pd.Series:

    mask = (
        group[
            "model"
        ]
        .str.lower()
        .str.contains(
            "seasonal"
        )
    )

    seasonal = group.loc[
        mask
    ]

    if len(
        seasonal
    ) != 1:
        raise ValueError(
            "Expected exactly one Seasonal Naive "
            "row within evaluation group."
        )

    return seasonal.iloc[
        0
    ]


def build_improvement_table(
    master: pd.DataFrame,
) -> pd.DataFrame:

    rows: list[
        dict[str, object]
    ] = []

    eligible_sources = [
        "NREL ResStock",
        "NREL ComStock",
        "BDG2",
    ]

    for source_name in eligible_sources:

        source = master.loc[
            master[
                "source_dataset"
            ]
            == source_name
        ]

        for split_name, group in (
            source.groupby(
                "evaluation_split",
                sort=False,
            )
        ):

            seasonal = (
                identify_seasonal_row(
                    group
                )
            )

            candidate_rows = (
                group.loc[
                    ~group[
                        "model"
                    ]
                    .str.lower()
                    .str.contains(
                        "seasonal"
                    )
                ]
            )

            if candidate_rows.empty:
                continue

            best_row = (
                candidate_rows
                .sort_values(
                    "MAE"
                )
                .iloc[
                    0
                ]
            )

            seasonal_mae = float(
                seasonal[
                    "MAE"
                ]
            )

            model_mae = float(
                best_row[
                    "MAE"
                ]
            )

            improvement = (
                (
                    seasonal_mae
                    - model_mae
                )
                / seasonal_mae
                * 100.0
            )

            rows.append(
                {
                    "dataset":
                        source_name,
                    "evaluation_split":
                        split_name,
                    "selected_model":
                        best_row[
                            "model"
                        ],
                    "model_MAE":
                        model_mae,
                    "seasonal_MAE":
                        seasonal_mae,
                    "MAE_improvement_percent":
                        improvement,
                }
            )

    return pd.DataFrame(
        rows
    )


def build_transfer_table(
    master: pd.DataFrame,
) -> pd.DataFrame:

    transfer = master.loc[
        master[
            "source_dataset"
        ]
        == "ComStock-to-BDG2 Transfer"
    ].copy()

    if transfer.empty:
        raise ValueError(
            "Transfer results were not found "
            "in the master results table."
        )

    return transfer[
        [
            "evaluation_split",
            "model",
            "MAE",
            "CVRMSE_percent",
            "MAPE_percent",
        ]
    ].reset_index(
        drop=True
    )


def create_project_findings(
    improvement: pd.DataFrame,
    transfer: pd.DataFrame,
) -> pd.DataFrame:

    require_file(
        EIA_PATH
    )

    eia = pd.read_csv(
        EIA_PATH
    )

    rows: list[
        dict[str, object]
    ] = []

    rows.append(
        {
            "topic":
                "UCI prototype",
            "finding":
                (
                    "Tuned Extra Trees completed the "
                    "prototype workflow with final-test "
                    "MAE 182.259, CVRMSE 52.608%, "
                    "and MAPE 29.991%."
                ),
        }
    )

    resstock_temporal = (
        improvement.loc[
            (
                improvement[
                    "dataset"
                ]
                == "NREL ResStock"
            )
            & (
                improvement[
                    "evaluation_split"
                ]
                .str.lower()
                .str.contains(
                    "temporal"
                )
            )
        ]
    )

    if not resstock_temporal.empty:
        value = float(
            resstock_temporal.iloc[
                0
            ][
                "MAE_improvement_percent"
            ]
        )

        rows.append(
            {
                "topic":
                    "Residential forecasting",
                "finding":
                    (
                        "ResStock final model improved "
                        f"temporal-test MAE by {value:.1f}% "
                        "relative to Seasonal Naive."
                    ),
            }
        )

    comstock_temporal = (
        improvement.loc[
            (
                improvement[
                    "dataset"
                ]
                == "NREL ComStock"
            )
            & (
                improvement[
                    "evaluation_split"
                ]
                .str.lower()
                .str.contains(
                    "temporal"
                )
            )
        ]
    )

    if not comstock_temporal.empty:
        value = float(
            comstock_temporal.iloc[
                0
            ][
                "MAE_improvement_percent"
            ]
        )

        rows.append(
            {
                "topic":
                    "Commercial simulated forecasting",
                "finding":
                    (
                        "ComStock final model improved "
                        f"temporal-test MAE by {value:.1f}% "
                        "relative to Seasonal Naive."
                    ),
            }
        )

    bdg2_heldout = (
        improvement.loc[
            (
                improvement[
                    "dataset"
                ]
                == "BDG2"
            )
            & (
                improvement[
                    "evaluation_split"
                ]
                .str.lower()
                .str.contains(
                    "held"
                )
            )
        ]
    )

    if not bdg2_heldout.empty:
        value = float(
            bdg2_heldout.iloc[
                0
            ][
                "MAE_improvement_percent"
            ]
        )

        rows.append(
            {
                "topic":
                    "Real-building generalization",
                "finding":
                    (
                        "BDG2 internal model improved "
                        f"held-out-building MAE by {value:.1f}% "
                        "relative to Seasonal Naive."
                    ),
            }
        )

    transfer_temporal = (
        transfer.loc[
            transfer[
                "evaluation_split"
            ]
            .str.lower()
            .str.contains(
                "temporal"
            )
        ]
    )

    if not transfer_temporal.empty:

        transfer_row = (
            transfer_temporal.loc[
                transfer_temporal[
                    "model"
                ]
                .str.lower()
                .str.contains(
                    "transfer"
                )
            ]
            .iloc[
                0
            ]
        )

        internal_row = (
            transfer_temporal.loc[
                transfer_temporal[
                    "model"
                ]
                .str.lower()
                .str.contains(
                    "internal"
                )
            ]
            .iloc[
                0
            ]
        )

        rows.append(
            {
                "topic":
                    "Simulation-to-real transfer",
                "finding":
                    (
                        "Direct ComStock-to-BDG2 transfer "
                        f"produced temporal MAE "
                        f"{float(transfer_row['MAE']):.3f}, "
                        "compared with "
                        f"{float(internal_row['MAE']):.3f} "
                        "for the internally trained BDG2 model."
                    ),
            }
        )

    eia_mae_column = find_column(
        eia,
        [
            "MAE_mwh",
            "MAE",
        ],
    )

    eia_cvrmse_column = find_column(
        eia,
        [
            "CVRMSE_percent",
            "CVRMSE",
        ],
    )

    eia_mape_column = find_column(
        eia,
        [
            "MAPE_percent",
            "MAPE",
        ],
    )

    eia_row = eia.iloc[
        0
    ]

    rows.append(
        {
            "topic":
                "California grid context",
            "finding":
                (
                    "EIA-930 demand forecast achieved "
                    f"MAE {float(eia_row[eia_mae_column]):.1f} MWh, "
                    f"CVRMSE {float(eia_row[eia_cvrmse_column]):.2f}%, "
                    f"and MAPE {float(eia_row[eia_mape_column]):.2f}%."
                ),
        }
    )

    return pd.DataFrame(
        rows
    )


def plot_improvement(
    improvement: pd.DataFrame,
) -> None:

    plot_data = (
        improvement.copy()
    )

    plot_data[
        "label"
    ] = (
        plot_data[
            "dataset"
        ]
        + "\n"
        + plot_data[
            "evaluation_split"
        ]
    )

    fig, ax = plt.subplots(
        figsize=(
            11,
            6,
        )
    )

    ax.bar(
        plot_data[
            "label"
        ],
        plot_data[
            "MAE_improvement_percent"
        ],
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_title(
        "Final Model MAE Improvement Relative to Seasonal Naive"
    )

    ax.set_ylabel(
        "MAE Improvement (%)"
    )

    ax.set_xlabel(
        ""
    )

    ax.tick_params(
        axis="x",
        rotation=30,
    )

    fig.tight_layout()

    fig.savefig(
        IMPROVEMENT_FIGURE_PATH,
        dpi=200,
    )

    plt.close(
        fig
    )


def plot_transfer(
    transfer: pd.DataFrame,
) -> None:

    pivot = (
        transfer
        .pivot(
            index="evaluation_split",
            columns="model",
            values="MAE",
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    pivot.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title(
        "Simulated-to-Real Generalization on BDG2"
    )

    ax.set_ylabel(
        "MAE"
    )

    ax.set_xlabel(
        ""
    )

    ax.tick_params(
        axis="x",
        rotation=0,
    )

    ax.legend(
        title="Model",
        fontsize=8,
    )

    fig.tight_layout()

    fig.savefig(
        TRANSFER_FIGURE_PATH,
        dpi=200,
    )

    plt.close(
        fig
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

    print(
        "Building final project synthesis..."
    )

    master = (
        build_master_results()
    )

    improvement = (
        build_improvement_table(
            master
        )
    )

    transfer = (
        build_transfer_table(
            master
        )
    )

    findings = (
        create_project_findings(
            improvement,
            transfer,
        )
    )

    master.to_csv(
        MASTER_RESULTS_PATH,
        index=False,
    )

    improvement.to_csv(
        IMPROVEMENT_PATH,
        index=False,
    )

    transfer.to_csv(
        TRANSFER_SUMMARY_PATH,
        index=False,
    )

    findings.to_csv(
        PROJECT_FINDINGS_PATH,
        index=False,
    )

    plot_improvement(
        improvement
    )

    plot_transfer(
        transfer
    )

    print()
    print(
        "=" * 70
    )

    print(
        "Final synthesis complete."
    )

    print()
    print(
        "MAE improvement versus Seasonal Naive:"
    )

    print(
        improvement.to_string(
            index=False
        )
    )

    print()
    print(
        "Simulated-to-real transfer comparison:"
    )

    print(
        transfer.to_string(
            index=False
        )
    )

    print()
    print(
        "Key project findings:"
    )

    print(
        findings.to_string(
            index=False
        )
    )

    print()
    print(
        f"Master performance table: "
        f"{MASTER_RESULTS_PATH}"
    )

    print(
        f"MAE improvement table: "
        f"{IMPROVEMENT_PATH}"
    )

    print(
        f"Transfer summary: "
        f"{TRANSFER_SUMMARY_PATH}"
    )

    print(
        f"Key findings: "
        f"{PROJECT_FINDINGS_PATH}"
    )

    print(
        f"MAE improvement figure: "
        f"{IMPROVEMENT_FIGURE_PATH}"
    )

    print(
        f"Transfer figure: "
        f"{TRANSFER_FIGURE_PATH}"
    )


if __name__ == "__main__":
    main()