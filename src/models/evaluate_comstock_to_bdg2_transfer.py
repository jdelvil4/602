from pathlib import Path

import joblib
import pandas as pd

from src.evaluation.metrics import calculate_all_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRANSFER_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "comstock_shared_feature_random_forest_100trees.joblib"
)

BDG2_INTERNAL_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "bdg2_random_forest_tuned.joblib"
)

TEMPORAL_TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bdg2_temporal_test.parquet"
)

HELD_OUT_TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bdg2_held_out_test.parquet"
)

OVERALL_RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "comstock_to_bdg2_transfer_metrics.csv"
)

PER_BUILDING_RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "comstock_to_bdg2_transfer_metrics_by_building.csv"
)

TRANSFER_WIN_SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "comstock_to_bdg2_transfer_win_summary.csv"
)

FEATURE_COLUMNS = [
    "electricity_lag_1",
    "electricity_lag_24",
    "electricity_rolling_24h_mean",
    "electricity_rolling_168h_mean",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "airTemperature",
    "windSpeed",
    "heating_degree",
    "cooling_degree",
]

TARGET_COLUMN = (
    "target_electricity_next_hour"
)

SEASONAL_NAIVE_COLUMN = (
    "seasonal_naive_24h"
)

BUILDING_COLUMN = (
    "building_id"
)

TIMESTAMP_COLUMN = (
    "target_timestamp"
)


def validate_dataset(
    data: pd.DataFrame,
    dataset_name: str,
) -> None:

    required_columns = [
        BUILDING_COLUMN,
        TIMESTAMP_COLUMN,
        TARGET_COLUMN,
        SEASONAL_NAIVE_COLUMN,
        *FEATURE_COLUMNS,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{missing_columns}"
        )

    missing_values = int(
        data[
            required_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    if missing_values > 0:
        raise ValueError(
            f"{dataset_name} contains "
            f"{missing_values:,} missing required values."
        )

    duplicate_count = int(
        data.duplicated(
            subset=[
                BUILDING_COLUMN,
                TIMESTAMP_COLUMN,
            ]
        ).sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"{dataset_name} contains "
            f"{duplicate_count:,} duplicate "
            "building-target timestamp rows."
        )


def evaluate_model(
    dataset_name: str,
    model_name: str,
    actual: pd.Series,
    predicted,
) -> dict[str, object]:

    metrics = calculate_all_metrics(
        actual,
        predicted,
    )

    return {
        "dataset": dataset_name,
        "model": model_name,
        "rows": len(actual),
        **metrics,
    }


def evaluate_per_building(
    data: pd.DataFrame,
    transfer_predictions,
    internal_predictions,
    dataset_name: str,
) -> list[dict[str, object]]:

    working = data[
        [
            BUILDING_COLUMN,
            TARGET_COLUMN,
            SEASONAL_NAIVE_COLUMN,
        ]
    ].copy()

    working[
        "transfer_prediction"
    ] = transfer_predictions

    working[
        "internal_prediction"
    ] = internal_predictions

    results: list[
        dict[str, object]
    ] = []

    for building_id, group in working.groupby(
        BUILDING_COLUMN,
        sort=True,
    ):

        actual = group[
            TARGET_COLUMN
        ]

        transfer_metrics = (
            calculate_all_metrics(
                actual,
                group[
                    "transfer_prediction"
                ],
            )
        )

        internal_metrics = (
            calculate_all_metrics(
                actual,
                group[
                    "internal_prediction"
                ],
            )
        )

        seasonal_metrics = (
            calculate_all_metrics(
                actual,
                group[
                    SEASONAL_NAIVE_COLUMN
                ],
            )
        )

        results.append(
            {
                "dataset":
                    dataset_name,
                "building_id":
                    building_id,
                "rows":
                    len(group),
                "transfer_MAE":
                    transfer_metrics[
                        "MAE"
                    ],
                "transfer_CVRMSE_percent":
                    transfer_metrics[
                        "CVRMSE_percent"
                    ],
                "transfer_MAPE_percent":
                    transfer_metrics[
                        "MAPE_percent"
                    ],
                "internal_MAE":
                    internal_metrics[
                        "MAE"
                    ],
                "internal_CVRMSE_percent":
                    internal_metrics[
                        "CVRMSE_percent"
                    ],
                "internal_MAPE_percent":
                    internal_metrics[
                        "MAPE_percent"
                    ],
                "seasonal_MAE":
                    seasonal_metrics[
                        "MAE"
                    ],
                "seasonal_CVRMSE_percent":
                    seasonal_metrics[
                        "CVRMSE_percent"
                    ],
                "seasonal_MAPE_percent":
                    seasonal_metrics[
                        "MAPE_percent"
                    ],
                "transfer_beats_seasonal_MAE":
                    transfer_metrics[
                        "MAE"
                    ]
                    < seasonal_metrics[
                        "MAE"
                    ],
                "transfer_beats_internal_MAE":
                    transfer_metrics[
                        "MAE"
                    ]
                    < internal_metrics[
                        "MAE"
                    ],
            }
        )

    return results


def evaluate_dataset(
    transfer_model,
    internal_model,
    data: pd.DataFrame,
    dataset_name: str,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
]:

    print()
    print(
        "=" * 70
    )

    print(
        f"Evaluating {dataset_name}..."
    )

    print(
        f"Rows: "
        f"{len(data):,}"
    )

    print(
        f"Buildings: "
        f"{data[BUILDING_COLUMN].nunique():,}"
    )

    print(
        f"Target period: "
        f"{data[TIMESTAMP_COLUMN].min()} "
        f"to "
        f"{data[TIMESTAMP_COLUMN].max()}"
    )

    X = data[
        FEATURE_COLUMNS
    ]

    y = data[
        TARGET_COLUMN
    ]

    transfer_predictions = (
        transfer_model.predict(
            X
        )
    )

    internal_predictions = (
        internal_model.predict(
            X
        )
    )

    seasonal_predictions = (
        data[
            SEASONAL_NAIVE_COLUMN
        ]
    )

    overall_results = [
        evaluate_model(
            dataset_name=dataset_name,
            model_name=(
                "ComStock-to-BDG2 "
                "Transfer Random Forest"
            ),
            actual=y,
            predicted=(
                transfer_predictions
            ),
        ),
        evaluate_model(
            dataset_name=dataset_name,
            model_name=(
                "BDG2 Internal "
                "Tuned Random Forest"
            ),
            actual=y,
            predicted=(
                internal_predictions
            ),
        ),
        evaluate_model(
            dataset_name=dataset_name,
            model_name="Seasonal Naive",
            actual=y,
            predicted=(
                seasonal_predictions
            ),
        ),
    ]

    for result in overall_results:

        print()
        print(
            result
        )

    per_building = (
        evaluate_per_building(
            data=data,
            transfer_predictions=(
                transfer_predictions
            ),
            internal_predictions=(
                internal_predictions
            ),
            dataset_name=(
                dataset_name
            ),
        )
    )

    return (
        overall_results,
        per_building,
    )


def create_win_summary(
    per_building_df: pd.DataFrame,
) -> pd.DataFrame:

    rows: list[
        dict[str, object]
    ] = []

    for dataset_name, group in (
        per_building_df.groupby(
            "dataset",
            sort=True,
        )
    ):

        total_buildings = (
            len(
                group
            )
        )

        transfer_vs_seasonal_wins = int(
            group[
                "transfer_beats_seasonal_MAE"
            ].sum()
        )

        transfer_vs_internal_wins = int(
            group[
                "transfer_beats_internal_MAE"
            ].sum()
        )

        rows.append(
            {
                "dataset":
                    dataset_name,
                "total_buildings":
                    total_buildings,
                "transfer_MAE_wins_vs_seasonal":
                    transfer_vs_seasonal_wins,
                "transfer_MAE_win_percent_vs_seasonal":
                    (
                        transfer_vs_seasonal_wins
                        / total_buildings
                        * 100.0
                    ),
                "transfer_MAE_wins_vs_internal":
                    transfer_vs_internal_wins,
                "transfer_MAE_win_percent_vs_internal":
                    (
                        transfer_vs_internal_wins
                        / total_buildings
                        * 100.0
                    ),
                "median_transfer_MAE":
                    group[
                        "transfer_MAE"
                    ].median(),
                "median_internal_MAE":
                    group[
                        "internal_MAE"
                    ].median(),
                "median_seasonal_MAE":
                    group[
                        "seasonal_MAE"
                    ].median(),
            }
        )

    return pd.DataFrame(
        rows
    )


def main() -> None:

    print(
        "Loading ComStock shared-feature "
        "transfer model..."
    )

    if not TRANSFER_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Transfer model not found: "
            f"{TRANSFER_MODEL_PATH}"
        )

    transfer_model = joblib.load(
        TRANSFER_MODEL_PATH
    )

    print(
        "Loading BDG2 internal benchmark model..."
    )

    if not BDG2_INTERNAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"BDG2 internal model not found: "
            f"{BDG2_INTERNAL_MODEL_PATH}"
        )

    internal_model = joblib.load(
        BDG2_INTERNAL_MODEL_PATH
    )

    required_columns = [
        BUILDING_COLUMN,
        TIMESTAMP_COLUMN,
        TARGET_COLUMN,
        SEASONAL_NAIVE_COLUMN,
        *FEATURE_COLUMNS,
    ]

    print(
        "Loading BDG2 temporal test..."
    )

    temporal_test = pd.read_parquet(
        TEMPORAL_TEST_PATH,
        columns=required_columns,
    )

    print(
        "Loading BDG2 held-out-building test..."
    )

    held_out_test = pd.read_parquet(
        HELD_OUT_TEST_PATH,
        columns=required_columns,
    )

    temporal_test[
        TIMESTAMP_COLUMN
    ] = pd.to_datetime(
        temporal_test[
            TIMESTAMP_COLUMN
        ],
        errors="raise",
    )

    held_out_test[
        TIMESTAMP_COLUMN
    ] = pd.to_datetime(
        held_out_test[
            TIMESTAMP_COLUMN
        ],
        errors="raise",
    )

    validate_dataset(
        temporal_test,
        "Temporal Test",
    )

    validate_dataset(
        held_out_test,
        "Held-Out Building Test",
    )

    overall_results: list[
        dict[str, object]
    ] = []

    per_building_results: list[
        dict[str, object]
    ] = []

    (
        temporal_overall,
        temporal_per_building,
    ) = evaluate_dataset(
        transfer_model=transfer_model,
        internal_model=internal_model,
        data=temporal_test,
        dataset_name="Temporal Test",
    )

    overall_results.extend(
        temporal_overall
    )

    per_building_results.extend(
        temporal_per_building
    )

    (
        held_out_overall,
        held_out_per_building,
    ) = evaluate_dataset(
        transfer_model=transfer_model,
        internal_model=internal_model,
        data=held_out_test,
        dataset_name=(
            "Held-Out Building Test"
        ),
    )

    overall_results.extend(
        held_out_overall
    )

    per_building_results.extend(
        held_out_per_building
    )

    overall_df = pd.DataFrame(
        overall_results
    )

    per_building_df = pd.DataFrame(
        per_building_results
    )

    win_summary_df = (
        create_win_summary(
            per_building_df
        )
    )

    OVERALL_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall_df.to_csv(
        OVERALL_RESULTS_PATH,
        index=False,
    )

    per_building_df.to_csv(
        PER_BUILDING_RESULTS_PATH,
        index=False,
    )

    win_summary_df.to_csv(
        TRANSFER_WIN_SUMMARY_PATH,
        index=False,
    )

    print()
    print(
        "=" * 70
    )

    print(
        "ComStock-to-BDG2 transfer "
        "evaluation complete."
    )

    print()
    print(
        "Overall transfer results:"
    )

    print(
        overall_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Per-building transfer win summary:"
    )

    print(
        win_summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Overall results: "
        f"{OVERALL_RESULTS_PATH}"
    )

    print(
        f"Per-building results: "
        f"{PER_BUILDING_RESULTS_PATH}"
    )

    print(
        f"Win summary: "
        f"{TRANSFER_WIN_SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()