from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.evaluation.metrics import calculate_all_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "bdg2_random_forest_tuned.joblib"
)

OVERALL_METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "bdg2_final_test_metrics.csv"
)

PER_BUILDING_METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "bdg2_final_test_metrics_by_building.csv"
)

BUILDING_WIN_SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "bdg2_final_building_win_summary.csv"
)

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
            f"building-target timestamp rows."
        )


def evaluate_overall(
    dataset_name: str,
    model_name: str,
    y_true: pd.Series,
    y_pred,
) -> dict[str, object]:

    metrics = calculate_all_metrics(
        y_true,
        y_pred,
    )

    return {
        "dataset": dataset_name,
        "model": model_name,
        "rows": len(y_true),
        **metrics,
    }


def evaluate_per_building(
    data: pd.DataFrame,
    predictions: np.ndarray,
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
        "rf_prediction"
    ] = predictions

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

        rf_pred = group[
            "rf_prediction"
        ]

        seasonal_pred = group[
            SEASONAL_NAIVE_COLUMN
        ]

        rf_metrics = calculate_all_metrics(
            actual,
            rf_pred,
        )

        seasonal_metrics = calculate_all_metrics(
            actual,
            seasonal_pred,
        )

        results.append(
            {
                "dataset":
                    dataset_name,
                "building_id":
                    building_id,
                "rows":
                    len(group),
                "rf_MAE":
                    rf_metrics[
                        "MAE"
                    ],
                "rf_CVRMSE_percent":
                    rf_metrics[
                        "CVRMSE_percent"
                    ],
                "rf_MAPE_percent":
                    rf_metrics[
                        "MAPE_percent"
                    ],
                "rf_MAPE_n":
                    rf_metrics[
                        "MAPE_n"
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
                "seasonal_MAPE_n":
                    seasonal_metrics[
                        "MAPE_n"
                    ],
                "rf_beats_seasonal_MAE":
                    rf_metrics[
                        "MAE"
                    ]
                    < seasonal_metrics[
                        "MAE"
                    ],
            }
        )

    return results


def evaluate_dataset(
    model,
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

    rf_predictions = model.predict(
        X
    )

    seasonal_predictions = data[
        SEASONAL_NAIVE_COLUMN
    ]

    rf_result = evaluate_overall(
        dataset_name=dataset_name,
        model_name="Tuned Random Forest",
        y_true=y,
        y_pred=rf_predictions,
    )

    seasonal_result = evaluate_overall(
        dataset_name=dataset_name,
        model_name="Seasonal Naive",
        y_true=y,
        y_pred=seasonal_predictions,
    )

    print()
    print(
        "Tuned Random Forest:"
    )

    print(
        rf_result
    )

    print()
    print(
        "Seasonal Naive:"
    )

    print(
        seasonal_result
    )

    per_building_results = (
        evaluate_per_building(
            data=data,
            predictions=rf_predictions,
            dataset_name=dataset_name,
        )
    )

    return (
        [
            rf_result,
            seasonal_result,
        ],
        per_building_results,
    )


def build_win_summary(
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

        total_buildings = len(
            group
        )

        rf_wins = int(
            group[
                "rf_beats_seasonal_MAE"
            ].sum()
        )

        seasonal_wins_or_ties = (
            total_buildings
            - rf_wins
        )

        rf_win_percent = (
            rf_wins
            / total_buildings
            * 100.0
        )

        rows.append(
            {
                "dataset":
                    dataset_name,
                "total_buildings":
                    total_buildings,
                "rf_MAE_wins":
                    rf_wins,
                "seasonal_MAE_wins_or_ties":
                    seasonal_wins_or_ties,
                "rf_MAE_win_percent":
                    rf_win_percent,
                "median_rf_MAE":
                    group[
                        "rf_MAE"
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
        "Loading locked BDG2 tuned Random Forest..."
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: "
            f"{MODEL_PATH}"
        )

    model = joblib.load(
        MODEL_PATH
    )

    required_columns = [
        BUILDING_COLUMN,
        TIMESTAMP_COLUMN,
        TARGET_COLUMN,
        SEASONAL_NAIVE_COLUMN,
        *FEATURE_COLUMNS,
    ]

    print(
        "Loading BDG2 temporal test data..."
    )

    temporal_test = pd.read_parquet(
        TEMPORAL_TEST_PATH,
        columns=required_columns,
    )

    print(
        "Loading BDG2 held-out-building test data..."
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

    print()
    print(
        "Dataset validation complete."
    )

    print(
        f"Temporal test rows: "
        f"{len(temporal_test):,}"
    )

    print(
        f"Temporal test buildings: "
        f"{temporal_test[BUILDING_COLUMN].nunique():,}"
    )

    print(
        f"Held-out test rows: "
        f"{len(held_out_test):,}"
    )

    print(
        f"Held-out test buildings: "
        f"{held_out_test[BUILDING_COLUMN].nunique():,}"
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
        model=model,
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
        model=model,
        data=held_out_test,
        dataset_name="Held-Out Building Test",
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
        build_win_summary(
            per_building_df
        )
    )

    OVERALL_METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall_df.to_csv(
        OVERALL_METRICS_PATH,
        index=False,
    )

    per_building_df.to_csv(
        PER_BUILDING_METRICS_PATH,
        index=False,
    )

    win_summary_df.to_csv(
        BUILDING_WIN_SUMMARY_PATH,
        index=False,
    )

    print()
    print(
        "=" * 70
    )

    print(
        "BDG2 final evaluation complete."
    )

    print()
    print(
        "Overall test metrics:"
    )

    print(
        overall_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Per-building MAE win summary:"
    )

    print(
        win_summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Overall metrics: "
        f"{OVERALL_METRICS_PATH}"
    )

    print(
        f"Per-building metrics: "
        f"{PER_BUILDING_METRICS_PATH}"
    )

    print(
        f"Building win summary: "
        f"{BUILDING_WIN_SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()