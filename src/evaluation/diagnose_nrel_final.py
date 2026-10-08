from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "nrel_final_test_predictions.parquet"
)

MODELING_BASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_modeling_base.parquet"
)

BUILDING_METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "nrel_final_metrics_by_building.csv"
)

BUILDING_TYPE_METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "nrel_final_metrics_by_building_type.csv"
)

CLIMATE_ZONE_METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "nrel_final_metrics_by_climate_zone.csv"
)

SPLIT_COMPARISON_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "nrel_final_split_comparison.csv"
)

RESIDUAL_SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "nrel_final_residual_summary.csv"
)


BUILDING_ID_COLUMN = "bldg_id"
ROLE_COLUMN = "sample_role"
SPLIT_COLUMN = "split"

TARGET_COLUMN = "target_1h_ahead"
MODEL_PREDICTION_COLUMN = "extra_trees_prediction"
SEASONAL_PREDICTION_COLUMN = "seasonal_naive_24h"

BUILDING_TYPE_COLUMN = (
    "in.geometry_building_type_recs"
)

CLIMATE_ZONE_COLUMN = (
    "in.cec_climate_zone"
)


EXPECTED_TEMPORAL_BUILDINGS = 200
EXPECTED_HELD_OUT_BUILDINGS = 50


def load_predictions() -> pd.DataFrame:

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Final NREL predictions not found: "
            f"{PREDICTIONS_PATH}"
        )

    df = pd.read_parquet(
        PREDICTIONS_PATH
    )

    required_columns = {
        BUILDING_ID_COLUMN,
        ROLE_COLUMN,
        SPLIT_COLUMN,
        TARGET_COLUMN,
        MODEL_PREDICTION_COLUMN,
        SEASONAL_PREDICTION_COLUMN,
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Prediction dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return df


def load_building_metadata() -> pd.DataFrame:

    if not MODELING_BASE_PATH.exists():
        raise FileNotFoundError(
            f"Modeling base dataset not found: "
            f"{MODELING_BASE_PATH}"
        )

    metadata = pd.read_parquet(
        MODELING_BASE_PATH,
        columns=[
            BUILDING_ID_COLUMN,
            BUILDING_TYPE_COLUMN,
            CLIMATE_ZONE_COLUMN,
        ],
    )

    metadata = (
        metadata
        .drop_duplicates(
            subset=[
                BUILDING_ID_COLUMN,
            ]
        )
        .copy()
    )

    duplicate_count = int(
        metadata[
            BUILDING_ID_COLUMN
        ]
        .duplicated()
        .sum()
    )

    if duplicate_count != 0:
        raise ValueError(
            "Building metadata contains duplicate building IDs."
        )

    return metadata


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict:

    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    error = (
        actual
        - predicted
    )

    absolute_error = np.abs(
        error
    )

    mae = float(
        absolute_error.mean()
    )

    rmse = float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )

    mean_actual = float(
        actual.mean()
    )

    cvrmse = (
        float(
            rmse
            / mean_actual
            * 100
        )
        if mean_actual != 0
        else np.nan
    )

    nonzero_mask = (
        actual
        != 0
    )

    mape_n = int(
        nonzero_mask.sum()
    )

    mape = (
        float(
            np.mean(
                np.abs(
                    error[
                        nonzero_mask
                    ]
                    / actual[
                        nonzero_mask
                    ]
                )
            )
            * 100
        )
        if mape_n > 0
        else np.nan
    )

    bias = float(
        error.mean()
    )

    median_absolute_error = float(
        np.median(
            absolute_error
        )
    )

    p90_absolute_error = float(
        np.percentile(
            absolute_error,
            90,
        )
    )

    p95_absolute_error = float(
        np.percentile(
            absolute_error,
            95,
        )
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "cvrmse_pct": cvrmse,
        "mape_pct": mape,
        "mape_n": mape_n,
        "bias": bias,
        "median_absolute_error": (
            median_absolute_error
        ),
        "p90_absolute_error": (
            p90_absolute_error
        ),
        "p95_absolute_error": (
            p95_absolute_error
        ),
    }


def evaluate_group(
    group: pd.DataFrame,
) -> dict:

    actual = (
        group[
            TARGET_COLUMN
        ]
        .to_numpy(
            dtype=float
        )
    )

    model_predictions = (
        group[
            MODEL_PREDICTION_COLUMN
        ]
        .to_numpy(
            dtype=float
        )
    )

    seasonal_predictions = (
        group[
            SEASONAL_PREDICTION_COLUMN
        ]
        .to_numpy(
            dtype=float
        )
    )

    model_metrics = calculate_metrics(
        actual,
        model_predictions,
    )

    seasonal_metrics = calculate_metrics(
        actual,
        seasonal_predictions,
    )

    improvement = (
        (
            seasonal_metrics[
                "mae"
            ]
            - model_metrics[
                "mae"
            ]
        )
        / seasonal_metrics[
            "mae"
        ]
        * 100
        if seasonal_metrics[
            "mae"
        ]
        != 0
        else np.nan
    )

    return {
        "rows": len(
            group
        ),
        "target_mean": float(
            np.mean(
                actual
            )
        ),
        "target_median": float(
            np.median(
                actual
            )
        ),
        "extra_trees_mae": (
            model_metrics[
                "mae"
            ]
        ),
        "extra_trees_rmse": (
            model_metrics[
                "rmse"
            ]
        ),
        "extra_trees_cvrmse_pct": (
            model_metrics[
                "cvrmse_pct"
            ]
        ),
        "extra_trees_mape_pct": (
            model_metrics[
                "mape_pct"
            ]
        ),
        "extra_trees_bias": (
            model_metrics[
                "bias"
            ]
        ),
        "extra_trees_median_absolute_error": (
            model_metrics[
                "median_absolute_error"
            ]
        ),
        "extra_trees_p90_absolute_error": (
            model_metrics[
                "p90_absolute_error"
            ]
        ),
        "extra_trees_p95_absolute_error": (
            model_metrics[
                "p95_absolute_error"
            ]
        ),
        "seasonal_naive_mae": (
            seasonal_metrics[
                "mae"
            ]
        ),
        "seasonal_naive_rmse": (
            seasonal_metrics[
                "rmse"
            ]
        ),
        "seasonal_naive_cvrmse_pct": (
            seasonal_metrics[
                "cvrmse_pct"
            ]
        ),
        "seasonal_naive_mape_pct": (
            seasonal_metrics[
                "mape_pct"
            ]
        ),
        "mae_improvement_vs_seasonal_naive_pct": (
            improvement
        ),
    }


def create_building_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    grouped = df.groupby(
        [
            SPLIT_COLUMN,
            BUILDING_ID_COLUMN,
            ROLE_COLUMN,
        ],
        sort=True,
    )

    for (
        split_name,
        bldg_id,
        sample_role,
    ), group in grouped:

        metrics = evaluate_group(
            group
        )

        rows.append(
            {
                SPLIT_COLUMN: split_name,
                BUILDING_ID_COLUMN: bldg_id,
                ROLE_COLUMN: sample_role,
                **metrics,
            }
        )

    return pd.DataFrame(
        rows
    )


def create_group_metrics(
    df: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:

    rows = []

    grouped = df.groupby(
        [
            SPLIT_COLUMN,
            group_column,
        ],
        dropna=False,
        sort=True,
    )

    for (
        split_name,
        group_value,
    ), group in grouped:

        metrics = evaluate_group(
            group
        )

        rows.append(
            {
                SPLIT_COLUMN: split_name,
                group_column: group_value,
                "buildings": group[
                    BUILDING_ID_COLUMN
                ].nunique(),
                **metrics,
            }
        )

    return pd.DataFrame(
        rows
    )


def create_split_comparison(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for split_name, group in df.groupby(
        SPLIT_COLUMN,
        sort=True,
    ):

        metrics = evaluate_group(
            group
        )

        rows.append(
            {
                SPLIT_COLUMN: split_name,
                "buildings": group[
                    BUILDING_ID_COLUMN
                ].nunique(),
                **metrics,
            }
        )

    return pd.DataFrame(
        rows
    )


def create_residual_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:

    working = df.copy()

    working[
        "extra_trees_residual"
    ] = (
        working[
            TARGET_COLUMN
        ]
        - working[
            MODEL_PREDICTION_COLUMN
        ]
    )

    rows = []

    for split_name, group in working.groupby(
        SPLIT_COLUMN,
        sort=True,
    ):

        residuals = (
            group[
                "extra_trees_residual"
            ]
            .to_numpy(
                dtype=float
            )
        )

        rows.append(
            {
                SPLIT_COLUMN: split_name,
                "rows": len(
                    group
                ),
                "residual_mean": float(
                    residuals.mean()
                ),
                "residual_std": float(
                    residuals.std()
                ),
                "residual_min": float(
                    residuals.min()
                ),
                "residual_p05": float(
                    np.percentile(
                        residuals,
                        5,
                    )
                ),
                "residual_p25": float(
                    np.percentile(
                        residuals,
                        25,
                    )
                ),
                "residual_median": float(
                    np.median(
                        residuals
                    )
                ),
                "residual_p75": float(
                    np.percentile(
                        residuals,
                        75,
                    )
                ),
                "residual_p95": float(
                    np.percentile(
                        residuals,
                        95,
                    )
                ),
                "residual_max": float(
                    residuals.max()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def validate_predictions(
    df: pd.DataFrame,
) -> None:

    split_counts = (
        df[
            SPLIT_COLUMN
        ]
        .value_counts()
    )

    required_splits = {
        "temporal_test",
        "held_out_test",
    }

    present_splits = set(
        split_counts.index
    )

    if present_splits != required_splits:
        raise ValueError(
            f"Unexpected prediction splits: "
            f"{sorted(present_splits)}"
        )

    temporal_buildings = df.loc[
        df[
            SPLIT_COLUMN
        ]
        == "temporal_test",
        BUILDING_ID_COLUMN,
    ].nunique()

    held_out_buildings = df.loc[
        df[
            SPLIT_COLUMN
        ]
        == "held_out_test",
        BUILDING_ID_COLUMN,
    ].nunique()

    if (
        temporal_buildings
        != EXPECTED_TEMPORAL_BUILDINGS
    ):
        raise ValueError(
            f"Temporal test contains "
            f"{temporal_buildings} buildings; "
            f"expected "
            f"{EXPECTED_TEMPORAL_BUILDINGS}."
        )

    if (
        held_out_buildings
        != EXPECTED_HELD_OUT_BUILDINGS
    ):
        raise ValueError(
            f"Held-out test contains "
            f"{held_out_buildings} buildings; "
            f"expected "
            f"{EXPECTED_HELD_OUT_BUILDINGS}."
        )

    prediction_columns = [
        TARGET_COLUMN,
        MODEL_PREDICTION_COLUMN,
        SEASONAL_PREDICTION_COLUMN,
    ]

    missing_count = int(
        df[
            prediction_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    if missing_count != 0:
        raise ValueError(
            f"Prediction dataset contains "
            f"{missing_count} missing prediction values."
        )


def print_key_findings(
    building_metrics: pd.DataFrame,
    split_comparison: pd.DataFrame,
) -> None:

    print()

    print(
        "Overall split comparison:"
    )

    print(
        split_comparison.to_string(
            index=False
        )
    )

    print()

    for split_name in [
        "temporal_test",
        "held_out_test",
    ]:

        split_buildings = (
            building_metrics.loc[
                building_metrics[
                    SPLIT_COLUMN
                ]
                == split_name
            ]
            .copy()
        )

        split_buildings = (
            split_buildings
            .sort_values(
                "extra_trees_mae"
            )
        )

        best_building = (
            split_buildings.iloc[
                0
            ]
        )

        worst_building = (
            split_buildings.iloc[
                -1
            ]
        )

        improved_count = int(
            (
                split_buildings[
                    "mae_improvement_vs_seasonal_naive_pct"
                ]
                > 0
            )
            .sum()
        )

        total_buildings = len(
            split_buildings
        )

        print(
            f"{split_name}:"
        )

        print(
            f"  Buildings where Extra Trees beats "
            f"Seasonal Naive by MAE: "
            f"{improved_count}/{total_buildings}"
        )

        print(
            f"  Median building MAE: "
            f"{split_buildings['extra_trees_mae'].median():.6f}"
        )

        print(
            f"  Best building ID: "
            f"{int(best_building[BUILDING_ID_COLUMN])}, "
            f"MAE="
            f"{best_building['extra_trees_mae']:.6f}"
        )

        print(
            f"  Worst building ID: "
            f"{int(worst_building[BUILDING_ID_COLUMN])}, "
            f"MAE="
            f"{worst_building['extra_trees_mae']:.6f}"
        )

        print()


def main() -> None:

    print(
        "Loading final NREL test predictions..."
    )

    predictions = load_predictions()

    print(
        f"Prediction rows: "
        f"{len(predictions):,}"
    )

    print()

    print(
        "Validating prediction dataset..."
    )

    validate_predictions(
        predictions
    )

    print(
        "Loading building metadata..."
    )

    metadata = load_building_metadata()

    predictions = predictions.merge(
        metadata,
        on=BUILDING_ID_COLUMN,
        how="left",
        validate="many_to_one",
    )

    metadata_missing = int(
        predictions[
            [
                BUILDING_TYPE_COLUMN,
                CLIMATE_ZONE_COLUMN,
            ]
        ]
        .isna()
        .sum()
        .sum()
    )

    if metadata_missing != 0:
        raise ValueError(
            f"Metadata merge produced "
            f"{metadata_missing} missing values."
        )

    print()

    print(
        "Calculating per-building diagnostics..."
    )

    building_metrics = (
        create_building_metrics(
            predictions
        )
    )

    print(
        "Calculating building-type diagnostics..."
    )

    building_type_metrics = (
        create_group_metrics(
            predictions,
            BUILDING_TYPE_COLUMN,
        )
    )

    print(
        "Calculating climate-zone diagnostics..."
    )

    climate_zone_metrics = (
        create_group_metrics(
            predictions,
            CLIMATE_ZONE_COLUMN,
        )
    )

    print(
        "Calculating familiar-vs-held-out "
        "split comparison..."
    )

    split_comparison = (
        create_split_comparison(
            predictions
        )
    )

    print(
        "Calculating residual summaries..."
    )

    residual_summary = (
        create_residual_summary(
            predictions
        )
    )

    BUILDING_METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    building_metrics.to_csv(
        BUILDING_METRICS_PATH,
        index=False,
    )

    building_type_metrics.to_csv(
        BUILDING_TYPE_METRICS_PATH,
        index=False,
    )

    climate_zone_metrics.to_csv(
        CLIMATE_ZONE_METRICS_PATH,
        index=False,
    )

    split_comparison.to_csv(
        SPLIT_COMPARISON_PATH,
        index=False,
    )

    residual_summary.to_csv(
        RESIDUAL_SUMMARY_PATH,
        index=False,
    )

    print_key_findings(
        building_metrics,
        split_comparison,
    )

    print(
        "NREL final diagnostics complete."
    )

    print()

    print(
        f"Saved building metrics to: "
        f"{BUILDING_METRICS_PATH}"
    )

    print(
        f"Saved building-type metrics to: "
        f"{BUILDING_TYPE_METRICS_PATH}"
    )

    print(
        f"Saved climate-zone metrics to: "
        f"{CLIMATE_ZONE_METRICS_PATH}"
    )

    print(
        f"Saved split comparison to: "
        f"{SPLIT_COMPARISON_PATH}"
    )

    print(
        f"Saved residual summary to: "
        f"{RESIDUAL_SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()