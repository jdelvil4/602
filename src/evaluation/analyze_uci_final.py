from pathlib import Path
import sys
from typing import cast

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "uci_final_test_predictions.csv"
)

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "uci_train.csv"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "uci_validation.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "uci_extra_trees_final.joblib"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
)

TABLE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
)

FEATURE_IMPORTANCE_PATH = (
    TABLE_DIR
    / "uci_feature_importance.csv"
)

TOP_ERRORS_PATH = (
    TABLE_DIR
    / "uci_top_test_errors.csv"
)

ACTUAL_VS_PREDICTED_PATH = (
    FIGURE_DIR
    / "uci_actual_vs_predicted.png"
)

RESIDUAL_TIME_PATH = (
    FIGURE_DIR
    / "uci_residuals_over_time.png"
)

RESIDUAL_DISTRIBUTION_PATH = (
    FIGURE_DIR
    / "uci_residual_distribution.png"
)

FEATURE_IMPORTANCE_FIGURE_PATH = (
    FIGURE_DIR
    / "uci_feature_importance_top20.png"
)


TARGET_COLUMN = "target_next_hour"

SEASONAL_NAIVE_COLUMN = "seasonal_naive_24h"

EXCLUDE_COLUMNS = {
    "date",
    TARGET_COLUMN,
    SEASONAL_NAIVE_COLUMN,
    "hour",
    "day_of_week",
    "month",
}


def load_predictions() -> pd.DataFrame:

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {PREDICTIONS_PATH}"
        )

    df = pd.read_csv(
        PREDICTIONS_PATH,
        parse_dates=["date"],
    )

    required_columns = {
        "date",
        "actual",
        "predicted",
        "residual",
        "absolute_error",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Predictions file is missing columns: "
            f"{sorted(missing_columns)}"
        )

    return (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )


def load_development_data() -> pd.DataFrame:

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_PATH}"
        )

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Validation file not found: {VALIDATION_PATH}"
        )

    train = pd.read_csv(
        TRAIN_PATH,
        parse_dates=["date"],
    )

    validation = pd.read_csv(
        VALIDATION_PATH,
        parse_dates=["date"],
    )

    development = pd.concat(
        [
            train,
            validation,
        ],
        ignore_index=True,
    )

    return (
        development
        .sort_values("date")
        .reset_index(drop=True)
    )


def get_feature_columns(
    df: pd.DataFrame,
) -> list[str]:

    candidate_columns = [
        column
        for column in df.columns
        if column not in EXCLUDE_COLUMNS
    ]

    numeric_columns = (
        df[candidate_columns]
        .select_dtypes(include="number")
        .columns
        .tolist()
    )

    if not numeric_columns:
        raise ValueError(
            "No numeric predictor columns found."
        )

    return numeric_columns


def load_final_model() -> Pipeline:

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Final model not found: {MODEL_PATH}"
        )

    loaded_model = joblib.load(
        MODEL_PATH
    )

    model = cast(
        Pipeline,
        loaded_model,
    )

    return model


def save_actual_vs_predicted_plot(
    predictions: pd.DataFrame,
) -> None:

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        predictions["date"],
        predictions["actual"],
        label="Actual",
    )

    plt.plot(
        predictions["date"],
        predictions["predicted"],
        label="Predicted",
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Appliance Energy Consumption"
    )

    plt.title(
        "UCI Final Test: Actual vs Predicted Consumption"
    )

    plt.legend()

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    plt.savefig(
        ACTUAL_VS_PREDICTED_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_residual_time_plot(
    predictions: pd.DataFrame,
) -> None:

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        predictions["date"],
        predictions["residual"],
    )

    plt.axhline(
        y=0,
        linestyle="--",
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Residual (Actual - Predicted)"
    )

    plt.title(
        "UCI Final Test Residuals Over Time"
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    plt.savefig(
        RESIDUAL_TIME_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_residual_distribution_plot(
    predictions: pd.DataFrame,
) -> None:

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        predictions["residual"],
        bins=30,
    )

    plt.axvline(
        x=0,
        linestyle="--",
    )

    plt.xlabel(
        "Residual (Actual - Predicted)"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "UCI Final Test Residual Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        RESIDUAL_DISTRIBUTION_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def create_feature_importance_table(
    pipeline: Pipeline,
    feature_columns: list[str],
) -> pd.DataFrame:

    estimator = cast(
        ExtraTreesRegressor,
        pipeline.named_steps["model"],
    )

    importances = (
        estimator.feature_importances_
    )

    if len(importances) != len(feature_columns):
        raise ValueError(
            "Feature-importance count does not match "
            "the number of predictor columns."
        )

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": importances,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    importance_df["rank"] = (
        importance_df.index
        + 1
    )

    importance_df = importance_df[
        [
            "rank",
            "feature",
            "importance",
        ]
    ]

    return importance_df


def save_feature_importance_plot(
    importance_df: pd.DataFrame,
) -> None:

    top_features = (
        importance_df
        .head(20)
        .sort_values(
            "importance",
            ascending=True,
        )
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        top_features["feature"],
        top_features["importance"],
    )

    plt.xlabel(
        "Feature Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Top 20 Extra Trees Feature Importances"
    )

    plt.tight_layout()

    plt.savefig(
        FEATURE_IMPORTANCE_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def create_top_errors_table(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    top_errors = (
        predictions
        .sort_values(
            "absolute_error",
            ascending=False,
        )
        .head(20)
        .copy()
    )

    top_errors.insert(
        0,
        "error_rank",
        range(
            1,
            len(top_errors) + 1,
        ),
    )

    return top_errors


def print_prediction_summary(
    predictions: pd.DataFrame,
) -> None:

    mean_actual = (
        predictions["actual"]
        .mean()
    )

    mean_predicted = (
        predictions["predicted"]
        .mean()
    )

    mean_residual = (
        predictions["residual"]
        .mean()
    )

    median_residual = (
        predictions["residual"]
        .median()
    )

    mean_absolute_error = (
        predictions["absolute_error"]
        .mean()
    )

    median_absolute_error = (
        predictions["absolute_error"]
        .median()
    )

    maximum_absolute_error = (
        predictions["absolute_error"]
        .max()
    )

    print()
    print(
        "Prediction diagnostics"
    )

    print(
        "======================"
    )

    print(
        f"Test observations: "
        f"{len(predictions):,}"
    )

    print(
        f"Mean actual: "
        f"{mean_actual:.3f}"
    )

    print(
        f"Mean predicted: "
        f"{mean_predicted:.3f}"
    )

    print(
        f"Mean residual: "
        f"{mean_residual:.3f}"
    )

    print(
        f"Median residual: "
        f"{median_residual:.3f}"
    )

    print(
        f"Mean absolute error: "
        f"{mean_absolute_error:.3f}"
    )

    print(
        f"Median absolute error: "
        f"{median_absolute_error:.3f}"
    )

    print(
        f"Maximum absolute error: "
        f"{maximum_absolute_error:.3f}"
    )


def main() -> None:

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions = load_predictions()

    development = load_development_data()

    feature_columns = get_feature_columns(
        development
    )

    final_model = load_final_model()

    print_prediction_summary(
        predictions
    )

    save_actual_vs_predicted_plot(
        predictions
    )

    save_residual_time_plot(
        predictions
    )

    save_residual_distribution_plot(
        predictions
    )

    importance_df = (
        create_feature_importance_table(
            final_model,
            feature_columns,
        )
    )

    importance_df.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False,
    )

    save_feature_importance_plot(
        importance_df
    )

    top_errors_df = (
        create_top_errors_table(
            predictions
        )
    )

    top_errors_df.to_csv(
        TOP_ERRORS_PATH,
        index=False,
    )

    print()
    print(
        "Top 10 feature importances"
    )

    print(
        "=========================="
    )

    print(
        importance_df
        .head(10)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Top 10 largest test errors"
    )

    print(
        "=========================="
    )

    print(
        top_errors_df[
            [
                "error_rank",
                "date",
                "actual",
                "predicted",
                "residual",
                "absolute_error",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Saved figures:"
    )

    print(
        ACTUAL_VS_PREDICTED_PATH
    )

    print(
        RESIDUAL_TIME_PATH
    )

    print(
        RESIDUAL_DISTRIBUTION_PATH
    )

    print(
        FEATURE_IMPORTANCE_FIGURE_PATH
    )

    print()
    print(
        "Saved tables:"
    )

    print(
        FEATURE_IMPORTANCE_PATH
    )

    print(
        TOP_ERRORS_PATH
    )


if __name__ == "__main__":
    main()