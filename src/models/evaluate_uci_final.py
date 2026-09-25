from pathlib import Path
import sys

import joblib
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.config import RANDOM_STATE
from src.evaluation.metrics import calculate_all_metrics


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

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "uci_test.csv"
)

METRICS_OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "uci_final_test_metrics.csv"
)

PREDICTIONS_OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "uci_final_test_predictions.csv"
)

MODEL_OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "uci_extra_trees_final.joblib"
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


FINAL_MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 10,
    "min_samples_leaf": 5,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def load_partition(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"Partition not found: {path}"
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
            "No numeric predictors found."
        )

    return numeric_columns


def build_final_model() -> Pipeline:

    model = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline


def validate_partitions(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:

    if train.empty:
        raise ValueError(
            "Training partition is empty."
        )

    if validation.empty:
        raise ValueError(
            "Validation partition is empty."
        )

    if test.empty:
        raise ValueError(
            "Test partition is empty."
        )

    if not train["date"].is_monotonic_increasing:
        raise ValueError(
            "Training partition is not chronological."
        )

    if not validation["date"].is_monotonic_increasing:
        raise ValueError(
            "Validation partition is not chronological."
        )

    if not test["date"].is_monotonic_increasing:
        raise ValueError(
            "Test partition is not chronological."
        )

    if (
        train["date"].max()
        >= validation["date"].min()
    ):
        raise ValueError(
            "Training and validation periods overlap."
        )

    if (
        validation["date"].max()
        >= test["date"].min()
    ):
        raise ValueError(
            "Validation and test periods overlap."
        )


def main() -> None:

    train = load_partition(
        TRAIN_PATH
    )

    validation = load_partition(
        VALIDATION_PATH
    )

    test = load_partition(
        TEST_PATH
    )

    validate_partitions(
        train=train,
        validation=validation,
        test=test,
    )

    development = pd.concat(
        [
            train,
            validation,
        ],
        ignore_index=True,
    )

    development = (
        development
        .sort_values("date")
        .reset_index(drop=True)
    )

    feature_columns = get_feature_columns(
        development
    )

    missing_test_features = [
        column
        for column in feature_columns
        if column not in test.columns
    ]

    if missing_test_features:
        raise ValueError(
            "Test partition is missing predictors: "
            f"{missing_test_features}"
        )

    X_development = development[
        feature_columns
    ]

    y_development = development[
        TARGET_COLUMN
    ]

    X_test = test[
        feature_columns
    ]

    y_test = test[
        TARGET_COLUMN
    ]

    print(
        f"Development rows: "
        f"{len(development):,}"
    )

    print(
        f"Test rows: "
        f"{len(test):,}"
    )

    print(
        f"Development period: "
        f"{development['date'].min()} "
        f"to {development['date'].max()}"
    )

    print(
        f"Test period: "
        f"{test['date'].min()} "
        f"to {test['date'].max()}"
    )

    print(
        f"Predictor columns: "
        f"{len(feature_columns)}"
    )

    print()

    print(
        "Locked final model:"
    )

    print(
        "Extra Trees Regressor"
    )

    print(
        FINAL_MODEL_PARAMS
    )

    model = build_final_model()

    model.fit(
        X_development,
        y_development,
    )

    predictions = model.predict(
        X_test
    )

    metrics = calculate_all_metrics(
        y_test,
        predictions,
    )

    metrics["Model"] = (
        "Extra Trees Final"
    )

    metrics_df = pd.DataFrame(
        [metrics]
    )

    metrics_df = metrics_df[
        [
            "Model",
            "MAE",
            "CVRMSE_percent",
            "MAPE_percent",
            "MAPE_n",
        ]
    ]

    print()

    print(
        "Final test results"
    )

    print(
        "=================="
    )

    print(
        metrics_df.to_string(
            index=False
        )
    )

    predictions_df = pd.DataFrame(
        {
            "date": test["date"],
            "actual": y_test,
            "predicted": predictions,
        }
    )

    predictions_df["residual"] = (
        predictions_df["actual"]
        - predictions_df["predicted"]
    )

    predictions_df["absolute_error"] = (
        predictions_df["residual"]
        .abs()
    )

    predictions_df["squared_error"] = (
        predictions_df["residual"]
        ** 2
    )

    METRICS_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_df.to_csv(
        METRICS_OUTPUT_PATH,
        index=False,
    )

    predictions_df.to_csv(
        PREDICTIONS_OUTPUT_PATH,
        index=False,
    )

    joblib.dump(
        model,
        MODEL_OUTPUT_PATH,
    )

    print()

    print(
        "Saved final metrics to:"
    )

    print(
        METRICS_OUTPUT_PATH
    )

    print()

    print(
        "Saved test predictions to:"
    )

    print(
        PREDICTIONS_OUTPUT_PATH
    )

    print()

    print(
        "Saved final fitted model to:"
    )

    print(
        MODEL_OUTPUT_PATH
    )


if __name__ == "__main__":
    main()