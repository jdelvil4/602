from pathlib import Path
import sys

import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.config import RANDOM_STATE
from src.evaluation.metrics import (
    calculate_all_metrics,
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

BASELINE_RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "uci_baseline_validation_metrics.csv"
)

TREE_RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "uci_tree_validation_metrics.csv"
)

COMBINED_RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "uci_model_validation_metrics.csv"
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

def load_partition(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"Partition not found: {path}"
        )

    return pd.read_csv(
        path,
        parse_dates=["date"],
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
            "No numeric features were found."
        )

    return numeric_columns

def build_tree_pipeline(
    model,
) -> Pipeline:

    return Pipeline(
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

def evaluate_model(
    model_name: str,
    model,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str],
) -> dict:

    X_train = train[
        feature_columns
    ]

    y_train = train[
        TARGET_COLUMN
    ]

    X_validation = validation[
        feature_columns
    ]

    y_validation = validation[
        TARGET_COLUMN
    ]

    pipeline = build_tree_pipeline(
        model
    )

    print()
    print(
        f"Training {model_name}..."
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    predictions = pipeline.predict(
        X_validation
    )

    metrics = calculate_all_metrics(
        y_validation,
        predictions,
    )

    metrics["Model"] = model_name

    return metrics


def main() -> None:

    train = load_partition(
        TRAIN_PATH
    )

    validation = load_partition(
        VALIDATION_PATH
    )

    feature_columns = (
        get_feature_columns(train)
    )

    print(
        f"Training rows: {len(train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation):,}"
    )

    print(
        f"Tree-model features: "
        f"{len(feature_columns)}"
    )

    models = [
        (
            "Random Forest",
            RandomForestRegressor(
                n_estimators=200,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "Extra Trees",
            ExtraTreesRegressor(
                n_estimators=200,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "Gradient Boosting",
            GradientBoostingRegressor(
                random_state=RANDOM_STATE,
            ),
        ),
    ]

    results = []

    for model_name, model in models:

        result = evaluate_model(
            model_name=model_name,
            model=model,
            train=train,
            validation=validation,
            feature_columns=feature_columns,
        )

        results.append(
            result
        )

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df[
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
        "Tree-model validation results"
    )

    print(
        "============================="
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    TREE_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        TREE_RESULTS_PATH,
        index=False,
    )

    if BASELINE_RESULTS_PATH.exists():

        baseline_df = pd.read_csv(
            BASELINE_RESULTS_PATH
        )

        combined_df = pd.concat(
            [
                baseline_df,
                results_df,
            ],
            ignore_index=True,
        )

        combined_df.to_csv(
            COMBINED_RESULTS_PATH,
            index=False,
        )

        print()
        print(
            "All validation results"
        )

        print(
            "======================"
        )

        print(
            combined_df.to_string(
                index=False
            )
        )

        print()
        print(
            "Saved combined metrics to:"
        )

        print(
            COMBINED_RESULTS_PATH
        )

    else:

        print()
        print(
            "Baseline metrics file was not found."
        )

        print(
            "Tree-model metrics were still saved."
        )

    print()
    print(
        "Saved tree-model metrics to:"
    )

    print(
        TREE_RESULTS_PATH
    )


if __name__ == "__main__":
    main()