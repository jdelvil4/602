from pathlib import Path
import sys

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    LinearRegression,
    Lasso,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


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

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "uci_baseline_validation_metrics.csv"
)

TARGET_COLUMN = "target_next_hour"

SEASONAL_NAIVE_COLUMN = (
    "seasonal_naive_24h"
)

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

    df = pd.read_csv(
        path,
        parse_dates=["date"],
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

    if len(numeric_columns) == 0:
        raise ValueError(
            "No numeric regression features found."
        )

    return numeric_columns

def build_linear_pipeline(
    model,
    feature_columns: list[str],
) -> Pipeline:

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                feature_columns,
            )
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

def evaluate_seasonal_naive(
    validation: pd.DataFrame,
) -> dict:

    y_true = validation[
        TARGET_COLUMN
    ]

    y_pred = validation[
        SEASONAL_NAIVE_COLUMN
    ]

    metrics = calculate_all_metrics(
        y_true,
        y_pred,
    )

    metrics["Model"] = (
        "Seasonal Naive"
    )

    return metrics

def evaluate_regression_model(
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

    pipeline = build_linear_pipeline(
        model=model,
        feature_columns=feature_columns,
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

    print(
        f"Training rows: {len(train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation):,}"
    )

    print(
        f"Training period: "
        f"{train['date'].min()} "
        f"to {train['date'].max()}"
    )

    print(
        f"Validation period: "
        f"{validation['date'].min()} "
        f"to {validation['date'].max()}"
    )

    feature_columns = (
        get_feature_columns(train)
    )

    print()
    print(
        "Regression features used:"
    )

    for column in feature_columns:
        print(
            f"  - {column}"
        )

    print()
    print(
        f"Total regression features: "
        f"{len(feature_columns)}"
    )

    results = []

    results.append(
        evaluate_seasonal_naive(
            validation
        )
    )

    results.append(
        evaluate_regression_model(
            model_name="Linear Regression",
            model=LinearRegression(),
            train=train,
            validation=validation,
            feature_columns=feature_columns,
        )
    )

    results.append(
        evaluate_regression_model(
            model_name="Lasso",
            model=Lasso(
                alpha=0.1,
                max_iter=10000,
            ),
            train=train,
            validation=validation,
            feature_columns=feature_columns,
        )
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
        "Validation results"
    )
    print(
        "=================="
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "Saved validation metrics to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()