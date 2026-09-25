from pathlib import Path
import json
import sys

import joblib
import pandas as pd

from typing import cast

from sklearn.base import RegressorMixin

from typing import Protocol, cast

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.model_selection import (
    GridSearchCV,
    TimeSeriesSplit,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class Predictor(Protocol):
    def predict(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        ...

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

UNTUNED_TREE_RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "uci_tree_validation_metrics.csv"
)

TUNED_RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "uci_tuned_validation_metrics.csv"
)

TUNING_SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "uci_tuning_summary.csv"
)

COMBINED_RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "uci_all_validation_metrics.csv"
)

PARAMETERS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "uci_best_parameters.json"
)

MODEL_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "models"
)

TARGET_COLUMN = "target_next_hour"

SEASONAL_NAIVE_COLUMN = "seasonal_naive_24h"

N_SPLITS = 5


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
            "No numeric predictor columns found."
        )

    return numeric_columns

def build_lasso_pipeline() -> Pipeline:

    return Pipeline(
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
            (
                "model",
                Lasso(
                    max_iter=20000,
                ),
            ),
        ]
    )


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


def get_model_searches():

    searches = {
        "Lasso": {
            "pipeline": build_lasso_pipeline(),
            "params": {
                "model__alpha": [
                    0.001,
                    0.01,
                    0.05,
                    0.1,
                    0.5,
                    1.0,
                ]
            },
        },

        "Random Forest": {
            "pipeline": build_tree_pipeline(
                RandomForestRegressor(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            ),
            "params": {
                "model__n_estimators": [
                    100,
                    300,
                ],
                "model__max_depth": [
                    None,
                    10,
                    20,
                ],
                "model__min_samples_leaf": [
                    1,
                    5,
                ],
                "model__max_features": [
                    "sqrt",
                    0.7,
                ],
            },
        },

        "Extra Trees": {
            "pipeline": build_tree_pipeline(
                ExtraTreesRegressor(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            ),
            "params": {
                "model__n_estimators": [
                    100,
                    300,
                ],
                "model__max_depth": [
                    None,
                    10,
                    20,
                ],
                "model__min_samples_leaf": [
                    1,
                    5,
                ],
            },
        },

        "Gradient Boosting": {
            "pipeline": build_tree_pipeline(
                GradientBoostingRegressor(
                    random_state=RANDOM_STATE,
                )
            ),
            "params": {
                "model__n_estimators": [
                    100,
                    200,
                ],
                "model__learning_rate": [
                    0.03,
                    0.1,
                ],
                "model__max_depth": [
                    2,
                    3,
                    5,
                ],
            },
        },
    }

    return searches

def tune_model(
    model_name: str,
    pipeline: Pipeline,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    time_series_cv: TimeSeriesSplit,
) -> GridSearchCV:

    print()
    print("=" * 60)
    print(f"Tuning {model_name}")
    print("=" * 60)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=time_series_cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )

    search.fit(
        X_train,
        y_train,
    )

    print(
        f"Best cross-validation MAE: "
        f"{-search.best_score_:.3f}"
    )

    print(
        f"Best parameters: "
        f"{search.best_params_}"
    )

    return search

def evaluate_tuned_model(
    model_name: str,
    search: GridSearchCV,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> dict:

    best_estimator = cast(
        Predictor,
        search.best_estimator_,
    )

    predictions = best_estimator.predict(
        X_validation
    )

    metrics = calculate_all_metrics(
        y_validation,
        predictions,
    )

    metrics["Model"] = (
        f"{model_name} Tuned"
    )

    return metrics

def save_best_estimator(
    model_name: str,
    search: GridSearchCV,
) -> None:

    MODEL_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    output_path = (
        MODEL_OUTPUT_DIR
        / f"uci_{safe_name}_tuned.joblib"
    )

    joblib.dump(
        search.best_estimator_,
        output_path,
    )

    print(
        f"Saved tuned estimator to: "
        f"{output_path}"
    )

def create_combined_results(
    tuned_results: pd.DataFrame,
) -> None:

    frames = []

    if BASELINE_RESULTS_PATH.exists():
        frames.append(
            pd.read_csv(
                BASELINE_RESULTS_PATH
            )
        )

    if UNTUNED_TREE_RESULTS_PATH.exists():
        frames.append(
            pd.read_csv(
                UNTUNED_TREE_RESULTS_PATH
            )
        )

    frames.append(
        tuned_results
    )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined.to_csv(
        COMBINED_RESULTS_PATH,
        index=False,
    )

    print()
    print(
        "Combined validation results"
    )

    print(
        "==========================="
    )

    print(
        combined.to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved combined results to: "
        f"{COMBINED_RESULTS_PATH}"
    )

def main() -> None:

    train = load_partition(
        TRAIN_PATH
    )

    validation = load_partition(
        VALIDATION_PATH
    )

    feature_columns = (
        get_feature_columns(
            train
        )
    )

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

    print(
        f"Training rows: {len(train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation):,}"
    )

    print(
        f"Predictor columns: "
        f"{len(feature_columns)}"
    )

    print(
        f"TimeSeriesSplit folds: "
        f"{N_SPLITS}"
    )

    time_series_cv = TimeSeriesSplit(
        n_splits=N_SPLITS
    )

    searches = get_model_searches()

    tuned_results = []

    tuning_summary = []

    best_parameters = {}

    for model_name, specification in searches.items():

        search = tune_model(
            model_name=model_name,
            pipeline=specification[
                "pipeline"
            ],
            param_grid=specification[
                "params"
            ],
            X_train=X_train,
            y_train=y_train,
            time_series_cv=time_series_cv,
        )

        result = evaluate_tuned_model(
            model_name=model_name,
            search=search,
            X_validation=X_validation,
            y_validation=y_validation,
        )

        tuned_results.append(
            result
        )

        tuning_summary.append(
            {
                "Model": model_name,
                "Best_CV_MAE": (
                    -search.best_score_
                ),
                "Number_of_Parameter_Combinations": (
                    len(
                        search.cv_results_[
                            "params"
                        ]
                    )
                ),
            }
        )

        best_parameters[
            model_name
        ] = search.best_params_

        save_best_estimator(
            model_name,
            search,
        )

    tuned_results_df = pd.DataFrame(
        tuned_results
    )

    tuned_results_df = tuned_results_df[
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
        "Tuned validation results"
    )

    print(
        "========================"
    )

    print(
        tuned_results_df.to_string(
            index=False
        )
    )

    TUNED_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tuned_results_df.to_csv(
        TUNED_RESULTS_PATH,
        index=False,
    )

    tuning_summary_df = pd.DataFrame(
        tuning_summary
    )

    tuning_summary_df.to_csv(
        TUNING_SUMMARY_PATH,
        index=False,
    )

    with open(
        PARAMETERS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            best_parameters,
            file,
            indent=4,
        )

    create_combined_results(
        tuned_results_df
    )

    print()
    print(
        "Tuning complete."
    )

    print(
        f"Tuned metrics: "
        f"{TUNED_RESULTS_PATH}"
    )

    print(
        f"Tuning summary: "
        f"{TUNING_SUMMARY_PATH}"
    )

    print(
        f"Best parameters: "
        f"{PARAMETERS_PATH}"
    )


if __name__ == "__main__":
    main()