from pathlib import Path
import time

import joblib
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
)

from src.evaluation.metrics import calculate_all_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bdg2_train.parquet"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bdg2_validation.parquet"
)

BASELINE_METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "bdg2_baseline_validation_metrics.csv"
)

TREE_METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "bdg2_tree_validation_metrics.csv"
)

COMBINED_METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "bdg2_model_validation_metrics.csv"
)

RF_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "bdg2_random_forest_initial.joblib"
)

ET_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "bdg2_extra_trees_initial.joblib"
)

TARGET_COLUMN = (
    "target_electricity_next_hour"
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

RANDOM_STATE = 42


def evaluate_predictions(
    model_name: str,
    y_true: pd.Series,
    y_pred,
    fit_seconds: float,
    predict_seconds: float,
) -> dict[str, object]:

    metrics = calculate_all_metrics(
        y_true,
        y_pred,
    )

    return {
        "model": model_name,
        **metrics,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
    }


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[
    RandomForestRegressor,
    dict[str, object],
]:

    print()
    print(
        "Training Random Forest..."
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=1,
        max_features=0.7,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    start = time.perf_counter()

    model.fit(
        X_train,
        y_train,
    )

    fit_seconds = (
        time.perf_counter()
        - start
    )

    print(
        f"Random Forest fit time: "
        f"{fit_seconds:.2f} seconds"
    )

    start = time.perf_counter()

    predictions = model.predict(
        X_validation
    )

    predict_seconds = (
        time.perf_counter()
        - start
    )

    result = evaluate_predictions(
        model_name="Random Forest",
        y_true=y_validation,
        y_pred=predictions,
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
    )

    print(
        result
    )

    return model, result


def train_extra_trees(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[
    ExtraTreesRegressor,
    dict[str, object],
]:

    print()
    print(
        "Training memory-conscious Extra Trees..."
    )

    model = ExtraTreesRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=5,
        max_features=0.7,
        random_state=RANDOM_STATE,
        n_jobs=2,
    )

    start = time.perf_counter()

    model.fit(
        X_train,
        y_train,
    )

    fit_seconds = (
        time.perf_counter()
        - start
    )

    print(
        f"Extra Trees fit time: "
        f"{fit_seconds:.2f} seconds"
    )

    start = time.perf_counter()

    predictions = model.predict(
        X_validation
    )

    predict_seconds = (
        time.perf_counter()
        - start
    )

    result = evaluate_predictions(
        model_name="Extra Trees",
        y_true=y_validation,
        y_pred=predictions,
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
    )

    print(
        result
    )

    return model, result


def main() -> None:

    print(
        "Loading BDG2 training and validation data..."
    )

    required_columns = [
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
    ]

    train = pd.read_parquet(
        TRAIN_PATH,
        columns=required_columns,
    )

    validation = pd.read_parquet(
        VALIDATION_PATH,
        columns=required_columns,
    )

    print(
        f"Training rows: "
        f"{len(train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation):,}"
    )

    missing_train = int(
        train[
            required_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    missing_validation = int(
        validation[
            required_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Missing required training values: "
        f"{missing_train:,}"
    )

    print(
        f"Missing required validation values: "
        f"{missing_validation:,}"
    )

    if missing_train > 0:
        raise ValueError(
            "Training data contain missing "
            "required values."
        )

    if missing_validation > 0:
        raise ValueError(
            "Validation data contain missing "
            "required values."
        )

    X_train = train[
        FEATURE_COLUMNS
    ].copy()

    y_train = train[
        TARGET_COLUMN
    ].copy()

    X_validation = validation[
        FEATURE_COLUMNS
    ].copy()

    y_validation = validation[
        TARGET_COLUMN
    ].copy()

    TREE_METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RF_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tree_results: list[
        dict[str, object]
    ] = []

    rf_model, rf_result = (
        train_random_forest(
            X_train,
            y_train,
            X_validation,
            y_validation,
        )
    )

    tree_results.append(
        rf_result
    )

    joblib.dump(
        rf_model,
        RF_MODEL_PATH,
    )

    pd.DataFrame(
        tree_results
    ).to_csv(
        TREE_METRICS_PATH,
        index=False,
    )

    print(
        f"Random Forest checkpoint saved: "
        f"{RF_MODEL_PATH}"
    )

    print(
        f"Metrics checkpoint saved: "
        f"{TREE_METRICS_PATH}"
    )

    et_model, et_result = (
        train_extra_trees(
            X_train,
            y_train,
            X_validation,
            y_validation,
        )
    )

    tree_results.append(
        et_result
    )

    joblib.dump(
        et_model,
        ET_MODEL_PATH,
    )

    tree_results_df = pd.DataFrame(
        tree_results
    )

    tree_results_df.to_csv(
        TREE_METRICS_PATH,
        index=False,
    )

    print(
        f"Extra Trees checkpoint saved: "
        f"{ET_MODEL_PATH}"
    )

    if BASELINE_METRICS_PATH.exists():

        baseline_results = pd.read_csv(
            BASELINE_METRICS_PATH
        )

        combined = pd.concat(
            [
                baseline_results,
                tree_results_df,
            ],
            ignore_index=True,
            sort=False,
        )

        combined.to_csv(
            COMBINED_METRICS_PATH,
            index=False,
        )

        print()
        print(
            "Combined baseline/tree results:"
        )

        print(
            combined.to_string(
                index=False
            )
        )

    else:

        print()
        print(
            "Baseline metrics file not found; "
            "combined table was not created."
        )

    print()
    print(
        "BDG2 tree evaluation complete."
    )

    print()
    print(
        "Tree validation results:"
    )

    print(
        tree_results_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Tree metrics: "
        f"{TREE_METRICS_PATH}"
    )

    print(
        f"Random Forest model: "
        f"{RF_MODEL_PATH}"
    )

    print(
        f"Extra Trees model: "
        f"{ET_MODEL_PATH}"
    )

    if BASELINE_METRICS_PATH.exists():
        print(
            f"Combined metrics: "
            f"{COMBINED_METRICS_PATH}"
        )


if __name__ == "__main__":
    main()