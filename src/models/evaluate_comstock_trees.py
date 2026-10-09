from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comstock_ca_train.parquet"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "comstock_ca_validation.parquet"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "comstock_tree_validation_metrics.csv"
)

RF_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "comstock_random_forest_initial.joblib"
)

ET_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "comstock_extra_trees_initial.joblib"
)

HGB_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "comstock_hist_gradient_boosting_initial.joblib"
)


TARGET_COLUMN = "target_next_hour"

RANDOM_STATE = 42


NUMERIC_FEATURES = [
    "in.sqft",
    "in.number_of_stories",
    "stat.occupant_density_ppl_per_m_2",
    "Dry Bulb Temperature [°C]",
    "Relative Humidity [%]",
    "Wind Speed [m/s]",
    "Global Horizontal Radiation [W/m2]",
    "Direct Normal Radiation [W/m2]",
    "Diffuse Horizontal Radiation [W/m2]",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "heating_degree",
    "cooling_degree",
    "lag_1h",
    "lag_24h",
    "rolling_mean_24h",
    "rolling_mean_168h",
]

CATEGORICAL_FEATURES = [
    "in.building_type",
    "in.hvac_system_type",
    "in.heating_fuel",
    "in.service_water_heating_fuel",
    "in.climate_zone_ashrae_2004",
    "in.climate_zone_building_america",
]


def load_split(
    path: Path,
    split_name: str,
) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"{split_name} dataset not found: "
            f"{path}"
        )

    data = pd.read_parquet(
        path
    )

    required_columns = {
        TARGET_COLUMN,
        *NUMERIC_FEATURES,
        *CATEGORICAL_FEATURES,
    }

    missing_columns = (
        required_columns
        - set(
            data.columns
        )
    )

    if missing_columns:
        raise ValueError(
            f"{split_name} dataset is missing "
            f"required columns: "
            f"{sorted(missing_columns)}"
        )

    return data


def validate_split(
    data: pd.DataFrame,
    split_name: str,
) -> None:

    required_columns = [
        TARGET_COLUMN,
        *NUMERIC_FEATURES,
        *CATEGORICAL_FEATURES,
    ]

    missing_counts = (
        data[
            required_columns
        ]
        .isna()
        .sum()
    )

    total_missing = int(
        missing_counts.sum()
    )

    if total_missing != 0:

        print()

        print(
            f"{split_name} missing-value counts:"
        )

        print(
            missing_counts.loc[
                missing_counts > 0
            ].to_string()
        )

        raise ValueError(
            f"{split_name} contains "
            f"{total_missing:,} missing values "
            "in required modeling fields."
        )


def build_preprocessor() -> ColumnTransformer:

    categorical_transformer = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_transformer,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_random_forest() -> Pipeline:

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    max_features=0.7,
                    min_samples_leaf=1,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_extra_trees() -> Pipeline:

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_leaf=5,
                    max_features=0.7,
                    random_state=RANDOM_STATE,
                    n_jobs=2,
                ),
            ),
        ]
    )


def build_hist_gradient_boosting() -> Pipeline:

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=200,
                    learning_rate=0.1,
                    max_leaf_nodes=31,
                    min_samples_leaf=20,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float | int]:

    mae = float(
        mean_absolute_error(
            actual,
            predicted,
        )
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                actual,
                predicted,
            )
        )
    )

    actual_mean = float(
        np.mean(
            actual
        )
    )

    if actual_mean == 0.0:
        cvrmse = float(
            "nan"
        )
    else:
        cvrmse = (
            rmse
            / actual_mean
            * 100.0
        )

    nonzero_mask = (
        actual != 0.0
    )

    mape_n = int(
        nonzero_mask.sum()
    )

    if mape_n == 0:
        mape = float(
            "nan"
        )
    else:
        mape = float(
            np.mean(
                np.abs(
                    (
                        actual[
                            nonzero_mask
                        ]
                        - predicted[
                            nonzero_mask
                        ]
                    )
                    / actual[
                        nonzero_mask
                    ]
                )
            )
            * 100.0
        )

    return {
        "mae": mae,
        "rmse": rmse,
        "cvrmse_percent": cvrmse,
        "mape_percent": mape,
        "mape_n": mape_n,
    }


def load_existing_results() -> pd.DataFrame:

    if not RESULTS_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(
        RESULTS_PATH
    )


def result_exists(
    results: pd.DataFrame,
    model_name: str,
) -> bool:

    if results.empty:
        return False

    if "model" not in results.columns:
        return False

    return bool(
        (
            results[
                "model"
            ]
            == model_name
        ).any()
    )


def model_is_complete(
    results: pd.DataFrame,
    model_name: str,
    model_path: Path,
) -> bool:

    return (
        result_exists(
            results,
            model_name,
        )
        and model_path.exists()
    )


def add_result(
    results: pd.DataFrame,
    result: dict[str, object],
) -> pd.DataFrame:

    result_frame = pd.DataFrame(
        [
            result
        ]
    )

    if results.empty:
        combined = result_frame

    elif "model" not in results.columns:
        combined = result_frame

    else:
        existing_without_model = results.loc[
            results[
                "model"
            ]
            != result[
                "model"
            ]
        ].copy()

        combined = pd.concat(
            [
                existing_without_model,
                result_frame,
            ],
            ignore_index=True,
        )

    return combined


def save_results(
    results: pd.DataFrame,
) -> None:

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean_results = (
        results
        .drop_duplicates(
            subset=[
                "model",
            ],
            keep="last",
        )
        .sort_values(
            "mae"
        )
        .reset_index(
            drop=True
        )
    )

    clean_results.to_csv(
        RESULTS_PATH,
        index=False,
    )


def checkpoint_model(
    pipeline: Pipeline,
    model_path: Path,
    model_name: str,
) -> None:

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        model_path,
    )

    print(
        f"Checkpointed {model_name} model to:"
    )

    print(
        model_path
    )

    print()


def evaluate_model(
    model_name: str,
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_validation: pd.DataFrame,
    y_validation: np.ndarray,
) -> tuple[
    dict[str, object],
    Pipeline,
]:

    print(
        f"Training {model_name}..."
    )

    training_start = perf_counter()

    pipeline.fit(
        x_train,
        y_train,
    )

    training_seconds = (
        perf_counter()
        - training_start
    )

    print(
        f"Predicting validation set with "
        f"{model_name}..."
    )

    prediction_start = perf_counter()

    predictions = np.asarray(
        pipeline.predict(
            x_validation
        ),
        dtype=float,
    )

    prediction_seconds = (
        perf_counter()
        - prediction_start
    )

    metrics = calculate_metrics(
        y_validation,
        predictions,
    )

    result = {
        "model": model_name,
        "mae": metrics[
            "mae"
        ],
        "rmse": metrics[
            "rmse"
        ],
        "cvrmse_percent": metrics[
            "cvrmse_percent"
        ],
        "mape_percent": metrics[
            "mape_percent"
        ],
        "mape_n": metrics[
            "mape_n"
        ],
        "training_seconds": training_seconds,
        "prediction_seconds": prediction_seconds,
        "status": "completed",
    }

    print(
        f"  MAE: "
        f"{result['mae']:.6f}"
    )

    print(
        f"  RMSE: "
        f"{result['rmse']:.6f}"
    )

    print(
        f"  CVRMSE: "
        f"{result['cvrmse_percent']:.3f}%"
    )

    print(
        f"  MAPE: "
        f"{result['mape_percent']:.3f}%"
    )

    print(
        f"  MAPE n: "
        f"{result['mape_n']:,}"
    )

    print(
        f"  Training time: "
        f"{training_seconds:,.2f} seconds"
    )

    print(
        f"  Prediction time: "
        f"{prediction_seconds:,.2f} seconds"
    )

    print()

    return (
        result,
        pipeline,
    )


def print_existing_result(
    results: pd.DataFrame,
    model_name: str,
) -> None:

    row = results.loc[
        results[
            "model"
        ]
        == model_name
    ].iloc[
        -1
    ]

    print(
        f"{model_name} already has both a "
        "saved model and validation result."
    )

    print(
        "Skipping retraining."
    )

    print(
        f"  MAE: "
        f"{float(row['mae']):.6f}"
    )

    print(
        f"  RMSE: "
        f"{float(row['rmse']):.6f}"
    )

    print(
        f"  CVRMSE: "
        f"{float(row['cvrmse_percent']):.3f}%"
    )

    print(
        f"  MAPE: "
        f"{float(row['mape_percent']):.3f}%"
    )

    print()


def train_and_checkpoint(
    model_name: str,
    pipeline: Pipeline,
    model_path: Path,
    results: pd.DataFrame,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_validation: pd.DataFrame,
    y_validation: np.ndarray,
) -> pd.DataFrame:

    result, fitted_pipeline = evaluate_model(
        model_name=model_name,
        pipeline=pipeline,
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
    )

    checkpoint_model(
        fitted_pipeline,
        model_path,
        model_name,
    )

    updated_results = add_result(
        results,
        result,
    )

    save_results(
        updated_results
    )

    print(
        f"{model_name} metrics checkpointed."
    )

    print()

    return updated_results


def main() -> None:

    total_start = perf_counter()

    print(
        "Loading ComStock training split..."
    )

    train = load_split(
        TRAIN_PATH,
        "Training",
    )

    print(
        f"Training rows: "
        f"{len(train):,}"
    )

    print()

    print(
        "Loading ComStock validation split..."
    )

    validation = load_split(
        VALIDATION_PATH,
        "Validation",
    )

    print(
        f"Validation rows: "
        f"{len(validation):,}"
    )

    print()

    print(
        "Validating tree-model fields..."
    )

    validate_split(
        train,
        "Training",
    )

    validate_split(
        validation,
        "Validation",
    )

    predictor_columns = [
        *NUMERIC_FEATURES,
        *CATEGORICAL_FEATURES,
    ]

    x_train = train[
        predictor_columns
    ]

    y_train = train[
        TARGET_COLUMN
    ].to_numpy(
        dtype=float
    )

    x_validation = validation[
        predictor_columns
    ]

    y_validation = validation[
        TARGET_COLUMN
    ].to_numpy(
        dtype=float
    )

    print()

    print(
        f"Numeric predictors: "
        f"{len(NUMERIC_FEATURES):,}"
    )

    print(
        f"Categorical predictors: "
        f"{len(CATEGORICAL_FEATURES):,}"
    )

    print(
        f"Total raw predictors: "
        f"{len(predictor_columns):,}"
    )

    print()

    results = load_existing_results()

    if model_is_complete(
        results,
        "Random Forest",
        RF_MODEL_PATH,
    ):

        print_existing_result(
            results,
            "Random Forest",
        )

    else:

        if result_exists(
            results,
            "Random Forest",
        ):

            print(
                "A Random Forest metrics row exists, "
                "but no saved model checkpoint exists."
            )

            print(
                "Retraining Random Forest so the "
                "experiment is fully reproducible."
            )

            print()

        print(
            "Random Forest configuration:"
        )

        print(
            "  n_estimators: 200"
        )

        print(
            "  max_depth: None"
        )

        print(
            "  min_samples_leaf: 1"
        )

        print(
            "  max_features: 0.7"
        )

        print(
            "  n_jobs: -1"
        )

        print()

        results = train_and_checkpoint(
            model_name="Random Forest",
            pipeline=build_random_forest(),
            model_path=RF_MODEL_PATH,
            results=results,
            x_train=x_train,
            y_train=y_train,
            x_validation=x_validation,
            y_validation=y_validation,
        )

    if model_is_complete(
        results,
        "Extra Trees",
        ET_MODEL_PATH,
    ):

        print_existing_result(
            results,
            "Extra Trees",
        )

    else:

        if result_exists(
            results,
            "Extra Trees",
        ):

            print(
                "An Extra Trees metrics row exists, "
                "but no saved model checkpoint exists."
            )

            print(
                "Retraining Extra Trees."
            )

            print()

        print(
            "Extra Trees configuration:"
        )

        print(
            "  n_estimators: 200"
        )

        print(
            "  max_depth: 20"
        )

        print(
            "  min_samples_leaf: 5"
        )

        print(
            "  max_features: 0.7"
        )

        print(
            "  n_jobs: 2"
        )

        print()

        results = train_and_checkpoint(
            model_name="Extra Trees",
            pipeline=build_extra_trees(),
            model_path=ET_MODEL_PATH,
            results=results,
            x_train=x_train,
            y_train=y_train,
            x_validation=x_validation,
            y_validation=y_validation,
        )

    if model_is_complete(
        results,
        "HistGradientBoosting",
        HGB_MODEL_PATH,
    ):

        print_existing_result(
            results,
            "HistGradientBoosting",
        )

    else:

        if result_exists(
            results,
            "HistGradientBoosting",
        ):

            print(
                "A HistGradientBoosting metrics row "
                "exists, but no saved model checkpoint "
                "exists."
            )

            print(
                "Retraining HistGradientBoosting."
            )

            print()

        print(
            "HistGradientBoosting configuration:"
        )

        print(
            "  max_iter: 200"
        )

        print(
            "  learning_rate: 0.1"
        )

        print(
            "  max_leaf_nodes: 31"
        )

        print(
            "  min_samples_leaf: 20"
        )

        print()

        results = train_and_checkpoint(
            model_name="HistGradientBoosting",
            pipeline=build_hist_gradient_boosting(),
            model_path=HGB_MODEL_PATH,
            results=results,
            x_train=x_train,
            y_train=y_train,
            x_validation=x_validation,
            y_validation=y_validation,
        )

    final_results = pd.read_csv(
        RESULTS_PATH
    )

    final_results = (
        final_results
        .sort_values(
            "mae"
        )
        .reset_index(
            drop=True
        )
    )

    total_seconds = (
        perf_counter()
        - total_start
    )

    print(
        "ComStock initial tree-model "
        "evaluation complete."
    )

    print()

    print(
        final_results[
            [
                "model",
                "mae",
                "rmse",
                "cvrmse_percent",
                "mape_percent",
                "mape_n",
                "training_seconds",
                "prediction_seconds",
                "status",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    best_model = str(
        final_results.iloc[
            0
        ][
            "model"
        ]
    )

    best_mae = float(
        final_results.iloc[
            0
        ][
            "mae"
        ]
    )

    print(
        f"Lowest validation MAE: "
        f"{best_model} "
        f"({best_mae:.6f})"
    )

    print()

    print(
        f"Current-run elapsed time: "
        f"{total_seconds:,.2f} seconds"
    )

    print()

    print(
        f"Saved metrics to: "
        f"{RESULTS_PATH}"
    )

    print(
        f"Random Forest model: "
        f"{RF_MODEL_PATH}"
    )

    print(
        f"Extra Trees model: "
        f"{ET_MODEL_PATH}"
    )

    print(
        f"HistGradientBoosting model: "
        f"{HGB_MODEL_PATH}"
    )


if __name__ == "__main__":
    main()