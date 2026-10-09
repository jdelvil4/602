from pathlib import Path
from time import perf_counter
from typing import cast

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import TimeSeriesSplit
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

CV_RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "comstock_tree_tuning_cv_results.csv"
)

VALIDATION_RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "comstock_tree_tuned_validation_metrics.csv"
)

BEST_PARAMS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "comstock_tree_tuned_best_params.csv"
)

RF_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "comstock_random_forest_tuned.joblib"
)

ET_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "comstock_extra_trees_tuned.joblib"
)


BUILDING_ID_COLUMN = "bldg_id"
TARGET_TIMESTAMP_COLUMN = "target_timestamp"
TARGET_COLUMN = "target_next_hour"

RANDOM_STATE = 42

CV_SPLITS = 3

TUNING_N_ESTIMATORS = 100
FINAL_N_ESTIMATORS = 200


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


RF_CANDIDATES = [
    {
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": 0.7,
    },
    {
        "max_depth": 20,
        "min_samples_leaf": 5,
        "max_features": 0.7,
    },
]

ET_CANDIDATES = [
    {
        "max_depth": 20,
        "min_samples_leaf": 5,
        "max_features": 0.7,
    },
    {
        "max_depth": 20,
        "min_samples_leaf": 10,
        "max_features": 0.7,
    },
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
        BUILDING_ID_COLUMN,
        TARGET_TIMESTAMP_COLUMN,
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

    data[
        TARGET_TIMESTAMP_COLUMN
    ] = pd.to_datetime(
        data[
            TARGET_TIMESTAMP_COLUMN
        ],
        errors="raise",
    )

    return data


def validate_split(
    data: pd.DataFrame,
    split_name: str,
) -> None:

    modeling_columns = [
        TARGET_COLUMN,
        *NUMERIC_FEATURES,
        *CATEGORICAL_FEATURES,
    ]

    missing_counts = (
        data[
            modeling_columns
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
            f"{split_name} missing values:"
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

    duplicate_count = int(
        data.duplicated(
            subset=[
                BUILDING_ID_COLUMN,
                TARGET_TIMESTAMP_COLUMN,
            ]
        ).sum()
    )

    if duplicate_count != 0:
        raise ValueError(
            f"{split_name} contains "
            f"{duplicate_count:,} duplicate "
            "building-target-timestamp rows."
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


def build_random_forest(
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    max_features: float,
) -> Pipeline:

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_extra_trees(
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    max_features: float,
) -> Pipeline:

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=RANDOM_STATE,
                    n_jobs=2,
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


def get_timestamp_folds(
    train: pd.DataFrame,
) -> list[
    tuple[
        pd.DatetimeIndex,
        pd.DatetimeIndex,
    ]
]:

    unique_timestamps = pd.DatetimeIndex(
        train[
            TARGET_TIMESTAMP_COLUMN
        ]
        .drop_duplicates()
        .sort_values()
    )

    timestamp_positions = np.arange(
        len(
            unique_timestamps
        )
    )

    splitter = TimeSeriesSplit(
        n_splits=CV_SPLITS
    )

    folds: list[
        tuple[
            pd.DatetimeIndex,
            pd.DatetimeIndex,
        ]
    ] = []

    for (
        train_indices,
        validation_indices,
    ) in splitter.split(
        timestamp_positions
    ):

        fold_train_timestamps = (
            unique_timestamps[
                train_indices
            ]
        )

        fold_validation_timestamps = (
            unique_timestamps[
                validation_indices
            ]
        )

        folds.append(
            (
                fold_train_timestamps,
                fold_validation_timestamps,
            )
        )

    return folds


def create_pipeline(
    model_name: str,
    params: dict[str, object],
    n_estimators: int,
) -> Pipeline:

    max_depth_value = params[
        "max_depth"
    ]

    if max_depth_value is None:
        max_depth: int | None = None

    else:
        max_depth = cast(
            int,
            max_depth_value,
        )

    min_samples_leaf = cast(
        int,
        params[
            "min_samples_leaf"
        ],
    )

    max_features = cast(
        float,
        params[
            "max_features"
        ],
    )

    if model_name == "Random Forest":

        return build_random_forest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
        )

    if model_name == "Extra Trees":

        return build_extra_trees(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
        )

    raise ValueError(
        f"Unsupported model: "
        f"{model_name}"
    )


def tune_candidate(
    model_name: str,
    candidate_number: int,
    params: dict[str, object],
    train: pd.DataFrame,
    folds: list[
        tuple[
            pd.DatetimeIndex,
            pd.DatetimeIndex,
        ]
    ],
) -> list[dict[str, object]]:

    predictor_columns = [
        *NUMERIC_FEATURES,
        *CATEGORICAL_FEATURES,
    ]

    candidate_results: list[
        dict[str, object]
    ] = []

    print(
        f"{model_name} candidate "
        f"{candidate_number}:"
    )

    print(
        f"  max_depth: "
        f"{params['max_depth']}"
    )

    print(
        f"  min_samples_leaf: "
        f"{params['min_samples_leaf']}"
    )

    print(
        f"  max_features: "
        f"{params['max_features']}"
    )

    print(
        f"  tuning trees: "
        f"{TUNING_N_ESTIMATORS}"
    )

    print()

    for fold_number, (
        fold_train_timestamps,
        fold_validation_timestamps,
    ) in enumerate(
        folds,
        start=1,
    ):

        fold_train = train.loc[
            train[
                TARGET_TIMESTAMP_COLUMN
            ].isin(
                fold_train_timestamps
            )
        ]

        fold_validation = train.loc[
            train[
                TARGET_TIMESTAMP_COLUMN
            ].isin(
                fold_validation_timestamps
            )
        ]

        x_train = fold_train[
            predictor_columns
        ]

        y_train = fold_train[
            TARGET_COLUMN
        ].to_numpy(
            dtype=float
        )

        x_validation = fold_validation[
            predictor_columns
        ]

        y_validation = fold_validation[
            TARGET_COLUMN
        ].to_numpy(
            dtype=float
        )

        pipeline = create_pipeline(
            model_name=model_name,
            params=params,
            n_estimators=TUNING_N_ESTIMATORS,
        )

        print(
            f"  Fold "
            f"{fold_number}/{CV_SPLITS}"
        )

        print(
            f"    Training rows: "
            f"{len(fold_train):,}"
        )

        print(
            f"    Validation rows: "
            f"{len(fold_validation):,}"
        )

        start = perf_counter()

        pipeline.fit(
            x_train,
            y_train,
        )

        fit_seconds = (
            perf_counter()
            - start
        )

        start = perf_counter()

        predictions = np.asarray(
            pipeline.predict(
                x_validation
            ),
            dtype=float,
        )

        prediction_seconds = (
            perf_counter()
            - start
        )

        metrics = calculate_metrics(
            y_validation,
            predictions,
        )

        result = {
            "model": model_name,
            "candidate": candidate_number,
            "fold": fold_number,
            "n_estimators": TUNING_N_ESTIMATORS,
            "max_depth": params[
                "max_depth"
            ],
            "min_samples_leaf": params[
                "min_samples_leaf"
            ],
            "max_features": params[
                "max_features"
            ],
            "train_rows": len(
                fold_train
            ),
            "validation_rows": len(
                fold_validation
            ),
            "train_first_target": str(
                fold_train[
                    TARGET_TIMESTAMP_COLUMN
                ].min()
            ),
            "train_last_target": str(
                fold_train[
                    TARGET_TIMESTAMP_COLUMN
                ].max()
            ),
            "validation_first_target": str(
                fold_validation[
                    TARGET_TIMESTAMP_COLUMN
                ].min()
            ),
            "validation_last_target": str(
                fold_validation[
                    TARGET_TIMESTAMP_COLUMN
                ].max()
            ),
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
            "fit_seconds": fit_seconds,
            "prediction_seconds": prediction_seconds,
        }

        candidate_results.append(
            result
        )

        print(
            f"    MAE: "
            f"{result['mae']:.6f}"
        )

        print(
            f"    CVRMSE: "
            f"{result['cvrmse_percent']:.3f}%"
        )

        print(
            f"    MAPE: "
            f"{result['mape_percent']:.3f}%"
        )

        print(
            f"    Fit time: "
            f"{fit_seconds:,.2f} seconds"
        )

        print()

        del pipeline
        del fold_train
        del fold_validation
        del x_train
        del x_validation
        del y_train
        del y_validation
        del predictions

    return candidate_results


def summarize_candidates(
    cv_results: pd.DataFrame,
) -> pd.DataFrame:

    summary = (
        cv_results
        .groupby(
            [
                "model",
                "candidate",
                "n_estimators",
                "max_depth",
                "min_samples_leaf",
                "max_features",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            mean_cv_mae=(
                "mae",
                "mean",
            ),
            std_cv_mae=(
                "mae",
                "std",
            ),
            mean_cv_rmse=(
                "rmse",
                "mean",
            ),
            mean_cv_cvrmse_percent=(
                "cvrmse_percent",
                "mean",
            ),
            mean_cv_mape_percent=(
                "mape_percent",
                "mean",
            ),
            total_fit_seconds=(
                "fit_seconds",
                "sum",
            ),
        )
    )

    return (
        summary
        .sort_values(
            [
                "model",
                "mean_cv_mae",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def get_best_candidate(
    summary: pd.DataFrame,
    model_name: str,
) -> dict[str, object]:

    model_summary = (
        summary.loc[
            summary[
                "model"
            ]
            == model_name
        ]
        .sort_values(
            "mean_cv_mae"
        )
    )

    if model_summary.empty:
        raise ValueError(
            f"No tuning results found for "
            f"{model_name}."
        )

    best = model_summary.iloc[
        0
    ]

    candidate_value = cast(
        int | np.integer,
        best[
            "candidate"
        ],
    )

    max_depth_value = best[
        "max_depth"
    ]

    min_samples_leaf_value = cast(
        int | np.integer,
        best[
            "min_samples_leaf"
        ],
    )

    max_features_value = cast(
        float | np.floating,
        best[
            "max_features"
        ],
    )

    mean_cv_mae_value = cast(
        float | np.floating,
        best[
            "mean_cv_mae"
        ],
    )

    std_cv_mae_value = cast(
        float | np.floating,
        best[
            "std_cv_mae"
        ],
    )

    if pd.isna(
        max_depth_value
    ):
        max_depth: int | None = None

    else:
        max_depth_numeric = cast(
            float
            | np.floating
            | int
            | np.integer,
            max_depth_value,
        )

        max_depth = int(
            max_depth_numeric
        )

    return {
        "model": model_name,
        "candidate": int(
            candidate_value
        ),
        "max_depth": max_depth,
        "min_samples_leaf": int(
            min_samples_leaf_value
        ),
        "max_features": float(
            max_features_value
        ),
        "mean_cv_mae": float(
            mean_cv_mae_value
        ),
        "std_cv_mae": float(
            std_cv_mae_value
        ),
    }


def refit_and_validate(
    model_name: str,
    best_params: dict[str, object],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    model_path: Path,
) -> dict[str, object]:

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

    params = {
        "max_depth": best_params[
            "max_depth"
        ],
        "min_samples_leaf": best_params[
            "min_samples_leaf"
        ],
        "max_features": best_params[
            "max_features"
        ],
    }

    pipeline = create_pipeline(
        model_name=model_name,
        params=params,
        n_estimators=FINAL_N_ESTIMATORS,
    )

    print(
        f"Refitting tuned "
        f"{model_name} "
        f"on full training set..."
    )

    print(
        f"  n_estimators: "
        f"{FINAL_N_ESTIMATORS}"
    )

    print(
        f"  max_depth: "
        f"{params['max_depth']}"
    )

    print(
        f"  min_samples_leaf: "
        f"{params['min_samples_leaf']}"
    )

    print(
        f"  max_features: "
        f"{params['max_features']}"
    )

    start = perf_counter()

    pipeline.fit(
        x_train,
        y_train,
    )

    fit_seconds = (
        perf_counter()
        - start
    )

    print(
        f"Predicting untouched "
        f"validation set with "
        f"tuned {model_name}..."
    )

    start = perf_counter()

    predictions = np.asarray(
        pipeline.predict(
            x_validation
        ),
        dtype=float,
    )

    prediction_seconds = (
        perf_counter()
        - start
    )

    metrics = calculate_metrics(
        y_validation,
        predictions,
    )

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        model_path,
    )

    result = {
        "model": model_name,
        "n_estimators": FINAL_N_ESTIMATORS,
        "max_depth": params[
            "max_depth"
        ],
        "min_samples_leaf": params[
            "min_samples_leaf"
        ],
        "max_features": params[
            "max_features"
        ],
        "mean_cv_mae": best_params[
            "mean_cv_mae"
        ],
        "std_cv_mae": best_params[
            "std_cv_mae"
        ],
        "validation_mae": metrics[
            "mae"
        ],
        "validation_rmse": metrics[
            "rmse"
        ],
        "validation_cvrmse_percent": metrics[
            "cvrmse_percent"
        ],
        "validation_mape_percent": metrics[
            "mape_percent"
        ],
        "validation_mape_n": metrics[
            "mape_n"
        ],
        "fit_seconds": fit_seconds,
        "prediction_seconds": prediction_seconds,
    }

    print(
        f"  Validation MAE: "
        f"{result['validation_mae']:.6f}"
    )

    print(
        f"  Validation RMSE: "
        f"{result['validation_rmse']:.6f}"
    )

    print(
        f"  Validation CVRMSE: "
        f"{result['validation_cvrmse_percent']:.3f}%"
    )

    print(
        f"  Validation MAPE: "
        f"{result['validation_mape_percent']:.3f}%"
    )

    print(
        f"  Fit time: "
        f"{fit_seconds:,.2f} seconds"
    )

    print(
        f"  Saved model to: "
        f"{model_path}"
    )

    print()

    return result


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

    print(
        f"Training buildings: "
        f"{train[BUILDING_ID_COLUMN].nunique():,}"
    )

    print()

    print(
        "Loading untouched validation split..."
    )

    validation = load_split(
        VALIDATION_PATH,
        "Validation",
    )

    print(
        f"Validation rows: "
        f"{len(validation):,}"
    )

    print(
        f"Validation buildings: "
        f"{validation[BUILDING_ID_COLUMN].nunique():,}"
    )

    print()

    print(
        "Validating modeling fields..."
    )

    validate_split(
        train,
        "Training",
    )

    validate_split(
        validation,
        "Validation",
    )

    print(
        "Creating expanding timestamp-aware "
        "cross-validation folds..."
    )

    folds = get_timestamp_folds(
        train
    )

    for fold_number, (
        fold_train_timestamps,
        fold_validation_timestamps,
    ) in enumerate(
        folds,
        start=1,
    ):

        print(
            f"Fold {fold_number}:"
        )

        print(
            f"  Train target timestamps: "
            f"{len(fold_train_timestamps):,}"
        )

        print(
            f"  Validation target timestamps: "
            f"{len(fold_validation_timestamps):,}"
        )

        print(
            f"  Train period: "
            f"{fold_train_timestamps.min()} "
            f"to "
            f"{fold_train_timestamps.max()}"
        )

        print(
            f"  Validation period: "
            f"{fold_validation_timestamps.min()} "
            f"to "
            f"{fold_validation_timestamps.max()}"
        )

        print()

    all_cv_results: list[
        dict[str, object]
    ] = []

    CV_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Tuning Random Forest..."
    )

    print()

    for candidate_number, params in enumerate(
        RF_CANDIDATES,
        start=1,
    ):

        candidate_results = tune_candidate(
            model_name="Random Forest",
            candidate_number=candidate_number,
            params=params,
            train=train,
            folds=folds,
        )

        all_cv_results.extend(
            candidate_results
        )

        pd.DataFrame(
            all_cv_results
        ).to_csv(
            CV_RESULTS_PATH,
            index=False,
        )

        print(
            "Checkpointed current CV results."
        )

        print()

    print(
        "Tuning Extra Trees..."
    )

    print()

    for candidate_number, params in enumerate(
        ET_CANDIDATES,
        start=1,
    ):

        candidate_results = tune_candidate(
            model_name="Extra Trees",
            candidate_number=candidate_number,
            params=params,
            train=train,
            folds=folds,
        )

        all_cv_results.extend(
            candidate_results
        )

        pd.DataFrame(
            all_cv_results
        ).to_csv(
            CV_RESULTS_PATH,
            index=False,
        )

        print(
            "Checkpointed current CV results."
        )

        print()

    cv_results = pd.DataFrame(
        all_cv_results
    )

    summary = summarize_candidates(
        cv_results
    )

    print(
        "Cross-validation candidate summary:"
    )

    print()

    print(
        summary[
            [
                "model",
                "candidate",
                "max_depth",
                "min_samples_leaf",
                "max_features",
                "mean_cv_mae",
                "std_cv_mae",
                "mean_cv_cvrmse_percent",
                "mean_cv_mape_percent",
                "total_fit_seconds",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    best_rf = get_best_candidate(
        summary,
        "Random Forest",
    )

    best_et = get_best_candidate(
        summary,
        "Extra Trees",
    )

    print(
        "Best Random Forest candidate:"
    )

    print(
        best_rf
    )

    print()

    print(
        "Best Extra Trees candidate:"
    )

    print(
        best_et
    )

    print()

    rf_validation_result = (
        refit_and_validate(
            model_name="Random Forest",
            best_params=best_rf,
            train=train,
            validation=validation,
            model_path=RF_MODEL_PATH,
        )
    )

    et_validation_result = (
        refit_and_validate(
            model_name="Extra Trees",
            best_params=best_et,
            train=train,
            validation=validation,
            model_path=ET_MODEL_PATH,
        )
    )

    validation_results = pd.DataFrame(
        [
            rf_validation_result,
            et_validation_result,
        ]
    )

    validation_results = (
        validation_results
        .sort_values(
            "validation_mae"
        )
        .reset_index(
            drop=True
        )
    )

    best_params_table = pd.DataFrame(
        [
            best_rf,
            best_et,
        ]
    )

    VALIDATION_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_results.to_csv(
        VALIDATION_RESULTS_PATH,
        index=False,
    )

    best_params_table.to_csv(
        BEST_PARAMS_PATH,
        index=False,
    )

    total_seconds = (
        perf_counter()
        - total_start
    )

    print(
        "ComStock tree tuning complete."
    )

    print()

    print(
        "Final tuned validation results:"
    )

    print()

    print(
        validation_results[
            [
                "model",
                "n_estimators",
                "max_depth",
                "min_samples_leaf",
                "max_features",
                "mean_cv_mae",
                "validation_mae",
                "validation_rmse",
                "validation_cvrmse_percent",
                "validation_mape_percent",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    winner = str(
        validation_results.iloc[
            0
        ][
            "model"
        ]
    )

    winner_mae_value = cast(
        float | np.floating,
        validation_results.iloc[
            0
        ][
            "validation_mae"
        ],
    )

    winner_mae = float(
        winner_mae_value
    )

    print(
        f"Lowest tuned validation MAE: "
        f"{winner} "
        f"({winner_mae:.6f})"
    )

    print()

    print(
        f"Total elapsed time: "
        f"{total_seconds:,.2f} seconds"
    )

    print()

    print(
        f"Saved fold-level CV results to: "
        f"{CV_RESULTS_PATH}"
    )

    print(
        f"Saved best parameters to: "
        f"{BEST_PARAMS_PATH}"
    )

    print(
        f"Saved tuned validation results to: "
        f"{VALIDATION_RESULTS_PATH}"
    )

    print(
        f"Saved tuned Random Forest to: "
        f"{RF_MODEL_PATH}"
    )

    print(
        f"Saved tuned Extra Trees to: "
        f"{ET_MODEL_PATH}"
    )


if __name__ == "__main__":
    main()