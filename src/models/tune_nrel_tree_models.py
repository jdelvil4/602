from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_train.parquet"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_validation.parquet"
)

CV_RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "nrel_tree_tuning_cv_results.csv"
)

VALIDATION_RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "nrel_tuned_tree_validation_metrics.csv"
)

BEST_PARAMETERS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "nrel_tree_best_parameters.csv"
)


TARGET_COLUMN = "target_1h_ahead"
TARGET_TIMESTAMP_COLUMN = "target_timestamp"

RANDOM_STATE = 42

CV_FOLDS = 3

FOREST_TUNING_TREES = 50
FOREST_FINAL_TREES = 200

SEASONAL_NAIVE_MAE = 0.256623


NUMERIC_FEATURES = [
    "in.sqft",
    "in.occupants",
    "in.bedrooms",
    "in.geometry_stories",
    "electricity_lag_1h",
    "electricity_lag_24h",
    "electricity_rolling_mean_24h",
    "electricity_rolling_mean_168h",
    "weather_dry_bulb_temperature_c",
    "weather_relative_humidity_pct",
    "weather_wind_speed_m_s",
    "weather_global_horizontal_radiation_w_m2",
    "weather_direct_normal_radiation_w_m2",
    "weather_diffuse_horizontal_radiation_w_m2",
    "heating_degree_c",
    "cooling_degree_c",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "is_weekend",
]


CATEGORICAL_FEATURES = [
    "in.geometry_building_type_recs",
    "in.cec_climate_zone",
    "in.vintage",
    "in.heating_fuel",
    "in.hvac_cooling_type",
    "in.hvac_heating_type",
]


FEATURE_COLUMNS = [
    *NUMERIC_FEATURES,
    *CATEGORICAL_FEATURES,
]


RANDOM_FOREST_CANDIDATES = [
    {
        "max_depth": 12,
        "min_samples_leaf": 5,
        "max_features": 0.7,
    },
    {
        "max_depth": 16,
        "min_samples_leaf": 5,
        "max_features": 0.7,
    },
    {
        "max_depth": 20,
        "min_samples_leaf": 5,
        "max_features": 0.7,
    },
    {
        "max_depth": 16,
        "min_samples_leaf": 10,
        "max_features": 1.0,
    },
]


EXTRA_TREES_CANDIDATES = [
    {
        "max_depth": 12,
        "min_samples_leaf": 5,
        "max_features": 0.7,
    },
    {
        "max_depth": 16,
        "min_samples_leaf": 5,
        "max_features": 0.7,
    },
    {
        "max_depth": 20,
        "min_samples_leaf": 5,
        "max_features": 0.7,
    },
    {
        "max_depth": 16,
        "min_samples_leaf": 10,
        "max_features": 1.0,
    },
]


HIST_GRADIENT_BOOSTING_CANDIDATES = [
    {
        "learning_rate": 0.03,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
    },
    {
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
    },
    {
        "learning_rate": 0.08,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
    },
    {
        "learning_rate": 0.05,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
    },
    {
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 50,
        "l2_regularization": 0.0,
    },
    {
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.1,
    },
]


def load_dataset(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    required_columns = [
        TARGET_COLUMN,
        TARGET_TIMESTAMP_COLUMN,
        *FEATURE_COLUMNS,
    ]

    df = pd.read_parquet(
        path,
        columns=required_columns,
    )

    missing_columns = (
        set(required_columns)
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df[
        TARGET_TIMESTAMP_COLUMN
    ] = pd.to_datetime(
        df[
            TARGET_TIMESTAMP_COLUMN
        ]
    )

    if df[
        TARGET_COLUMN
    ].isna().any():
        raise ValueError(
            f"{path.name} contains missing target values."
        )

    return (
        df
        .sort_values(
            TARGET_TIMESTAMP_COLUMN
        )
        .reset_index(drop=True)
    )


def build_preprocessor() -> ColumnTransformer:

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                    dtype=np.float64,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def calculate_mae(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:

    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    return float(
        np.mean(
            np.abs(
                actual
                - predicted
            )
        )
    )


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

    mae = float(
        np.mean(
            np.abs(
                error
            )
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )

    mean_actual = float(
        np.mean(
            actual
        )
    )

    if mean_actual == 0:
        raise ValueError(
            "Cannot calculate CVRMSE because "
            "mean actual is zero."
        )

    cvrmse = float(
        rmse
        / mean_actual
        * 100
    )

    nonzero_mask = (
        actual
        != 0
    )

    mape_n = int(
        nonzero_mask.sum()
    )

    if mape_n == 0:
        raise ValueError(
            "Cannot calculate MAPE because "
            "all actual values are zero."
        )

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
        * 100
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "cvrmse_pct": cvrmse,
        "mape_pct": mape,
        "mape_n": mape_n,
    }


def create_temporal_folds(
    train: pd.DataFrame,
) -> list[dict]:

    unique_timestamps = (
        train[
            TARGET_TIMESTAMP_COLUMN
        ]
        .drop_duplicates()
        .sort_values()
        .to_numpy()
    )

    blocks = np.array_split(
        unique_timestamps,
        CV_FOLDS + 1,
    )

    folds = []

    for fold_number in range(
        1,
        CV_FOLDS + 1,
    ):

        train_timestamps = np.concatenate(
            blocks[
                :fold_number
            ]
        )

        validation_timestamps = (
            blocks[
                fold_number
            ]
        )

        fold = {
            "fold": fold_number,
            "train_timestamps": train_timestamps,
            "validation_timestamps": validation_timestamps,
        }

        folds.append(
            fold
        )

    return folds


def build_model(
    model_name: str,
    parameters: dict,
    final_fit: bool = False,
):

    if model_name == "Random Forest":

        n_estimators = (
            FOREST_FINAL_TREES
            if final_fit
            else FOREST_TUNING_TREES
        )

        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=parameters[
                "max_depth"
            ],
            min_samples_leaf=parameters[
                "min_samples_leaf"
            ],
            max_features=parameters[
                "max_features"
            ],
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=0,
        )

    if model_name == "Extra Trees":

        n_estimators = (
            FOREST_FINAL_TREES
            if final_fit
            else FOREST_TUNING_TREES
        )

        return ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=parameters[
                "max_depth"
            ],
            min_samples_leaf=parameters[
                "min_samples_leaf"
            ],
            max_features=parameters[
                "max_features"
            ],
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=0,
        )

    if model_name == "Hist Gradient Boosting":

        return HistGradientBoostingRegressor(
            learning_rate=parameters[
                "learning_rate"
            ],
            max_iter=200,
            max_leaf_nodes=parameters[
                "max_leaf_nodes"
            ],
            max_depth=None,
            min_samples_leaf=parameters[
                "min_samples_leaf"
            ],
            l2_regularization=parameters[
                "l2_regularization"
            ],
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=RANDOM_STATE,
        )

    raise ValueError(
        f"Unsupported model: {model_name}"
    )


def build_pipeline(
    model_name: str,
    parameters: dict,
    final_fit: bool = False,
) -> Pipeline:

    model = build_model(
        model_name,
        parameters,
        final_fit=final_fit,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                model,
            ),
        ]
    )


def evaluate_candidate_cv(
    model_name: str,
    candidate_number: int,
    parameters: dict,
    train: pd.DataFrame,
    folds: list[dict],
) -> list[dict]:

    fold_results = []

    for fold in folds:

        fold_number = int(
            fold[
                "fold"
            ]
        )

        print(
            f"    Fold {fold_number}/{CV_FOLDS}"
        )

        train_mask = train[
            TARGET_TIMESTAMP_COLUMN
        ].isin(
            fold[
                "train_timestamps"
            ]
        )

        validation_mask = train[
            TARGET_TIMESTAMP_COLUMN
        ].isin(
            fold[
                "validation_timestamps"
            ]
        )

        fold_train = (
            train.loc[
                train_mask
            ]
        )

        fold_validation = (
            train.loc[
                validation_mask
            ]
        )

        x_train = fold_train[
            FEATURE_COLUMNS
        ]

        y_train = (
            fold_train[
                TARGET_COLUMN
            ]
            .to_numpy(
                dtype=float
            )
        )

        x_validation = (
            fold_validation[
                FEATURE_COLUMNS
            ]
        )

        y_validation = (
            fold_validation[
                TARGET_COLUMN
            ]
            .to_numpy(
                dtype=float
            )
        )

        pipeline = build_pipeline(
            model_name,
            parameters,
            final_fit=False,
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

        prediction_start = (
            perf_counter()
        )

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

        mae = calculate_mae(
            y_validation,
            predictions,
        )

        result = {
            "model": model_name,
            "candidate": candidate_number,
            "fold": fold_number,
            "train_rows": len(
                fold_train
            ),
            "validation_rows": len(
                fold_validation
            ),
            "train_target_start": fold_train[
                TARGET_TIMESTAMP_COLUMN
            ].min(),
            "train_target_end": fold_train[
                TARGET_TIMESTAMP_COLUMN
            ].max(),
            "validation_target_start": fold_validation[
                TARGET_TIMESTAMP_COLUMN
            ].min(),
            "validation_target_end": fold_validation[
                TARGET_TIMESTAMP_COLUMN
            ].max(),
            "mae": mae,
            "fit_seconds": fit_seconds,
            "prediction_seconds": (
                prediction_seconds
            ),
            **parameters,
        }

        fold_results.append(
            result
        )

        print(
            f"      MAE: {mae:.6f}"
        )

        print(
            f"      Fit time: "
            f"{fit_seconds:.2f} seconds"
        )

        del pipeline

    return fold_results


def tune_model(
    model_name: str,
    candidates: list[dict],
    train: pd.DataFrame,
    folds: list[dict],
) -> tuple[
    dict,
    pd.DataFrame,
]:

    print()

    print(
        f"Tuning {model_name}..."
    )

    all_results = []

    candidate_summaries = []

    for candidate_number, parameters in enumerate(
        candidates,
        start=1,
    ):

        print()

        print(
            f"  Candidate "
            f"{candidate_number}/{len(candidates)}"
        )

        print(
            f"  Parameters: {parameters}"
        )

        fold_results = evaluate_candidate_cv(
            model_name,
            candidate_number,
            parameters,
            train,
            folds,
        )

        all_results.extend(
            fold_results
        )

        fold_df = pd.DataFrame(
            fold_results
        )

        mean_mae = float(
            fold_df[
                "mae"
            ].mean()
        )

        std_mae = float(
            fold_df[
                "mae"
            ].std(
                ddof=0
            )
        )

        total_fit_seconds = float(
            fold_df[
                "fit_seconds"
            ].sum()
        )

        summary = {
            "model": model_name,
            "candidate": candidate_number,
            "mean_cv_mae": mean_mae,
            "std_cv_mae": std_mae,
            "total_cv_fit_seconds": (
                total_fit_seconds
            ),
            **parameters,
        }

        candidate_summaries.append(
            summary
        )

        print(
            f"  Mean CV MAE: "
            f"{mean_mae:.6f}"
        )

        print(
            f"  CV MAE std: "
            f"{std_mae:.6f}"
        )

    candidate_summary_df = (
        pd.DataFrame(
            candidate_summaries
        )
        .sort_values(
            [
                "mean_cv_mae",
                "std_cv_mae",
            ]
        )
        .reset_index(drop=True)
    )

    best_candidate_number = int(
        candidate_summary_df.iloc[
            0
        ][
            "candidate"
        ]
    )

    best_parameters = candidates[
        best_candidate_number
        - 1
    ]

    print()

    print(
        f"Best {model_name} candidate: "
        f"{best_candidate_number}"
    )

    print(
        f"Best parameters: "
        f"{best_parameters}"
    )

    print(
        f"Best mean CV MAE: "
        f"{candidate_summary_df.iloc[0]['mean_cv_mae']:.6f}"
    )

    detailed_results = pd.DataFrame(
        all_results
    )

    return (
        best_parameters,
        detailed_results,
    )


def evaluate_final_configuration(
    model_name: str,
    parameters: dict,
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> dict:

    print()

    print(
        f"Refitting tuned {model_name} "
        "on full training data..."
    )

    x_train = train[
        FEATURE_COLUMNS
    ]

    y_train = (
        train[
            TARGET_COLUMN
        ]
        .to_numpy(
            dtype=float
        )
    )

    x_validation = validation[
        FEATURE_COLUMNS
    ]

    y_validation = (
        validation[
            TARGET_COLUMN
        ]
        .to_numpy(
            dtype=float
        )
    )

    pipeline = build_pipeline(
        model_name,
        parameters,
        final_fit=True,
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
        f"Training time: "
        f"{training_seconds:.2f} seconds"
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

    improvement = float(
        (
            SEASONAL_NAIVE_MAE
            - metrics[
                "mae"
            ]
        )
        / SEASONAL_NAIVE_MAE
        * 100
    )

    result = {
        "model": model_name,
        "rows_train": len(
            train
        ),
        "rows_validation": len(
            validation
        ),
        "mae": metrics[
            "mae"
        ],
        "rmse": metrics[
            "rmse"
        ],
        "cvrmse_pct": metrics[
            "cvrmse_pct"
        ],
        "mape_pct": metrics[
            "mape_pct"
        ],
        "mape_n": metrics[
            "mape_n"
        ],
        "mae_improvement_vs_seasonal_naive_pct": (
            improvement
        ),
        "training_seconds": (
            training_seconds
        ),
        "prediction_seconds": (
            prediction_seconds
        ),
        **parameters,
    }

    print(
        f"  Validation MAE: "
        f"{metrics['mae']:.6f}"
    )

    print(
        f"  Validation RMSE: "
        f"{metrics['rmse']:.6f}"
    )

    print(
        f"  Validation CVRMSE: "
        f"{metrics['cvrmse_pct']:.3f}%"
    )

    print(
        f"  Validation MAPE: "
        f"{metrics['mape_pct']:.3f}%"
    )

    print(
        f"  Improvement over Seasonal Naive: "
        f"{improvement:.2f}%"
    )

    del pipeline

    return result


def create_best_parameter_rows(
    best_parameters_by_model: dict,
) -> pd.DataFrame:

    rows = []

    for (
        model_name,
        parameters,
    ) in best_parameters_by_model.items():

        row = {
            "model": model_name,
        }

        row.update(
            parameters
        )

        if model_name in {
            "Random Forest",
            "Extra Trees",
        }:
            row[
                "final_n_estimators"
            ] = FOREST_FINAL_TREES

        if model_name == "Hist Gradient Boosting":
            row[
                "max_iter"
            ] = 200

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def main() -> None:

    print(
        "Loading NREL training dataset..."
    )

    train = load_dataset(
        TRAIN_PATH
    )

    print(
        f"Training rows: "
        f"{len(train):,}"
    )

    print(
        f"Unique training target timestamps: "
        f"{train[TARGET_TIMESTAMP_COLUMN].nunique():,}"
    )

    print()

    print(
        "Loading NREL validation dataset..."
    )

    validation = load_dataset(
        VALIDATION_PATH
    )

    print(
        f"Validation rows: "
        f"{len(validation):,}"
    )

    print()

    print(
        "Creating expanding timestamp-aware "
        f"{CV_FOLDS}-fold cross-validation..."
    )

    folds = create_temporal_folds(
        train
    )

    for fold in folds:

        train_timestamp_count = len(
            fold[
                "train_timestamps"
            ]
        )

        validation_timestamp_count = len(
            fold[
                "validation_timestamps"
            ]
        )

        print(
            f"  Fold {fold['fold']}: "
            f"{train_timestamp_count:,} train timestamps, "
            f"{validation_timestamp_count:,} validation timestamps"
        )

    all_cv_results = []

    best_parameters_by_model = {}

    model_configs = [
        (
            "Random Forest",
            RANDOM_FOREST_CANDIDATES,
        ),
        (
            "Extra Trees",
            EXTRA_TREES_CANDIDATES,
        ),
        (
            "Hist Gradient Boosting",
            HIST_GRADIENT_BOOSTING_CANDIDATES,
        ),
    ]

    for (
        model_name,
        candidates,
    ) in model_configs:

        (
            best_parameters,
            cv_results,
        ) = tune_model(
            model_name,
            candidates,
            train,
            folds,
        )

        best_parameters_by_model[
            model_name
        ] = best_parameters

        all_cv_results.append(
            cv_results
        )

    combined_cv_results = pd.concat(
        all_cv_results,
        ignore_index=True,
    )

    CV_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined_cv_results.to_csv(
        CV_RESULTS_PATH,
        index=False,
    )

    print()

    print(
        "Cross-validation tuning complete."
    )

    print()

    print(
        "Evaluating tuned models on the "
        "untouched validation set..."
    )

    validation_results = []

    for model_name in [
        "Random Forest",
        "Extra Trees",
        "Hist Gradient Boosting",
    ]:

        result = (
            evaluate_final_configuration(
                model_name,
                best_parameters_by_model[
                    model_name
                ],
                train,
                validation,
            )
        )

        validation_results.append(
            result
        )

    validation_results_df = (
        pd.DataFrame(
            validation_results
        )
        .sort_values(
            "mae"
        )
        .reset_index(drop=True)
    )

    validation_results_df.to_csv(
        VALIDATION_RESULTS_PATH,
        index=False,
    )

    best_parameters_df = (
        create_best_parameter_rows(
            best_parameters_by_model
        )
    )

    best_parameters_df.to_csv(
        BEST_PARAMETERS_PATH,
        index=False,
    )

    print()

    print(
        "NREL tree-model tuning complete."
    )

    print()

    print(
        "Tuned validation results:"
    )

    print(
        validation_results_df.to_string(
            index=False
        )
    )

    print()

    best_model = (
        validation_results_df.iloc[
            0
        ]
    )

    print(
        f"Best tuned model by validation MAE: "
        f"{best_model['model']}"
    )

    print(
        f"Best tuned validation MAE: "
        f"{best_model['mae']:.6f}"
    )

    print(
        f"Seasonal-naive benchmark MAE: "
        f"{SEASONAL_NAIVE_MAE:.6f}"
    )

    print(
        "Best tuned-model improvement "
        f"over Seasonal Naive: "
        f"{best_model['mae_improvement_vs_seasonal_naive_pct']:.2f}%"
    )

    print()

    print(
        f"Saved CV results to: "
        f"{CV_RESULTS_PATH}"
    )

    print(
        f"Saved tuned validation metrics to: "
        f"{VALIDATION_RESULTS_PATH}"
    )

    print(
        f"Saved best parameters to: "
        f"{BEST_PARAMETERS_PATH}"
    )


if __name__ == "__main__":
    main()