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

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "nrel_tree_validation_metrics.csv"
)


TARGET_COLUMN = "target_1h_ahead"

RANDOM_STATE = 42

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


def load_dataset(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    required_columns = [
        TARGET_COLUMN,
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

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"{path.name} contains missing target values."
        )

    return df


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

    preprocessor = ColumnTransformer(
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

    return preprocessor


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
            "mean actual value is zero."
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


def evaluate_model(
    model_name: str,
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_validation: pd.DataFrame,
    y_validation: np.ndarray,
) -> dict:

    print()

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
        f"Training time: "
        f"{training_seconds:.2f} seconds"
    )

    print(
        f"Generating validation predictions for "
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

    improvement = float(
        (
            SEASONAL_NAIVE_MAE
            - metrics["mae"]
        )
        / SEASONAL_NAIVE_MAE
        * 100
    )

    result = {
        "model": model_name,
        "rows_train": len(
            x_train
        ),
        "rows_validation": len(
            x_validation
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
    }

    print(
        f"  MAE: "
        f"{metrics['mae']:.6f}"
    )

    print(
        f"  RMSE: "
        f"{metrics['rmse']:.6f}"
    )

    print(
        f"  CVRMSE: "
        f"{metrics['cvrmse_pct']:.3f}%"
    )

    print(
        f"  MAPE: "
        f"{metrics['mape_pct']:.3f}%"
    )

    print(
        f"  MAPE n: "
        f"{metrics['mape_n']:,}"
    )

    print(
        "  MAE improvement over "
        f"Seasonal Naive: "
        f"{improvement:.2f}%"
    )

    return result


def build_random_forest_pipeline() -> Pipeline:

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=16,
        min_samples_leaf=5,
        max_features=0.7,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0,
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


def build_extra_trees_pipeline() -> Pipeline:

    model = ExtraTreesRegressor(
        n_estimators=100,
        max_depth=16,
        min_samples_leaf=5,
        max_features=0.7,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0,
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


def build_hist_gradient_boosting_pipeline() -> Pipeline:

    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=100,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=20,
        l2_regularization=0.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=RANDOM_STATE,
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
        f"Numeric predictors: "
        f"{len(NUMERIC_FEATURES)}"
    )

    print(
        f"Categorical predictors: "
        f"{len(CATEGORICAL_FEATURES)}"
    )

    print(
        f"Raw predictor columns: "
        f"{len(FEATURE_COLUMNS)}"
    )

    print()

    print(
        "Primary benchmark:"
    )

    print(
        f"  Seasonal Naive MAE: "
        f"{SEASONAL_NAIVE_MAE:.6f}"
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

    results = []

    random_forest = (
        build_random_forest_pipeline()
    )

    random_forest_result = (
        evaluate_model(
            "Random Forest",
            random_forest,
            x_train,
            y_train,
            x_validation,
            y_validation,
        )
    )

    results.append(
        random_forest_result
    )

    del random_forest

    extra_trees = (
        build_extra_trees_pipeline()
    )

    extra_trees_result = (
        evaluate_model(
            "Extra Trees",
            extra_trees,
            x_train,
            y_train,
            x_validation,
            y_validation,
        )
    )

    results.append(
        extra_trees_result
    )

    del extra_trees

    hist_gradient_boosting = (
        build_hist_gradient_boosting_pipeline()
    )

    hist_gradient_boosting_result = (
        evaluate_model(
            "Hist Gradient Boosting",
            hist_gradient_boosting,
            x_train,
            y_train,
            x_validation,
            y_validation,
        )
    )

    results.append(
        hist_gradient_boosting_result
    )

    del hist_gradient_boosting

    results_df = (
        pd.DataFrame(
            results
        )
        .sort_values(
            "mae"
        )
        .reset_index(
            drop=True
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
        "NREL tree-model evaluation complete."
    )

    print()

    print(
        results_df.to_string(
            index=False
        )
    )

    print()

    best_row = (
        results_df.iloc[
            0
        ]
    )

    print(
        f"Best tree model by validation MAE: "
        f"{best_row['model']}"
    )

    print(
        f"Best validation MAE: "
        f"{best_row['mae']:.6f}"
    )

    print(
        f"Seasonal-naive benchmark MAE: "
        f"{SEASONAL_NAIVE_MAE:.6f}"
    )

    print(
        "Best tree-model MAE improvement "
        f"over Seasonal Naive: "
        f"{best_row['mae_improvement_vs_seasonal_naive_pct']:.2f}%"
    )

    print()

    print(
        f"Saved validation metrics to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()