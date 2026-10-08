from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
    / "nrel_linear_validation_metrics.csv"
)


TARGET_COLUMN = "target_1h_ahead"


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


LASSO_ALPHA = 0.1


def load_dataset(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    required_columns = [
        TARGET_COLUMN,
        *NUMERIC_FEATURES,
        *CATEGORICAL_FEATURES,
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

    if df[
        TARGET_COLUMN
    ].isna().any():
        raise ValueError(
            f"{path.name} contains missing target values."
        )

    return df


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict:

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


def build_preprocessor() -> ColumnTransformer:

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

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
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
        ]
    )


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

    pipeline.fit(
        x_train,
        y_train,
    )

    print(
        f"Generating validation predictions for "
        f"{model_name}..."
    )

    predictions = np.asarray(
        pipeline.predict(
            x_validation
        ),
        dtype=float,
    )

    metrics = calculate_metrics(
        y_validation,
        predictions,
    )

    result = {
        "model": model_name,
        "rows_train": len(
            x_train
        ),
        "rows_validation": len(
            x_validation
        ),
        **metrics,
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

    return result

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

    feature_columns = [
        *NUMERIC_FEATURES,
        *CATEGORICAL_FEATURES,
    ]

    x_train = train[
        feature_columns
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
        feature_columns
    ]

    y_validation = (
        validation[
            TARGET_COLUMN
        ]
        .to_numpy(
            dtype=float
        )
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
        f"{len(feature_columns)}"
    )

    results = []

    linear_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                LinearRegression(
                    n_jobs=-1
                ),
            ),
        ]
    )

    linear_result = evaluate_model(
        "Linear Regression",
        linear_pipeline,
        x_train,
        y_train,
        x_validation,
        y_validation,
    )

    results.append(
        linear_result
    )

    lasso_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                Lasso(
                    alpha=LASSO_ALPHA,
                    max_iter=5000,
                    selection="cyclic",
                ),
            ),
        ]
    )

    lasso_result = evaluate_model(
        f"Lasso alpha={LASSO_ALPHA}",
        lasso_pipeline,
        x_train,
        y_train,
        x_validation,
        y_validation,
    )

    results.append(
        lasso_result
    )

    results_df = (
        pd.DataFrame(
            results
        )
        .sort_values(
            "mae"
        )
        .reset_index(drop=True)
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
        "NREL linear-model evaluation complete."
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
        f"Best learned validation model by MAE: "
        f"{best_row['model']}"
    )

    print(
        f"Best learned validation MAE: "
        f"{best_row['mae']:.6f}"
    )

    print()

    print(
        "Seasonal-naive benchmark MAE: "
        "0.256623"
    )

    improvement = (
        (
            0.256623
            - best_row[
                "mae"
            ]
        )
        / 0.256623
        * 100
    )

    print(
        f"Best learned-model MAE improvement "
        f"over seasonal naive: "
        f"{improvement:.2f}%"
    )

    print()

    print(
        f"Saved validation metrics to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()