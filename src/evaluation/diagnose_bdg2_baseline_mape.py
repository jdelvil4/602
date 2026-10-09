from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bdg2_validation.parquet"
)

LINEAR_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "bdg2_linear_baseline.joblib"
)

LASSO_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "bdg2_lasso_baseline.joblib"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "bdg2_baseline_mape_diagnostics.csv"
)

TARGET_COLUMN = "target_electricity_next_hour"

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


def summarize_predictions(
    name: str,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, object]:

    absolute_error = np.abs(
        actual - predicted
    )

    percentage_error = (
        absolute_error
        / np.abs(actual)
        * 100.0
    )

    return {
        "model": name,
        "prediction_min":
            float(np.min(predicted)),
        "prediction_median":
            float(np.median(predicted)),
        "prediction_max":
            float(np.max(predicted)),
        "negative_predictions":
            int(
                np.sum(
                    predicted < 0
                )
            ),
        "median_absolute_error":
            float(
                np.median(
                    absolute_error
                )
            ),
        "median_percentage_error":
            float(
                np.median(
                    percentage_error
                )
            ),
        "p95_percentage_error":
            float(
                np.quantile(
                    percentage_error,
                    0.95,
                )
            ),
        "p99_percentage_error":
            float(
                np.quantile(
                    percentage_error,
                    0.99,
                )
            ),
        "maximum_percentage_error":
            float(
                np.max(
                    percentage_error
                )
            ),
    }


def main() -> None:

    print(
        "Loading BDG2 validation data..."
    )

    validation = pd.read_parquet(
        VALIDATION_PATH,
        columns=[
            *FEATURE_COLUMNS,
            TARGET_COLUMN,
        ],
    )

    actual = (
        validation[
            TARGET_COLUMN
        ]
        .to_numpy()
    )

    print(
        f"Validation rows: "
        f"{len(actual):,}"
    )

    print()
    print(
        "Actual target distribution:"
    )

    quantiles = [
        0.00,
        0.001,
        0.01,
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
        1.00,
    ]

    target_quantiles = (
        validation[
            TARGET_COLUMN
        ]
        .quantile(
            quantiles
        )
    )

    print(
        target_quantiles.to_string()
    )

    print()
    print(
        "Near-zero target counts:"
    )

    thresholds = [
        0.001,
        0.01,
        0.1,
        1.0,
        5.0,
        10.0,
    ]

    for threshold in thresholds:

        count = int(
            np.sum(
                np.abs(actual)
                < threshold
            )
        )

        percentage = (
            count
            / len(actual)
            * 100.0
        )

        print(
            f"|actual| < {threshold}: "
            f"{count:,} "
            f"({percentage:.4f}%)"
        )

    X_validation = validation[
        FEATURE_COLUMNS
    ]

    print()
    print(
        "Loading fitted baseline models..."
    )

    linear = joblib.load(
        LINEAR_MODEL_PATH
    )

    lasso = joblib.load(
        LASSO_MODEL_PATH
    )

    linear_predictions = (
        linear.predict(
            X_validation
        )
    )

    lasso_predictions = (
        lasso.predict(
            X_validation
        )
    )

    diagnostics = pd.DataFrame(
        [
            summarize_predictions(
                "Linear Regression",
                actual,
                linear_predictions,
            ),
            summarize_predictions(
                "Lasso alpha=0.1",
                actual,
                lasso_predictions,
            ),
        ]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostics.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "Prediction diagnostics:"
    )

    print(
        diagnostics.to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved diagnostics: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()