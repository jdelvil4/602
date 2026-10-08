from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

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
    / "nrel_baseline_validation_metrics.csv"
)


TARGET_COLUMN = "target_1h_ahead"

BASELINE_COLUMNS = {
    "Persistence": "electricity_lag_1h",
    "Seasonal Naive": "seasonal_naive_24h",
    "Rolling Mean 24h": "electricity_rolling_mean_24h",
}


def load_validation_data() -> pd.DataFrame:

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"NREL validation dataset not found: "
            f"{VALIDATION_PATH}"
        )

    df = pd.read_parquet(
        VALIDATION_PATH
    )

    required_columns = {
        TARGET_COLUMN,
        *BASELINE_COLUMNS.values(),
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Validation dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    missing_required = int(
        df[
            list(
                required_columns
            )
        ]
        .isna()
        .sum()
        .sum()
    )

    if missing_required != 0:
        raise ValueError(
            f"Validation dataset contains "
            f"{missing_required} missing values "
            "in required baseline columns."
        )

    return df


def calculate_mae(
    actual: pd.Series,
    predicted: pd.Series,
) -> float:

    return float(
        np.mean(
            np.abs(
                actual
                - predicted
            )
        )
    )


def calculate_rmse(
    actual: pd.Series,
    predicted: pd.Series,
) -> float:

    return float(
        np.sqrt(
            np.mean(
                (
                    actual
                    - predicted
                )
                ** 2
            )
        )
    )


def calculate_cvrmse(
    actual: pd.Series,
    predicted: pd.Series,
) -> float:

    rmse = calculate_rmse(
        actual,
        predicted,
    )

    mean_actual = float(
        actual.mean()
    )

    if mean_actual == 0:
        raise ValueError(
            "Cannot calculate CVRMSE because "
            "mean actual value is zero."
        )

    return float(
        rmse
        / mean_actual
        * 100
    )


def calculate_mape(
    actual: pd.Series,
    predicted: pd.Series,
) -> tuple[float, int]:

    nonzero_mask = (
        actual
        != 0
    )

    actual_nonzero = (
        actual.loc[
            nonzero_mask
        ]
    )

    predicted_nonzero = (
        predicted.loc[
            nonzero_mask
        ]
    )

    n_nonzero = len(
        actual_nonzero
    )

    if n_nonzero == 0:
        raise ValueError(
            "Cannot calculate MAPE because "
            "all actual values are zero."
        )

    mape = float(
        np.mean(
            np.abs(
                (
                    actual_nonzero
                    - predicted_nonzero
                )
                / actual_nonzero
            )
        )
        * 100
    )

    return (
        mape,
        n_nonzero,
    )


def evaluate_baseline(
    model_name: str,
    prediction_column: str,
    df: pd.DataFrame,
) -> dict:

    actual = (
        df[
            TARGET_COLUMN
        ]
        .astype(float)
    )

    predicted = (
        df[
            prediction_column
        ]
        .astype(float)
    )

    mae = calculate_mae(
        actual,
        predicted,
    )

    rmse = calculate_rmse(
        actual,
        predicted,
    )

    cvrmse = calculate_cvrmse(
        actual,
        predicted,
    )

    mape, mape_n = calculate_mape(
        actual,
        predicted,
    )

    return {
        "model": model_name,
        "prediction_column": prediction_column,
        "rows_evaluated": len(
            df
        ),
        "mae": mae,
        "rmse": rmse,
        "cvrmse_pct": cvrmse,
        "mape_pct": mape,
        "mape_n": mape_n,
    }


def main() -> None:

    print(
        "Loading NREL validation dataset..."
    )

    validation = load_validation_data()

    print(
        f"Validation rows: "
        f"{len(validation):,}"
    )

    print()

    print(
        "Evaluating baseline forecasts..."
    )

    results = []

    for (
        model_name,
        prediction_column,
    ) in BASELINE_COLUMNS.items():

        result = evaluate_baseline(
            model_name,
            prediction_column,
            validation,
        )

        results.append(
            result
        )

        print()

        print(
            model_name
        )

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
            f"{result['cvrmse_pct']:.3f}%"
        )

        print(
            f"  MAPE: "
            f"{result['mape_pct']:.3f}%"
        )

        print(
            f"  MAPE n: "
            f"{result['mape_n']:,}"
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
        "NREL baseline evaluation complete."
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
        f"Best validation baseline by MAE: "
        f"{best_row['model']}"
    )

    print(
        f"Best validation MAE: "
        f"{best_row['mae']:.6f}"
    )

    print()

    print(
        f"Saved baseline metrics to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()