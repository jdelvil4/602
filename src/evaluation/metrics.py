import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


def calculate_mae(
    y_true,
    y_pred,
) -> float:

    return mean_absolute_error(
        y_true,
        y_pred,
    )


def calculate_cvrmse(
    y_true,
    y_pred,
) -> float:


    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    mean_actual = np.mean(y_true)

    if mean_actual == 0:
        return np.nan

    return (
        rmse / mean_actual
    ) * 100


def calculate_mape(
    y_true,
    y_pred,
) -> tuple[float, int]:


    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)

    nonzero_mask = (
        y_true_array != 0
    )

    valid_count = int(
        nonzero_mask.sum()
    )

    if valid_count == 0:
        return np.nan, 0

    mape = np.mean(
        np.abs(
            (
                y_true_array[nonzero_mask]
                - y_pred_array[nonzero_mask]
            )
            / y_true_array[nonzero_mask]
        )
    ) * 100

    return mape, valid_count


def calculate_all_metrics(
    y_true,
    y_pred,
) -> dict:

    mae = calculate_mae(
        y_true,
        y_pred,
    )

    cvrmse = calculate_cvrmse(
        y_true,
        y_pred,
    )

    mape, mape_n = calculate_mape(
        y_true,
        y_pred,
    )

    return {
        "MAE": mae,
        "CVRMSE_percent": cvrmse,
        "MAPE_percent": mape,
        "MAPE_n": mape_n,
    }