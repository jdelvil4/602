from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "uci_all_validation_metrics.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "uci_model_validation_mae_comparison.png"
)


def main() -> None:

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Results file not found: {INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH
    )

    required_columns = {
        "Model",
        "MAE",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    plot_df = (
        df
        .sort_values(
            "MAE",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    plt.figure(
        figsize=(11, 7)
    )

    plt.barh(
        plot_df.loc[:, "Model"],
        plot_df.loc[:, "MAE"],
    )

    plt.xlabel(
        "Validation MAE"
    )

    plt.ylabel(
        "Model"
    )

    plt.title(
        "UCI One-Hour-Ahead Forecasting Model Comparison"
    )

    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved comparison figure to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()