from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.config import (
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
    TEST_FRACTION,
)

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "uci_forecast_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

TRAIN_PATH = (
    OUTPUT_DIR
    / "uci_train.csv"
)

VALIDATION_PATH = (
    OUTPUT_DIR
    / "uci_validation.csv"
)

TEST_PATH = (
    OUTPUT_DIR
    / "uci_test.csv"
)

def load_feature_data(
    path: Path,
) -> pd.DataFrame:


    if not path.exists():
        raise FileNotFoundError(
            f"Forecast feature file not found: {path}"
        )

    df = pd.read_csv(
        path,
        parse_dates=["date"],
    )

    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )

    print(
        f"Loaded {len(df):,} feature rows."
    )

    return df

def validate_split_fractions() -> None:


    total = (
        TRAIN_FRACTION
        + VALIDATION_FRACTION
        + TEST_FRACTION
    )

    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            "Train/validation/test fractions "
            f"sum to {total}, not 1.0."
        )

def chronological_split(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    n_rows = len(df)

    train_end = int(
        n_rows * TRAIN_FRACTION
    )

    validation_end = int(
        n_rows
        * (
            TRAIN_FRACTION
            + VALIDATION_FRACTION
        )
    )

    train = (
        df
        .iloc[:train_end]
        .copy()
        .reset_index(drop=True)
    )

    validation = (
        df
        .iloc[
            train_end:validation_end
        ]
        .copy()
        .reset_index(drop=True)
    )

    test = (
        df
        .iloc[
            validation_end:
        ]
        .copy()
        .reset_index(drop=True)
    )

    return (
        train,
        validation,
        test,
    )

def validate_partitions(
    full_df: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:

    total_split_rows = (
        len(train)
        + len(validation)
        + len(test)
    )

    if total_split_rows != len(full_df):
        raise ValueError(
            "Combined split row counts do not "
            "equal the original dataset row count."
        )

    if len(train) == 0:
        raise ValueError(
            "Training partition is empty."
        )

    if len(validation) == 0:
        raise ValueError(
            "Validation partition is empty."
        )

    if len(test) == 0:
        raise ValueError(
            "Test partition is empty."
        )

    if not train["date"].is_monotonic_increasing:
        raise ValueError(
            "Training data are not chronological."
        )

    if not validation["date"].is_monotonic_increasing:
        raise ValueError(
            "Validation data are not chronological."
        )

    if not test["date"].is_monotonic_increasing:
        raise ValueError(
            "Test data are not chronological."
        )

    if (
        train["date"].max()
        >= validation["date"].min()
    ):
        raise ValueError(
            "Training and validation periods overlap."
        )

    if (
        validation["date"].max()
        >= test["date"].min()
    ):
        raise ValueError(
            "Validation and test periods overlap."
        )

    duplicate_train = (
        train["date"]
        .duplicated()
        .sum()
    )

    duplicate_validation = (
        validation["date"]
        .duplicated()
        .sum()
    )

    duplicate_test = (
        test["date"]
        .duplicated()
        .sum()
    )

    if (
        duplicate_train
        + duplicate_validation
        + duplicate_test
        != 0
    ):
        raise ValueError(
            "Duplicate timestamps detected "
            "within one or more partitions."
        )

def print_split_summary(
    full_df: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:

    total = len(full_df)

    partitions = {
        "Train": train,
        "Validation": validation,
        "Test": test,
    }

    print()
    print(
        "Chronological split summary"
    )
    print(
        "==========================="
    )

    for name, partition in partitions.items():

        percentage = (
            len(partition)
            / total
            * 100
        )

        print()
        print(name)
        print("-" * len(name))

        print(
            f"Rows: {len(partition):,}"
        )

        print(
            f"Share: {percentage:.2f}%"
        )

        print(
            f"Start: {partition['date'].min()}"
        )

        print(
            f"End:   {partition['date'].max()}"
        )

def save_partitions(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train.to_csv(
        TRAIN_PATH,
        index=False,
    )

    validation.to_csv(
        VALIDATION_PATH,
        index=False,
    )

    test.to_csv(
        TEST_PATH,
        index=False,
    )

    print()
    print(
        f"Saved training data to: {TRAIN_PATH}"
    )

    print(
        "Saved validation data to: "
        f"{VALIDATION_PATH}"
    )

    print(
        f"Saved test data to: {TEST_PATH}"
    )

def main() -> None:

    validate_split_fractions()

    df = load_feature_data(
        INPUT_PATH
    )

    (
        train,
        validation,
        test,
    ) = chronological_split(
        df
    )

    validate_partitions(
        full_df=df,
        train=train,
        validation=validation,
        test=test,
    )

    print_split_summary(
        full_df=df,
        train=train,
        validation=validation,
        test=test,
    )

    save_partitions(
        train=train,
        validation=validation,
        test=test,
    )


if __name__ == "__main__":
    main()