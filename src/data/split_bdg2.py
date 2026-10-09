from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bdg2_forecast_features.parquet"
)

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bdg2_train.parquet"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bdg2_validation.parquet"
)

TEMPORAL_TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bdg2_temporal_test.parquet"
)

HELD_OUT_TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bdg2_held_out_test.parquet"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bdg2_split_summary.csv"
)

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15


def summarize_split(
    name: str,
    data: pd.DataFrame,
) -> dict[str, object]:

    return {
        "split": name,
        "rows": len(data),
        "buildings": (
            data["building_id"].nunique()
        ),
        "unique_target_timestamps": (
            data["target_timestamp"].nunique()
        ),
        "minimum_target_timestamp": (
            data["target_timestamp"].min()
        ),
        "maximum_target_timestamp": (
            data["target_timestamp"].max()
        ),
        "minimum_predictor_timestamp": (
            data["timestamp"].min()
        ),
        "maximum_predictor_timestamp": (
            data["timestamp"].max()
        ),
    }


def main() -> None:

    print(
        "Loading BDG2 forecast features..."
    )

    data = pd.read_parquet(
        INPUT_PATH
    )

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="raise",
    )

    data["target_timestamp"] = (
        pd.to_datetime(
            data["target_timestamp"],
            errors="raise",
        )
    )

    required_columns = [
        "building_id",
        "timestamp",
        "target_timestamp",
        "sample_group",
        "target_electricity_next_hour",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{missing_columns}"
        )

    duplicate_rows = int(
        data.duplicated(
            subset=[
                "building_id",
                "timestamp",
            ]
        ).sum()
    )

    if duplicate_rows > 0:
        raise ValueError(
            f"Duplicate building-timestamp rows: "
            f"{duplicate_rows:,}"
        )

    development = data[
        data["sample_group"]
        == "development"
    ].copy()

    held_out = data[
        data["sample_group"]
        == "held_out"
    ].copy()

    development_ids = set(
        development["building_id"]
        .unique()
    )

    held_out_ids = set(
        held_out["building_id"]
        .unique()
    )

    overlap = (
        development_ids
        & held_out_ids
    )

    if overlap:
        raise ValueError(
            "Development and held-out "
            "building IDs overlap."
        )

    print(
        f"Development buildings: "
        f"{len(development_ids):,}"
    )

    print(
        f"Held-out buildings: "
        f"{len(held_out_ids):,}"
    )

    target_timestamps = (
        development[
            "target_timestamp"
        ]
        .drop_duplicates()
        .sort_values()
        .reset_index(
            drop=True
        )
    )

    n_timestamps = len(
        target_timestamps
    )

    train_end = int(
        n_timestamps
        * TRAIN_FRACTION
    )

    validation_end = (
        train_end
        + int(
            n_timestamps
            * VALIDATION_FRACTION
        )
    )

    train_timestamps = set(
        target_timestamps.iloc[
            :train_end
        ]
    )

    validation_timestamps = set(
        target_timestamps.iloc[
            train_end:
            validation_end
        ]
    )

    test_timestamps = set(
        target_timestamps.iloc[
            validation_end:
        ]
    )

    train = development[
        development[
            "target_timestamp"
        ].isin(
            train_timestamps
        )
    ].copy()

    validation = development[
        development[
            "target_timestamp"
        ].isin(
            validation_timestamps
        )
    ].copy()

    temporal_test = development[
        development[
            "target_timestamp"
        ].isin(
            test_timestamps
        )
    ].copy()

    held_out_test = held_out[
        held_out[
            "target_timestamp"
        ].isin(
            test_timestamps
        )
    ].copy()

    if (
        train_timestamps
        & validation_timestamps
    ):
        raise ValueError(
            "Train and validation "
            "timestamps overlap."
        )

    if (
        train_timestamps
        & test_timestamps
    ):
        raise ValueError(
            "Train and test "
            "timestamps overlap."
        )

    if (
        validation_timestamps
        & test_timestamps
    ):
        raise ValueError(
            "Validation and test "
            "timestamps overlap."
        )

    if (
        set(train["building_id"])
        & held_out_ids
    ):
        raise ValueError(
            "Held-out building found "
            "in training data."
        )

    if (
        set(validation["building_id"])
        & held_out_ids
    ):
        raise ValueError(
            "Held-out building found "
            "in validation data."
        )

    if (
        set(temporal_test["building_id"])
        & held_out_ids
    ):
        raise ValueError(
            "Held-out building found "
            "in temporal test data."
        )

    train = train.sort_values(
        [
            "target_timestamp",
            "building_id",
        ]
    ).reset_index(
        drop=True
    )

    validation = validation.sort_values(
        [
            "target_timestamp",
            "building_id",
        ]
    ).reset_index(
        drop=True
    )

    temporal_test = temporal_test.sort_values(
        [
            "target_timestamp",
            "building_id",
        ]
    ).reset_index(
        drop=True
    )

    held_out_test = held_out_test.sort_values(
        [
            "target_timestamp",
            "building_id",
        ]
    ).reset_index(
        drop=True
    )

    summaries = [
        summarize_split(
            "train",
            train,
        ),
        summarize_split(
            "validation",
            validation,
        ),
        summarize_split(
            "temporal_test",
            temporal_test,
        ),
        summarize_split(
            "held_out_test",
            held_out_test,
        ),
    ]

    summary = pd.DataFrame(
        summaries
    )

    TRAIN_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    train.to_parquet(
        TRAIN_PATH,
        index=False,
    )

    validation.to_parquet(
        VALIDATION_PATH,
        index=False,
    )

    temporal_test.to_parquet(
        TEMPORAL_TEST_PATH,
        index=False,
    )

    held_out_test.to_parquet(
        HELD_OUT_TEST_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print()
    print(
        "BDG2 chronological split complete."
    )
    print()

    print(
        f"Unique development target "
        f"timestamps: {n_timestamps:,}"
    )

    print(
        f"Training timestamps: "
        f"{len(train_timestamps):,}"
    )

    print(
        f"Validation timestamps: "
        f"{len(validation_timestamps):,}"
    )

    print(
        f"Test timestamps: "
        f"{len(test_timestamps):,}"
    )

    print()

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Building-ID overlap checks:"
    )

    print(
        f"Development / held-out overlap: "
        f"{len(overlap)}"
    )

    print()
    print(
        f"Training set: "
        f"{TRAIN_PATH}"
    )

    print(
        f"Validation set: "
        f"{VALIDATION_PATH}"
    )

    print(
        f"Temporal test set: "
        f"{TEMPORAL_TEST_PATH}"
    )

    print(
        f"Held-out test set: "
        f"{HELD_OUT_TEST_PATH}"
    )

    print(
        f"Summary: "
        f"{SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()