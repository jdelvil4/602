from pathlib import Path
import time
from typing import TypedDict

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from src.evaluation.metrics import calculate_all_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

CV_RESULTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "bdg2_rf_tuning_cv_results.csv"
)

BEST_PARAMS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "bdg2_rf_tuned_best_params.csv"
)

VALIDATION_METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "bdg2_rf_tuned_validation_metrics.csv"
)

FINAL_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "bdg2_random_forest_tuned.joblib"
)

TARGET_COLUMN = (
    "target_electricity_next_hour"
)

TIMESTAMP_COLUMN = (
    "target_timestamp"
)

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

RANDOM_STATE = 42

CV_N_ESTIMATORS = 100
FINAL_N_ESTIMATORS = 200


class CandidateConfig(TypedDict):
    candidate: str
    max_depth: int | None
    min_samples_leaf: int
    max_features: float


CANDIDATES: list[CandidateConfig] = [
    {
        "candidate": "RF_candidate_1",
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": 0.7,
    },
    {
        "candidate": "RF_candidate_2",
        "max_depth": 20,
        "min_samples_leaf": 5,
        "max_features": 0.7,
    },
]


def build_expanding_folds(
    timestamps: pd.Series,
) -> list[
    tuple[
        pd.DatetimeIndex,
        pd.DatetimeIndex,
    ]
]:

    unique_timestamps = (
        pd.DatetimeIndex(
            timestamps
            .drop_duplicates()
            .sort_values()
        )
    )

    n_timestamps = len(
        unique_timestamps
    )

    if n_timestamps < 4:
        raise ValueError(
            "Not enough unique timestamps "
            "to construct three expanding folds."
        )

    fold_size = (
        n_timestamps // 4
    )

    if fold_size == 0:
        raise ValueError(
            "Fold size is zero."
        )

    fold_1_train_end = (
        fold_size
    )

    fold_1_validation_end = (
        fold_size * 2
    )

    fold_2_train_end = (
        fold_size * 2
    )

    fold_2_validation_end = (
        fold_size * 3
    )

    fold_3_train_end = (
        fold_size * 3
    )

    folds = [
        (
            unique_timestamps[
                :fold_1_train_end
            ],
            unique_timestamps[
                fold_1_train_end:
                fold_1_validation_end
            ],
        ),
        (
            unique_timestamps[
                :fold_2_train_end
            ],
            unique_timestamps[
                fold_2_train_end:
                fold_2_validation_end
            ],
        ),
        (
            unique_timestamps[
                :fold_3_train_end
            ],
            unique_timestamps[
                fold_3_train_end:
            ],
        ),
    ]

    return folds


def fit_candidate_fold(
    candidate: CandidateConfig,
    fold_number: int,
    train_fold: pd.DataFrame,
    validation_fold: pd.DataFrame,
) -> dict[str, object]:

    X_train = train_fold[
        FEATURE_COLUMNS
    ]

    y_train = train_fold[
        TARGET_COLUMN
    ]

    X_validation = validation_fold[
        FEATURE_COLUMNS
    ]

    y_validation = validation_fold[
        TARGET_COLUMN
    ]

    model = RandomForestRegressor(
        n_estimators=CV_N_ESTIMATORS,
        max_depth=candidate[
            "max_depth"
        ],
        min_samples_leaf=candidate[
            "min_samples_leaf"
        ],
        max_features=candidate[
            "max_features"
        ],
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    start = time.perf_counter()

    model.fit(
        X_train,
        y_train,
    )

    fit_seconds = (
        time.perf_counter()
        - start
    )

    start = time.perf_counter()

    predictions = model.predict(
        X_validation
    )

    predict_seconds = (
        time.perf_counter()
        - start
    )

    metrics = calculate_all_metrics(
        y_validation,
        predictions,
    )

    result: dict[str, object] = {
        "candidate":
            candidate[
                "candidate"
            ],
        "fold":
            fold_number,
        "max_depth":
            candidate[
                "max_depth"
            ],
        "min_samples_leaf":
            candidate[
                "min_samples_leaf"
            ],
        "max_features":
            candidate[
                "max_features"
            ],
        "n_estimators":
            CV_N_ESTIMATORS,
        "train_rows":
            len(
                train_fold
            ),
        "validation_rows":
            len(
                validation_fold
            ),
        "train_unique_timestamps":
            train_fold[
                TIMESTAMP_COLUMN
            ].nunique(),
        "validation_unique_timestamps":
            validation_fold[
                TIMESTAMP_COLUMN
            ].nunique(),
        "train_start":
            train_fold[
                TIMESTAMP_COLUMN
            ].min(),
        "train_end":
            train_fold[
                TIMESTAMP_COLUMN
            ].max(),
        "validation_start":
            validation_fold[
                TIMESTAMP_COLUMN
            ].min(),
        "validation_end":
            validation_fold[
                TIMESTAMP_COLUMN
            ].max(),
        **metrics,
        "fit_seconds":
            fit_seconds,
        "predict_seconds":
            predict_seconds,
    }

    return result


def select_best_candidate(
    cv_results: pd.DataFrame,
) -> tuple[
    CandidateConfig,
    pd.DataFrame,
]:

    summary = (
        cv_results
        .groupby(
            "candidate",
            observed=True,
        )
        .agg(
            mean_cv_MAE=(
                "MAE",
                "mean",
            ),
            std_cv_MAE=(
                "MAE",
                "std",
            ),
            mean_cv_CVRMSE_percent=(
                "CVRMSE_percent",
                "mean",
            ),
            mean_cv_MAPE_percent=(
                "MAPE_percent",
                "mean",
            ),
            total_fit_seconds=(
                "fit_seconds",
                "sum",
            ),
        )
        .reset_index()
        .sort_values(
            "mean_cv_MAE",
            ascending=True,
        )
        .reset_index(
            drop=True
        )
    )

    best_name = str(
        summary.loc[
            0,
            "candidate",
        ]
    )

    best_candidate = next(
        candidate
        for candidate
        in CANDIDATES
        if candidate[
            "candidate"
        ]
        == best_name
    )

    return (
        best_candidate,
        summary,
    )


def main() -> None:

    print(
        "Loading BDG2 training and validation data..."
    )

    required_columns = [
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
        TIMESTAMP_COLUMN,
    ]

    train = pd.read_parquet(
        TRAIN_PATH,
        columns=required_columns,
    )

    validation = pd.read_parquet(
        VALIDATION_PATH,
        columns=required_columns,
    )

    train[
        TIMESTAMP_COLUMN
    ] = pd.to_datetime(
        train[
            TIMESTAMP_COLUMN
        ],
        errors="raise",
    )

    validation[
        TIMESTAMP_COLUMN
    ] = pd.to_datetime(
        validation[
            TIMESTAMP_COLUMN
        ],
        errors="raise",
    )

    missing_train = int(
        train[
            required_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    missing_validation = int(
        validation[
            required_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    if missing_train > 0:
        raise ValueError(
            f"Training data contain "
            f"{missing_train:,} missing "
            f"required values."
        )

    if missing_validation > 0:
        raise ValueError(
            f"Validation data contain "
            f"{missing_validation:,} missing "
            f"required values."
        )

    print(
        f"Training rows: "
        f"{len(train):,}"
    )

    print(
        f"Training target timestamps: "
        f"{train[TIMESTAMP_COLUMN].nunique():,}"
    )

    print(
        f"Untouched validation rows: "
        f"{len(validation):,}"
    )

    folds = build_expanding_folds(
        train[
            TIMESTAMP_COLUMN
        ]
    )

    print()
    print(
        "Expanding timestamp-aware folds:"
    )

    for fold_number, (
        train_timestamps,
        validation_timestamps,
    ) in enumerate(
        folds,
        start=1,
    ):

        print()
        print(
            f"Fold {fold_number}"
        )

        print(
            f"  Train timestamps: "
            f"{len(train_timestamps):,}"
        )

        print(
            f"  Validation timestamps: "
            f"{len(validation_timestamps):,}"
        )

        print(
            f"  Train period: "
            f"{train_timestamps.min()} "
            f"to "
            f"{train_timestamps.max()}"
        )

        print(
            f"  Validation period: "
            f"{validation_timestamps.min()} "
            f"to "
            f"{validation_timestamps.max()}"
        )

    CV_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    FINAL_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv_results: list[
        dict[str, object]
    ] = []

    overall_start = (
        time.perf_counter()
    )

    for candidate in CANDIDATES:

        print()
        print(
            "=" * 70
        )

        print(
            f"Evaluating "
            f"{candidate['candidate']}"
        )

        print(
            f"max_depth="
            f"{candidate['max_depth']}, "
            f"min_samples_leaf="
            f"{candidate['min_samples_leaf']}, "
            f"max_features="
            f"{candidate['max_features']}"
        )

        for fold_number, (
            train_timestamps,
            validation_timestamps,
        ) in enumerate(
            folds,
            start=1,
        ):

            train_mask = (
                train[
                    TIMESTAMP_COLUMN
                ].isin(
                    train_timestamps
                )
            )

            validation_mask = (
                train[
                    TIMESTAMP_COLUMN
                ].isin(
                    validation_timestamps
                )
            )

            train_fold = train.loc[
                train_mask
            ]

            validation_fold = train.loc[
                validation_mask
            ]

            print()
            print(
                f"Candidate "
                f"{candidate['candidate']} "
                f"- Fold {fold_number}"
            )

            print(
                f"  Training rows: "
                f"{len(train_fold):,}"
            )

            print(
                f"  Validation rows: "
                f"{len(validation_fold):,}"
            )

            result = fit_candidate_fold(
                candidate,
                fold_number,
                train_fold,
                validation_fold,
            )

            cv_results.append(
                result
            )

            print(
                f"  MAE: "
                f"{result['MAE']:.6f}"
            )

            print(
                f"  CVRMSE: "
                f"{result['CVRMSE_percent']:.6f}%"
            )

            print(
                f"  MAPE: "
                f"{result['MAPE_percent']:.6f}%"
            )

            print(
                f"  Fit seconds: "
                f"{result['fit_seconds']:.2f}"
            )

            pd.DataFrame(
                cv_results
            ).to_csv(
                CV_RESULTS_PATH,
                index=False,
            )

    cv_results_df = pd.DataFrame(
        cv_results
    )

    best_candidate, cv_summary = (
        select_best_candidate(
            cv_results_df
        )
    )

    print()
    print(
        "=" * 70
    )

    print(
        "Cross-validation summary:"
    )

    print()

    print(
        cv_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Selected candidate:"
    )

    print(
        best_candidate
    )

    best_summary_row = (
        cv_summary.loc[
            cv_summary[
                "candidate"
            ]
            == best_candidate[
                "candidate"
            ]
        ]
        .iloc[
            0
        ]
    )

    best_params = pd.DataFrame(
        [
            {
                "candidate":
                    best_candidate[
                        "candidate"
                    ],
                "n_estimators":
                    FINAL_N_ESTIMATORS,
                "max_depth":
                    best_candidate[
                        "max_depth"
                    ],
                "min_samples_leaf":
                    best_candidate[
                        "min_samples_leaf"
                    ],
                "max_features":
                    best_candidate[
                        "max_features"
                    ],
                "random_state":
                    RANDOM_STATE,
                "mean_cv_MAE":
                    best_summary_row[
                        "mean_cv_MAE"
                    ],
                "std_cv_MAE":
                    best_summary_row[
                        "std_cv_MAE"
                    ],
                "mean_cv_CVRMSE_percent":
                    best_summary_row[
                        "mean_cv_CVRMSE_percent"
                    ],
                "mean_cv_MAPE_percent":
                    best_summary_row[
                        "mean_cv_MAPE_percent"
                    ],
            }
        ]
    )

    best_params.to_csv(
        BEST_PARAMS_PATH,
        index=False,
    )

    print()
    print(
        "Refitting selected Random Forest "
        "on full BDG2 training set..."
    )

    final_model = RandomForestRegressor(
        n_estimators=FINAL_N_ESTIMATORS,
        max_depth=best_candidate[
            "max_depth"
        ],
        min_samples_leaf=best_candidate[
            "min_samples_leaf"
        ],
        max_features=best_candidate[
            "max_features"
        ],
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    X_train = train[
        FEATURE_COLUMNS
    ]

    y_train = train[
        TARGET_COLUMN
    ]

    X_validation = validation[
        FEATURE_COLUMNS
    ]

    y_validation = validation[
        TARGET_COLUMN
    ]

    start = time.perf_counter()

    final_model.fit(
        X_train,
        y_train,
    )

    final_fit_seconds = (
        time.perf_counter()
        - start
    )

    print(
        f"Full training fit time: "
        f"{final_fit_seconds:.2f} seconds"
    )

    start = time.perf_counter()

    validation_predictions = (
        final_model.predict(
            X_validation
        )
    )

    validation_predict_seconds = (
        time.perf_counter()
        - start
    )

    validation_metrics = (
        calculate_all_metrics(
            y_validation,
            validation_predictions,
        )
    )

    validation_result = pd.DataFrame(
        [
            {
                "model":
                    "Tuned Random Forest",
                "candidate":
                    best_candidate[
                        "candidate"
                    ],
                "n_estimators":
                    FINAL_N_ESTIMATORS,
                "max_depth":
                    best_candidate[
                        "max_depth"
                    ],
                "min_samples_leaf":
                    best_candidate[
                        "min_samples_leaf"
                    ],
                "max_features":
                    best_candidate[
                        "max_features"
                    ],
                **validation_metrics,
                "fit_seconds":
                    final_fit_seconds,
                "predict_seconds":
                    validation_predict_seconds,
            }
        ]
    )

    validation_result.to_csv(
        VALIDATION_METRICS_PATH,
        index=False,
    )

    joblib.dump(
        final_model,
        FINAL_MODEL_PATH,
    )

    total_seconds = (
        time.perf_counter()
        - overall_start
    )

    print()
    print(
        "=" * 70
    )

    print(
        "BDG2 Random Forest tuning complete."
    )

    print()
    print(
        "Untouched validation result:"
    )

    print(
        validation_result.to_string(
            index=False
        )
    )

    print()
    print(
        f"Total elapsed time: "
        f"{total_seconds:.2f} seconds"
    )

    print()
    print(
        f"CV results: "
        f"{CV_RESULTS_PATH}"
    )

    print(
        f"Best parameters: "
        f"{BEST_PARAMS_PATH}"
    )

    print(
        f"Validation metrics: "
        f"{VALIDATION_METRICS_PATH}"
    )

    print(
        f"Tuned model: "
        f"{FINAL_MODEL_PATH}"
    )


if __name__ == "__main__":
    main()