from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_modeling_hourly.parquet"
)

SAMPLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nrel_ca_modeling_sample.csv"
)

SILHOUETTE_OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "nrel_load_shape_silhouette_scores.csv"
)

CLUSTER_ASSIGNMENTS_OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "nrel_load_shape_cluster_assignments.csv"
)

CLUSTER_PROFILES_OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "nrel_load_shape_cluster_profiles.csv"
)

BUILDING_PROFILES_OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "nrel_building_mean_daily_profiles.csv"
)

CLUSTER_SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "nrel_load_shape_cluster_summary.csv"
)


BUILDING_ID_COLUMN = "bldg_id"
TIMESTAMP_COLUMN = "timestamp"
TARGET_COLUMN = "electricity_total_hourly"
ROLE_COLUMN = "sample_role"

RANDOM_STATE = 42

MIN_K = 2
MAX_K = 8
N_INIT = 20


def load_hourly_data() -> pd.DataFrame:

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"NREL hourly modeling dataset not found: "
            f"{INPUT_PATH}"
        )

    df = pd.read_parquet(
        INPUT_PATH,
        columns=[
            BUILDING_ID_COLUMN,
            TIMESTAMP_COLUMN,
            TARGET_COLUMN,
        ],
    )

    required_columns = {
        BUILDING_ID_COLUMN,
        TIMESTAMP_COLUMN,
        TARGET_COLUMN,
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Hourly dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df[TIMESTAMP_COLUMN] = pd.to_datetime(
        df[TIMESTAMP_COLUMN]
    )

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            "Hourly dataset contains missing electricity values."
        )

    return df


def load_sample_roles() -> pd.DataFrame:

    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"NREL modeling sample not found: "
            f"{SAMPLE_PATH}"
        )

    sample = pd.read_csv(
        SAMPLE_PATH,
        usecols=[
            BUILDING_ID_COLUMN,
            ROLE_COLUMN,
        ],
    )

    sample = (
        sample
        .drop_duplicates(
            subset=[
                BUILDING_ID_COLUMN,
            ]
        )
        .copy()
    )

    if sample[
        BUILDING_ID_COLUMN
    ].duplicated().any():
        raise ValueError(
            "Sample file contains duplicate building IDs."
        )

    return sample


def create_mean_daily_profiles(
    hourly: pd.DataFrame,
) -> pd.DataFrame:

    working = hourly.copy()

    working[
        "hour"
    ] = (
        working[
            TIMESTAMP_COLUMN
        ]
        .dt.hour
    )

    profiles = (
        working
        .groupby(
            [
                BUILDING_ID_COLUMN,
                "hour",
            ],
            as_index=False,
        )[
            TARGET_COLUMN
        ]
        .mean()
    )

    profile_wide = (
        profiles
        .pivot(
            index=BUILDING_ID_COLUMN,
            columns="hour",
            values=TARGET_COLUMN,
        )
        .sort_index(axis=1)
    )

    expected_hours = list(
        range(24)
    )

    missing_hours = (
        set(expected_hours)
        - set(profile_wide.columns)
    )

    if missing_hours:
        raise ValueError(
            "Mean daily profiles are missing hours: "
            f"{sorted(missing_hours)}"
        )

    profile_wide = profile_wide[
        expected_hours
    ]

    if profile_wide.isna().any().any():
        raise ValueError(
            "Mean daily profile matrix contains missing values."
        )

    profile_wide.columns = [
        f"hour_{hour:02d}"
        for hour in profile_wide.columns
    ]

    return profile_wide


def normalize_profiles(
    profile_wide: pd.DataFrame,
) -> pd.DataFrame:

    row_means = (
        profile_wide
        .mean(
            axis=1
        )
    )

    zero_mean_buildings = (
        row_means
        == 0
    )

    if zero_mean_buildings.any():
        problem_buildings = (
            profile_wide.index[
                zero_mean_buildings
            ]
            .tolist()
        )

        raise ValueError(
            "Cannot normalize profiles for buildings "
            "with zero mean electricity consumption: "
            f"{problem_buildings}"
        )

    normalized = profile_wide.div(
        row_means,
        axis=0,
    )

    return normalized


def evaluate_k_values(
    x_scaled: np.ndarray,
) -> pd.DataFrame:

    rows = []

    for k in range(
        MIN_K,
        MAX_K + 1,
    ):

        print(
            f"Evaluating k={k}..."
        )

        model = KMeans(
            n_clusters=k,
            n_init=N_INIT,
            random_state=RANDOM_STATE,
        )

        labels = model.fit_predict(
            x_scaled
        )

        silhouette = float(
            silhouette_score(
                x_scaled,
                labels,
            )
        )

        inertia = float(
            model.inertia_
        )

        cluster_counts = (
            pd.Series(
                labels
            )
            .value_counts()
            .sort_index()
        )

        minimum_cluster_size = int(
            cluster_counts.min()
        )

        maximum_cluster_size = int(
            cluster_counts.max()
        )

        rows.append(
            {
                "k": k,
                "silhouette_score": silhouette,
                "inertia": inertia,
                "minimum_cluster_size": (
                    minimum_cluster_size
                ),
                "maximum_cluster_size": (
                    maximum_cluster_size
                ),
            }
        )

        print(
            f"  Silhouette score: "
            f"{silhouette:.6f}"
        )

        print(
            f"  Inertia: "
            f"{inertia:.6f}"
        )

        print(
            f"  Cluster size range: "
            f"{minimum_cluster_size} to "
            f"{maximum_cluster_size}"
        )

    return pd.DataFrame(
        rows
    )


def select_best_k(
    silhouette_results: pd.DataFrame,
) -> int:

    best_row = (
        silhouette_results
        .sort_values(
            [
                "silhouette_score",
                "k",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .iloc[
            0
        ]
    )

    return int(
        best_row[
            "k"
        ]
    )


def fit_final_clustering(
    x_scaled: np.ndarray,
    best_k: int,
) -> tuple[
    KMeans,
    np.ndarray,
]:

    model = KMeans(
        n_clusters=best_k,
        n_init=N_INIT,
        random_state=RANDOM_STATE,
    )

    labels = model.fit_predict(
        x_scaled
    )

    return (
        model,
        labels,
    )


def create_assignment_table(
    normalized_profiles: pd.DataFrame,
    labels: np.ndarray,
    sample_roles: pd.DataFrame,
) -> pd.DataFrame:

    assignments = pd.DataFrame(
        {
            BUILDING_ID_COLUMN: (
                normalized_profiles.index
            ),
            "cluster": labels,
        }
    )

    assignments[
        "cluster"
    ] = (
        assignments[
            "cluster"
        ]
        + 1
    )

    assignments = assignments.merge(
        sample_roles,
        on=BUILDING_ID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    if assignments[
        ROLE_COLUMN
    ].isna().any():
        raise ValueError(
            "Cluster assignments contain buildings "
            "without sample-role metadata."
        )

    return assignments


def create_cluster_profiles(
    normalized_profiles: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:

    profile_table = (
        normalized_profiles
        .reset_index()
        .merge(
            assignments[
                [
                    BUILDING_ID_COLUMN,
                    "cluster",
                ]
            ],
            on=BUILDING_ID_COLUMN,
            how="left",
            validate="one_to_one",
        )
    )

    hour_columns = [
        column
        for column in profile_table.columns
        if column.startswith(
            "hour_"
        )
    ]

    cluster_profiles = (
        profile_table
        .groupby(
            "cluster",
            as_index=False,
        )[
            hour_columns
        ]
        .mean()
    )

    return cluster_profiles


def create_cluster_summary(
    profile_wide: pd.DataFrame,
    normalized_profiles: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:

    original_profiles = (
        profile_wide
        .reset_index()
        .merge(
            assignments,
            on=BUILDING_ID_COLUMN,
            how="left",
            validate="one_to_one",
        )
    )

    normalized_table = (
        normalized_profiles
        .reset_index()
        .merge(
            assignments[
                [
                    BUILDING_ID_COLUMN,
                    "cluster",
                ]
            ],
            on=BUILDING_ID_COLUMN,
            how="left",
            validate="one_to_one",
        )
    )

    hour_columns = [
        f"hour_{hour:02d}"
        for hour in range(
            24
        )
    ]

    rows = []

    for cluster_id, group in assignments.groupby(
        "cluster",
        sort=True,
    ):

        cluster_value = int(
            str(
                cluster_id
            )
        )

        building_ids = set(
            group[
                BUILDING_ID_COLUMN
            ]
            .tolist()
        )

        original_cluster = (
            original_profiles.loc[
                original_profiles[
                    BUILDING_ID_COLUMN
                ].isin(
                    building_ids
                )
            ]
        )

        normalized_cluster = (
            normalized_table.loc[
                normalized_table[
                    BUILDING_ID_COLUMN
                ].isin(
                    building_ids
                )
            ]
        )

        mean_hourly_profile = (
            normalized_cluster[
                hour_columns
            ]
            .mean(
                axis=0
            )
        )

        peak_hour_label = str(
            mean_hourly_profile.idxmax()
        )

        trough_hour_label = str(
            mean_hourly_profile.idxmin()
        )

        peak_hour = int(
            peak_hour_label.rsplit(
                "_",
                maxsplit=1,
            )[
                -1
            ]
        )

        trough_hour = int(
            trough_hour_label.rsplit(
                "_",
                maxsplit=1,
            )[
                -1
            ]
        )

        original_values = (
            original_cluster[
                hour_columns
            ]
            .to_numpy(
                dtype=float
            )
        )

        overall_mean_consumption = float(
            np.mean(
                original_values
            )
        )

        peak_normalized_load = float(
            mean_hourly_profile.max()
        )

        trough_normalized_load = float(
            mean_hourly_profile.min()
        )

        development_count = int(
            (
                group[
                    ROLE_COLUMN
                ]
                == "development"
            )
            .sum()
        )

        held_out_count = int(
            (
                group[
                    ROLE_COLUMN
                ]
                == "held_out"
            )
            .sum()
        )

        if trough_normalized_load != 0:
            peak_to_trough_ratio = float(
                peak_normalized_load
                / trough_normalized_load
            )
        else:
            peak_to_trough_ratio = np.nan

        rows.append(
            {
                "cluster": cluster_value,
                "buildings": int(
                    len(
                        group
                    )
                ),
                "development_buildings": (
                    development_count
                ),
                "held_out_buildings": (
                    held_out_count
                ),
                "mean_hourly_consumption": (
                    overall_mean_consumption
                ),
                "peak_hour": peak_hour,
                "peak_normalized_load": (
                    peak_normalized_load
                ),
                "trough_hour": trough_hour,
                "trough_normalized_load": (
                    trough_normalized_load
                ),
                "peak_to_trough_ratio": (
                    peak_to_trough_ratio
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def create_building_profile_output(
    profile_wide: pd.DataFrame,
    normalized_profiles: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:

    raw_profiles = (
        profile_wide
        .reset_index()
        .copy()
    )

    normalized = (
        normalized_profiles
        .reset_index()
        .copy()
    )

    raw_hour_columns = {
        f"hour_{hour:02d}": (
            f"raw_hour_{hour:02d}"
        )
        for hour in range(
            24
        )
    }

    normalized_hour_columns = {
        f"hour_{hour:02d}": (
            f"normalized_hour_{hour:02d}"
        )
        for hour in range(
            24
        )
    }

    raw_profiles = (
        raw_profiles
        .rename(
            columns=raw_hour_columns
        )
    )

    normalized = (
        normalized
        .rename(
            columns=normalized_hour_columns
        )
    )

    output = (
        assignments
        .merge(
            raw_profiles,
            on=BUILDING_ID_COLUMN,
            how="left",
            validate="one_to_one",
        )
        .merge(
            normalized,
            on=BUILDING_ID_COLUMN,
            how="left",
            validate="one_to_one",
        )
    )

    return output


def main() -> None:

    print(
        "Loading NREL hourly modeling data..."
    )

    hourly = load_hourly_data()

    print(
        f"Hourly rows: "
        f"{len(hourly):,}"
    )

    print(
        f"Buildings: "
        f"{hourly[BUILDING_ID_COLUMN].nunique():,}"
    )

    print()

    print(
        "Loading modeling-sample roles..."
    )

    sample_roles = load_sample_roles()

    print(
        f"Sample-role buildings: "
        f"{len(sample_roles):,}"
    )

    print()

    print(
        "Creating mean 24-hour building profiles..."
    )

    profile_wide = create_mean_daily_profiles(
        hourly
    )

    print(
        f"Profile matrix shape: "
        f"{profile_wide.shape}"
    )

    print()

    print(
        "Normalizing each building profile "
        "by its own mean load..."
    )

    normalized_profiles = normalize_profiles(
        profile_wide
    )

    scaler = StandardScaler()

    x_scaled = scaler.fit_transform(
        normalized_profiles
    )

    print(
        "Standardized normalized profile matrix."
    )

    print()

    print(
        f"Evaluating KMeans clusters from "
        f"k={MIN_K} through k={MAX_K}..."
    )

    silhouette_results = evaluate_k_values(
        x_scaled
    )

    best_k = select_best_k(
        silhouette_results
    )

    print()

    print(
        f"Best k by silhouette score: "
        f"{best_k}"
    )

    best_silhouette = float(
        silhouette_results.loc[
            silhouette_results[
                "k"
            ]
            == best_k,
            "silhouette_score",
        ].iloc[
            0
        ]
    )

    print(
        f"Best silhouette score: "
        f"{best_silhouette:.6f}"
    )

    print()

    print(
        f"Fitting final KMeans model "
        f"with k={best_k}..."
    )

    (
        final_model,
        labels,
    ) = fit_final_clustering(
        x_scaled,
        best_k,
    )

    assignments = create_assignment_table(
        normalized_profiles,
        labels,
        sample_roles,
    )

    cluster_profiles = (
        create_cluster_profiles(
            normalized_profiles,
            assignments,
        )
    )

    cluster_summary = (
        create_cluster_summary(
            profile_wide,
            normalized_profiles,
            assignments,
        )
    )

    building_profiles = (
        create_building_profile_output(
            profile_wide,
            normalized_profiles,
            assignments,
        )
    )

    SILHOUETTE_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    silhouette_results.to_csv(
        SILHOUETTE_OUTPUT_PATH,
        index=False,
    )

    assignments.to_csv(
        CLUSTER_ASSIGNMENTS_OUTPUT_PATH,
        index=False,
    )

    cluster_profiles.to_csv(
        CLUSTER_PROFILES_OUTPUT_PATH,
        index=False,
    )

    building_profiles.to_csv(
        BUILDING_PROFILES_OUTPUT_PATH,
        index=False,
    )

    cluster_summary.to_csv(
        CLUSTER_SUMMARY_OUTPUT_PATH,
        index=False,
    )

    print()

    print(
        "Cluster summary:"
    )

    print(
        cluster_summary.to_string(
            index=False
        )
    )

    print()

    print(
        "Cluster membership counts:"
    )

    print(
        assignments[
            "cluster"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()

    print(
        "NREL load-shape clustering complete."
    )

    print()

    print(
        f"Saved silhouette scores to: "
        f"{SILHOUETTE_OUTPUT_PATH}"
    )

    print(
        f"Saved cluster assignments to: "
        f"{CLUSTER_ASSIGNMENTS_OUTPUT_PATH}"
    )

    print(
        f"Saved cluster profiles to: "
        f"{CLUSTER_PROFILES_OUTPUT_PATH}"
    )

    print(
        f"Saved building profiles to: "
        f"{BUILDING_PROFILES_OUTPUT_PATH}"
    )

    print(
        f"Saved cluster summary to: "
        f"{CLUSTER_SUMMARY_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()