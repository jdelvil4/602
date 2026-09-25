from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.config import RANDOM_STATE


INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "uci_hourly.csv"
)

OUTPUT_TABLE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
)

OUTPUT_FIGURE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
)

CLUSTERED_DATA_PATH = (
    OUTPUT_TABLE_DIR
    / "uci_clustered_hourly.csv"
)

CLUSTER_SELECTION_PATH = (
    OUTPUT_TABLE_DIR
    / "uci_kmeans_cluster_selection.csv"
)

CLUSTER_PROFILE_PATH = (
    OUTPUT_TABLE_DIR
    / "uci_kmeans_cluster_profiles.csv"
)

CLUSTER_SELECTION_FIGURE_PATH = (
    OUTPUT_FIGURE_DIR
    / "uci_kmeans_cluster_selection.png"
)

CONSUMPTION_TEMPERATURE_FIGURE_PATH = (
    OUTPUT_FIGURE_DIR
    / "uci_kmeans_consumption_vs_temperature.png"
)

CLUSTER_SIZE_FIGURE_PATH = (
    OUTPUT_FIGURE_DIR
    / "uci_kmeans_cluster_sizes.png"
)


MIN_K = 2
MAX_K = 8


CLUSTER_FEATURES = [
    "Appliances",
    "lights",
    "T_out",
    "RH_out",
    "Windspeed",
    "hour_sin",
    "hour_cos",
    "is_weekend",
]


def load_hourly_data(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"Hourly UCI dataset not found: {path}"
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

    return df


def add_cluster_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df.loc[:, "hour"] = (
        df.loc[:, "date"]
        .dt.hour
    )

    df.loc[:, "day_of_week"] = (
        df.loc[:, "date"]
        .dt.dayofweek
    )

    df.loc[:, "is_weekend"] = (
        df.loc[:, "day_of_week"]
        .ge(5)
        .astype(int)
    )

    df.loc[:, "hour_sin"] = np.sin(
        2
        * np.pi
        * df.loc[:, "hour"]
        / 24
    )

    df.loc[:, "hour_cos"] = np.cos(
        2
        * np.pi
        * df.loc[:, "hour"]
        / 24
    )

    return df


def validate_cluster_features(
    df: pd.DataFrame,
) -> None:

    missing_columns = [
        column
        for column in CLUSTER_FEATURES
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing clustering features: "
            f"{missing_columns}"
        )

    cluster_frame = df.filter(
        items=CLUSTER_FEATURES
    )

    missing_values = (
        cluster_frame
        .isna()
        .sum()
        .sum()
    )

    if int(missing_values) != 0:
        raise ValueError(
            "Missing values exist in clustering features."
        )


def scale_features(
    df: pd.DataFrame,
) -> tuple[StandardScaler, pd.DataFrame]:

    cluster_frame = df.filter(
        items=CLUSTER_FEATURES
    )

    scaler = StandardScaler()

    scaled_array = scaler.fit_transform(
        cluster_frame
    )

    scaled_df = pd.DataFrame(
        scaled_array,
        columns=CLUSTER_FEATURES,
        index=df.index,
    )

    return scaler, scaled_df


def evaluate_cluster_counts(
    scaled_df: pd.DataFrame,
) -> pd.DataFrame:

    results: list[dict[str, float | int]] = []

    for k in range(
        MIN_K,
        MAX_K + 1,
    ):

        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=20,
        )

        labels = model.fit_predict(
            scaled_df
        )

        silhouette = silhouette_score(
            scaled_df,
            labels,
        )

        results.append(
            {
                "k": k,
                "inertia": float(
                    model.inertia_
                ),
                "silhouette_score": float(
                    silhouette
                ),
            }
        )

        print(
            f"k={k}: "
            f"inertia={model.inertia_:.3f}, "
            f"silhouette={silhouette:.4f}"
        )

    return pd.DataFrame(
        results
    )


def select_best_k(
    selection_df: pd.DataFrame,
) -> int:

    silhouette_values = (
        selection_df
        .loc[:, "silhouette_score"]
        .to_numpy()
    )

    best_position = int(
        np.argmax(
            silhouette_values
        )
    )

    k_values = (
        selection_df
        .loc[:, "k"]
        .to_numpy()
    )

    best_k_value = k_values[
        best_position
    ]

    return int(
        best_k_value
    )


def fit_final_kmeans(
    scaled_df: pd.DataFrame,
    best_k: int,
) -> KMeans:

    model = KMeans(
        n_clusters=best_k,
        random_state=RANDOM_STATE,
        n_init=20,
    )

    model.fit(
        scaled_df
    )

    return model


def create_cluster_profiles(
    df: pd.DataFrame,
) -> pd.DataFrame:

    profile_columns = [
        "Appliances",
        "lights",
        "T_out",
        "RH_out",
        "Windspeed",
        "hour",
        "is_weekend",
    ]

    profile_source = df.filter(
        items=[
            "cluster",
            *profile_columns,
        ]
    )

    profiles = (
        profile_source
        .groupby(
            "cluster",
            as_index=False,
        )
        .mean(
            numeric_only=True
        )
    )

    cluster_sizes = (
        df
        .groupby(
            "cluster"
        )
        .size()
        .rename(
            "observation_count"
        )
        .reset_index()
    )

    profiles = profiles.merge(
        cluster_sizes,
        on="cluster",
        how="left",
    )

    observation_count = (
        profiles.loc[
            :,
            "observation_count",
        ]
    )

    profiles.loc[
        :,
        "share_percent",
    ] = (
        observation_count
        / len(df)
        * 100
    )

    return profiles


def save_cluster_selection_plot(
    selection_df: pd.DataFrame,
) -> None:

    k_values = selection_df.loc[
        :,
        "k",
    ]

    silhouette_values = selection_df.loc[
        :,
        "silhouette_score",
    ]

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        k_values,
        silhouette_values,
        marker="o",
    )

    plt.xlabel(
        "Number of Clusters (k)"
    )

    plt.ylabel(
        "Silhouette Score"
    )

    plt.title(
        "UCI K-Means Cluster Selection"
    )

    plt.xticks(
        k_values
    )

    plt.tight_layout()

    plt.savefig(
        CLUSTER_SELECTION_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_consumption_temperature_plot(
    df: pd.DataFrame,
) -> None:

    outdoor_temperature = df.loc[
        :,
        "T_out",
    ]

    appliances = df.loc[
        :,
        "Appliances",
    ]

    clusters = df.loc[
        :,
        "cluster",
    ]

    plt.figure(
        figsize=(10, 7)
    )

    scatter = plt.scatter(
        outdoor_temperature,
        appliances,
        c=clusters,
        alpha=0.6,
    )

    plt.xlabel(
        "Outdoor Temperature"
    )

    plt.ylabel(
        "Appliance Energy Consumption"
    )

    plt.title(
        "UCI K-Means Clusters: Consumption vs Outdoor Temperature"
    )

    plt.colorbar(
        scatter,
        label="Cluster",
    )

    plt.tight_layout()

    plt.savefig(
        CONSUMPTION_TEMPERATURE_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_cluster_size_plot(
    profile_df: pd.DataFrame,
) -> None:

    cluster_values = (
        profile_df.loc[
            :,
            "cluster",
        ]
        .astype(str)
    )

    observation_counts = (
        profile_df.loc[
            :,
            "observation_count",
        ]
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        cluster_values,
        observation_counts,
    )

    plt.xlabel(
        "Cluster"
    )

    plt.ylabel(
        "Number of Hourly Observations"
    )

    plt.title(
        "UCI K-Means Cluster Sizes"
    )

    plt.tight_layout()

    plt.savefig(
        CLUSTER_SIZE_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def main() -> None:

    OUTPUT_TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_hourly_data(
        INPUT_PATH
    )

    df = add_cluster_features(
        df
    )

    validate_cluster_features(
        df
    )

    print(
        f"Hourly observations: "
        f"{len(df):,}"
    )

    print()

    print(
        "Clustering features:"
    )

    for feature in CLUSTER_FEATURES:
        print(
            f"  - {feature}"
        )

    print()

    _, scaled_df = scale_features(
        df
    )

    selection_df = (
        evaluate_cluster_counts(
            scaled_df
        )
    )

    best_k = select_best_k(
        selection_df
    )

    print()

    print(
        f"Selected cluster count: "
        f"k={best_k}"
    )

    final_model = fit_final_kmeans(
        scaled_df,
        best_k,
    )

    df.loc[
        :,
        "cluster",
    ] = final_model.labels_

    profiles_df = (
        create_cluster_profiles(
            df
        )
    )

    selection_df.to_csv(
        CLUSTER_SELECTION_PATH,
        index=False,
    )

    profiles_df.to_csv(
        CLUSTER_PROFILE_PATH,
        index=False,
    )

    df.to_csv(
        CLUSTERED_DATA_PATH,
        index=False,
    )

    save_cluster_selection_plot(
        selection_df
    )

    save_consumption_temperature_plot(
        df
    )

    save_cluster_size_plot(
        profiles_df
    )

    print()

    print(
        "Cluster profiles"
    )

    print(
        "================"
    )

    print(
        profiles_df.to_string(
            index=False
        )
    )

    print()

    print(
        "Saved tables:"
    )

    print(
        CLUSTER_SELECTION_PATH
    )

    print(
        CLUSTER_PROFILE_PATH
    )

    print(
        CLUSTERED_DATA_PATH
    )

    print()

    print(
        "Saved figures:"
    )

    print(
        CLUSTER_SELECTION_FIGURE_PATH
    )

    print(
        CONSUMPTION_TEMPERATURE_FIGURE_PATH
    )

    print(
        CLUSTER_SIZE_FIGURE_PATH
    )


if __name__ == "__main__":
    main()