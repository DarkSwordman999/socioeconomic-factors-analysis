"""Типология регионов: кластеризация по социально-экономическому профилю."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["gdp_real", "income_real", "poverty", "life_exp", "housing",
            "unemployment", "doctors", "urban", "elderly"]
LOG_FEATURES = ["gdp_real", "income_real"]
SEED = 42
N_CLUSTERS = 3   # по силуэту лучше k=2, но оно лишь отделяет бедные республики от остальных


def _matrix(frame: pd.DataFrame) -> np.ndarray:
    data = frame[FEATURES].copy()
    data[LOG_FEATURES] = np.log(data[LOG_FEATURES])
    return StandardScaler().fit_transform(data)


def cluster_regions(panel: pd.DataFrame, year: int, k: int = N_CLUSTERS, k_range=range(2, 8)):
    """Делит регионы на k кластеров методом k-средних.

    Возвращает (регионы с номером кластера и координатами на плоскости главных
    компонент, средние профили кластеров, силуэт для каждого k из k_range).
    Кластеры нумеруются по убыванию среднего реального дохода.
    """
    frame = panel[panel["year"] == year].dropna(subset=FEATURES).reset_index(drop=True)
    matrix = _matrix(frame)

    scores = []
    for n in k_range:
        labels = KMeans(n_clusters=n, n_init=20, random_state=SEED).fit_predict(matrix)
        scores.append({"k": n, "silhouette": silhouette_score(matrix, labels)})
    scores = pd.DataFrame(scores)

    labels = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit_predict(matrix)
    order = frame.groupby(labels)["income_real"].mean().sort_values(ascending=False).index
    rank = {old: new + 1 for new, old in enumerate(order)}

    points = PCA(n_components=2, random_state=SEED).fit_transform(matrix)
    result = frame[["region", "district"] + FEATURES].copy()
    result.insert(2, "cluster", [rank[label] for label in labels])
    result["pc1"], result["pc2"] = points[:, 0], points[:, 1]

    profiles = result.groupby("cluster")[FEATURES].mean()
    profiles.insert(0, "regions", result.groupby("cluster").size())
    return result.sort_values(["cluster", "region"]).reset_index(drop=True), profiles.reset_index(), scores
