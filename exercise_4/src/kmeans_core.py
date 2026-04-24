from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.datasets import fetch_covtype, make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass
class KMeansResult:
    centroids: np.ndarray
    labels: np.ndarray
    inertia: float
    iterations: int
    runtime_s: float
    silhouette: float | None


def load_dataset(samples: int = 20000, features: int = 20, use_covtype: bool = False, seed: int = 42) -> np.ndarray:
    if use_covtype:
        data = fetch_covtype(return_X_y=True)[0]
        data = data[:samples]
    else:
        data, _ = make_blobs(n_samples=samples, n_features=features, centers=5, random_state=seed)
    return StandardScaler().fit_transform(data)


def initialize_centroids(x: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=k, replace=False)
    return x[idx].copy()


def assign_clusters(x: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dists = np.linalg.norm(x[:, None, :] - centroids[None, :, :], axis=2)
    labels = np.argmin(dists, axis=1)
    min_dists = dists[np.arange(x.shape[0]), labels]
    return labels, min_dists


def update_centroids(x: np.ndarray, labels: np.ndarray, k: int, old_centroids: np.ndarray) -> np.ndarray:
    new = old_centroids.copy()
    for i in range(k):
        members = x[labels == i]
        if len(members) > 0:
            new[i] = members.mean(axis=0)
    return new


def run_serial_kmeans(x: np.ndarray, k: int, max_iter: int = 50, tol: float = 1e-4, seed: int = 42) -> KMeansResult:
    centroids = initialize_centroids(x, k, seed)
    start = time.perf_counter()
    labels = np.zeros(x.shape[0], dtype=int)
    inertia = 0.0
    iterations = 0
    for it in range(1, max_iter + 1):
        labels, min_dists = assign_clusters(x, centroids)
        new_centroids = update_centroids(x, labels, k, centroids)
        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        inertia = float((min_dists ** 2).sum())
        iterations = it
        if shift < tol:
            break
    runtime = time.perf_counter() - start
    silhouette = None
    if 1 < k < x.shape[0]:
        sample = x[: min(1500, x.shape[0])]
        sample_labels = labels[: sample.shape[0]]
        if len(np.unique(sample_labels)) > 1:
            silhouette = float(silhouette_score(sample, sample_labels))
    return KMeansResult(centroids, labels, inertia, iterations, runtime, silhouette)
