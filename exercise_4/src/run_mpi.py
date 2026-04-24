from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from mpi4py import MPI

from kmeans_core import initialize_centroids, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=20000)
    parser.add_argument('--features', type=int, default=20)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--max-iter', type=int, default=50)
    parser.add_argument('--tol', type=float, default=1e-4)
    parser.add_argument('--use-covtype', action='store_true')
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        x = load_dataset(samples=args.samples, features=args.features, use_covtype=args.use_covtype)
        chunks = np.array_split(x, size, axis=0)
        centroids = initialize_centroids(x, args.k)
        n_features = x.shape[1]
    else:
        chunks = None
        centroids = None
        n_features = None

    local_x = comm.scatter(chunks, root=0)
    centroids = comm.bcast(centroids, root=0)
    n_features = comm.bcast(n_features, root=0)

    comm.Barrier()
    start = time.perf_counter()
    iterations = 0
    inertia = None
    for it in range(1, args.max_iter + 1):
        dists = np.linalg.norm(local_x[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        min_dists = dists[np.arange(local_x.shape[0]), labels]
        local_inertia = float((min_dists ** 2).sum())

        local_sums = np.zeros((args.k, n_features), dtype=np.float64)
        local_counts = np.zeros(args.k, dtype=np.int64)
        for i in range(args.k):
            members = local_x[labels == i]
            if len(members) > 0:
                local_sums[i] = members.sum(axis=0)
                local_counts[i] = len(members)

        global_sums = np.zeros_like(local_sums)
        global_counts = np.zeros_like(local_counts)
        inertia = comm.allreduce(local_inertia, op=MPI.SUM)
        comm.Allreduce(local_sums, global_sums, op=MPI.SUM)
        comm.Allreduce(local_counts, global_counts, op=MPI.SUM)

        new_centroids = centroids.copy()
        mask = global_counts > 0
        new_centroids[mask] = global_sums[mask] / global_counts[mask, None]
        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        iterations = it
        if shift < args.tol:
            break

    elapsed = comm.reduce(time.perf_counter() - start, op=MPI.MAX, root=0)
    if rank == 0:
        out_dir = Path(__file__).resolve().parents[1] / 'results'
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([
            {
                'variant': 'mpi',
                'samples': args.samples,
                'features': n_features,
                'k': args.k,
                'iterations': iterations,
                'runtime_s': elapsed,
                'inertia': inertia,
                'processes': size,
            }
        ]).to_csv(out_dir / 'mpi_results.csv', index=False)
        print({'processes': size, 'iterations': iterations, 'runtime_s': elapsed, 'inertia': inertia})


if __name__ == '__main__':
    main()
