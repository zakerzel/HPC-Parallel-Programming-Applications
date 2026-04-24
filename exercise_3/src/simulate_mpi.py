from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from mpi4py import MPI

from fire_model import BURNED, BURNING, SUSCEPTIBLE, firms_to_grid, load_firms


def local_neighbor_counts(local: np.ndarray) -> np.ndarray:
    counts = np.zeros((local.shape[0] - 2, local.shape[1]), dtype=int)
    core = local[1:-1]
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            shifted = np.roll(local, shift=(dr, dc), axis=(0, 1))[1:-1]
            counts += (shifted == BURNING)
    return counts


def exchange_ghost_rows(comm, local_core: np.ndarray, rank: int, size: int) -> np.ndarray:
    top = np.zeros((1, local_core.shape[1]), dtype=local_core.dtype)
    bottom = np.zeros((1, local_core.shape[1]), dtype=local_core.dtype)
    if rank > 0:
        comm.Sendrecv(local_core[0:1], dest=rank - 1, recvbuf=top, source=rank - 1)
    if rank < size - 1:
        comm.Sendrecv(local_core[-1:], dest=rank + 1, recvbuf=bottom, source=rank + 1)
    return np.vstack([top, local_core, bottom])


def step_local(core: np.ndarray, intensity: np.ndarray, rng: np.random.Generator, with_ghosts: np.ndarray) -> np.ndarray:
    counts = local_neighbor_counts(with_ghosts)
    ignite_prob = np.clip(0.15 * counts + 0.002 * intensity, 0.0, 0.95)
    random_field = rng.random(core.shape)
    next_core = core.copy()
    ignite = (core == SUSCEPTIBLE) & (counts > 0) & (random_field < ignite_prob)
    next_core[ignite] = BURNING
    next_core[core == BURNING] = BURNED
    return next_core


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--grid-size', type=int, default=128)
    parser.add_argument('--steps', type=int, default=20)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        df = load_firms(args.input)
        grid, intensity = firms_to_grid(df, grid_size=args.grid_size)
        row_counts = [args.grid_size // size + (1 if i < args.grid_size % size else 0) for i in range(size)]
        offsets = np.cumsum([0] + row_counts[:-1])
        grid_chunks = [grid[offsets[i]: offsets[i] + row_counts[i], :] for i in range(size)]
        int_chunks = [intensity[offsets[i]: offsets[i] + row_counts[i], :] for i in range(size)]
    else:
        grid_chunks = int_chunks = None

    local_grid = comm.scatter(grid_chunks, root=0)
    local_intensity = comm.scatter(int_chunks, root=0)
    rng = np.random.default_rng(42 + rank)

    comm.Barrier()
    start = time.perf_counter()
    for _ in range(args.steps):
        local_with_ghosts = exchange_ghost_rows(comm, local_grid, rank, size)
        local_grid = step_local(local_grid, local_intensity, rng, local_with_ghosts)
    elapsed = comm.reduce(time.perf_counter() - start, op=MPI.MAX, root=0)

    gathered = comm.gather(local_grid, root=0)
    if rank == 0:
        final_grid = np.vstack(gathered)
        out_dir = Path(__file__).resolve().parents[1] / 'results'
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{'grid_size': args.grid_size, 'steps': args.steps, 'processes': size, 'time_s': elapsed, 'burned_cells': int((final_grid == BURNED).sum())}]).to_csv(
            out_dir / 'mpi_runtime.csv', index=False
        )
        print({'grid_size': args.grid_size, 'steps': args.steps, 'processes': size, 'time_s': elapsed})


if __name__ == '__main__':
    main()
