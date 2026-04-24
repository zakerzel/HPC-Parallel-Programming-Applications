from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NON_BURNABLE = 0
SUSCEPTIBLE = 1
BURNING = 2
BURNED = 3


def load_firms(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def firms_to_grid(df: pd.DataFrame, grid_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    lat = df['latitude'].to_numpy()
    lon = df['longitude'].to_numpy()
    frp = df['frp'].to_numpy()

    lat_scaled = ((lat - lat.min()) / (lat.max() - lat.min() + 1e-12) * (grid_size - 1)).astype(int)
    lon_scaled = ((lon - lon.min()) / (lon.max() - lon.min() + 1e-12) * (grid_size - 1)).astype(int)

    state = np.ones((grid_size, grid_size), dtype=np.int8)
    intensity = np.zeros_like(state, dtype=np.float64)
    for r, c, f in zip(lat_scaled, lon_scaled, frp):
        state[r, c] = BURNING
        intensity[r, c] = max(intensity[r, c], f)
    return state, intensity


def count_burning_neighbors(grid: np.ndarray) -> np.ndarray:
    padded = np.pad(grid == BURNING, 1)
    counts = np.zeros_like(grid, dtype=int)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            counts += padded[1 + dr : 1 + dr + grid.shape[0], 1 + dc : 1 + dc + grid.shape[1]]
    return counts


def step(grid: np.ndarray, intensity: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    counts = count_burning_neighbors(grid)
    next_grid = grid.copy()
    ignite_prob = np.clip(0.15 * counts + 0.002 * intensity, 0.0, 0.95)
    random_field = rng.random(grid.shape)
    ignite = (grid == SUSCEPTIBLE) & (counts > 0) & (random_field < ignite_prob)
    next_grid[ignite] = BURNING
    next_grid[grid == BURNING] = BURNED
    return next_grid


def simulate(grid: np.ndarray, intensity: np.ndarray, steps: int, seed: int = 42) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    states = [grid.copy()]
    current = grid.copy()
    for _ in range(steps):
        current = step(current, intensity, rng)
        states.append(current.copy())
    return states


def save_snapshots(states: list[np.ndarray], out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap('viridis', 4)
    images = []
    for i, state in enumerate(states):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(state, cmap=cmap, vmin=0, vmax=3)
        ax.set_title(f'{prefix} step {i}')
        ax.set_axis_off()
        path = out_dir / f'{prefix}_step_{i:03d}.png'
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)
        images.append(imageio.imread(path))
    imageio.mimsave(out_dir / f'{prefix}.gif', images, duration=0.4)
