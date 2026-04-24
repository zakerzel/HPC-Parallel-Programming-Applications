import argparse
import time
from pathlib import Path

import pandas as pd

from fire_model import firms_to_grid, load_firms, save_snapshots, simulate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--grid-size', type=int, default=128)
    parser.add_argument('--steps', type=int, default=20)
    args = parser.parse_args()

    df = load_firms(args.input)
    grid, intensity = firms_to_grid(df, grid_size=args.grid_size)
    start = time.perf_counter()
    states = simulate(grid, intensity, steps=args.steps)
    elapsed = time.perf_counter() - start

    out_dir = Path(__file__).resolve().parents[1] / 'results'
    save_snapshots(states[:: max(1, len(states)//5)], out_dir, 'serial_fire')
    pd.DataFrame([{'grid_size': args.grid_size, 'steps': args.steps, 'time_s': elapsed}]).to_csv(out_dir / 'serial_runtime.csv', index=False)
    print({'grid_size': args.grid_size, 'steps': args.steps, 'time_s': elapsed})


if __name__ == '__main__':
    main()
