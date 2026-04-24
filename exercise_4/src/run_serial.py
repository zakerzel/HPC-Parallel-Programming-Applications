import argparse
from pathlib import Path

import pandas as pd

from kmeans_core import load_dataset, run_serial_kmeans


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=20000)
    parser.add_argument('--features', type=int, default=20)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--use-covtype', action='store_true')
    args = parser.parse_args()

    x = load_dataset(samples=args.samples, features=args.features, use_covtype=args.use_covtype)
    result = run_serial_kmeans(x, args.k)
    out_dir = Path(__file__).resolve().parents[1] / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {
            'variant': 'serial',
            'samples': args.samples,
            'features': x.shape[1],
            'k': args.k,
            'iterations': result.iterations,
            'runtime_s': result.runtime_s,
            'inertia': result.inertia,
            'silhouette': result.silhouette,
        }
    ]).to_csv(out_dir / 'serial_results.csv', index=False)
    print(result)


if __name__ == '__main__':
    main()
