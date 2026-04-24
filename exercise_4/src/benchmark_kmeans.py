from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from kmeans_core import load_dataset, run_serial_kmeans


def main() -> None:
    configs = [(3000, 10, 3), (6000, 10, 5), (10000, 20, 5)]
    rows = []
    for samples, features, k in configs:
        x = load_dataset(samples=samples, features=features, use_covtype=False)
        result = run_serial_kmeans(x, k)
        rows.append({
            'variant': 'serial',
            'samples': samples,
            'features': features,
            'k': k,
            'iterations': result.iterations,
            'runtime_s': result.runtime_s,
            'inertia': result.inertia,
            'silhouette': result.silhouette,
        })
    df = pd.DataFrame(rows)
    out_dir = Path(__file__).resolve().parents[1] / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'benchmark_serial.csv', index=False)

    plt.figure(figsize=(7, 4))
    plt.plot(df['samples'], df['runtime_s'], marker='o')
    plt.xlabel('Samples')
    plt.ylabel('Runtime (s)')
    plt.title('Serial K-Means benchmark')
    plt.tight_layout()
    plt.savefig(out_dir / 'serial_runtime_plot.png', dpi=200)
    plt.close()


if __name__ == '__main__':
    main()
