from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from matmul_core import benchmark_sparse


def main() -> None:
    configs = [(256, 0.01), (256, 0.05), (512, 0.01)]
    rows = []
    for n, density in configs:
        rows.extend(benchmark_sparse(n, density))
    df = pd.DataFrame(rows)
    out_dir = Path(__file__).resolve().parents[1] / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'sparse_benchmarks.csv', index=False)

    plt.figure(figsize=(8, 5))
    for method, group in df.groupby('method'):
        label = method.replace('_', ' ')
        plt.plot(group['density'].astype(str) + '\n' + group['n'].astype(str), group['time_s'], marker='o', label=label)
    plt.xlabel('density / n')
    plt.ylabel('Runtime (s)')
    plt.title('Sparse matrix multiplication comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'sparse_runtime_plot.png', dpi=200)
    plt.close()

    print(df)


if __name__ == '__main__':
    main()
