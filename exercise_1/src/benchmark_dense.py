from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from matmul_core import benchmark_dense


def main() -> None:
    sizes = [64, 128, 256]
    workers = 4
    rows = []
    for n in sizes:
        rows.extend(benchmark_dense(n, workers=workers))
    df = pd.DataFrame(rows)
    out_dir = Path(__file__).resolve().parents[1] / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'dense_benchmarks.csv', index=False)

    serial = df[df['method'] == 'serial'][['n', 'time_s']].rename(columns={'time_s': 'serial_time'})
    plot_df = df.merge(serial, on='n')
    plot_df['speedup'] = plot_df['serial_time'] / plot_df['time_s']

    plt.figure(figsize=(8, 5))
    for method, group in plot_df.groupby('method'):
        plt.plot(group['n'], group['time_s'], marker='o', label=method)
    plt.xlabel('Matrix size (n x n)')
    plt.ylabel('Runtime (s)')
    plt.title('Dense matrix multiplication runtimes')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'dense_runtime_plot.png', dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    speed_methods = plot_df[plot_df['method'] != 'serial']
    for method, group in speed_methods.groupby('method'):
        plt.plot(group['n'], group['speedup'], marker='o', label=method)
    plt.xlabel('Matrix size (n x n)')
    plt.ylabel('Speedup versus serial')
    plt.title('Dense speedup by method')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'dense_speedup_plot.png', dpi=200)
    plt.close()

    print(df)


if __name__ == '__main__':
    main()
