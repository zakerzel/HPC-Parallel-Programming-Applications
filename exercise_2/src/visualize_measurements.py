from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / 'results'
    serial_path = out_dir / 'summary_serial.csv'
    parallel_path = out_dir / 'summary_parallel.csv'
    if not serial_path.exists():
        raise FileNotFoundError('Run process_images.py first.')
    serial = pd.read_csv(serial_path)
    timing_frames = []
    for path in out_dir.glob('timing_*.csv'):
        timing_frames.append(pd.read_csv(path))
    timing = pd.concat(timing_frames, ignore_index=True) if timing_frames else pd.DataFrame()

    plt.figure(figsize=(8, 5))
    plt.bar(serial['image'], serial['detected_cells'])
    plt.xticks(rotation=90)
    plt.ylabel('Detected cells')
    plt.title('Detected cells per image')
    plt.tight_layout()
    plt.savefig(out_dir / 'cells_per_image.png', dpi=200)
    plt.close()

    if not timing.empty:
        plt.figure(figsize=(6, 4))
        plt.bar(timing['mode'], timing['time_s'])
        plt.ylabel('Runtime (s)')
        plt.title('Serial versus parallel runtime')
        plt.tight_layout()
        plt.savefig(out_dir / 'runtime_comparison.png', dpi=200)
        plt.close()


if __name__ == '__main__':
    main()
