from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

commands = [
    [sys.executable, str(ROOT / 'exercise_1/src/benchmark_dense.py')],
    [sys.executable, str(ROOT / 'exercise_1/src/benchmark_sparse.py')],
    [sys.executable, str(ROOT / 'exercise_3/src/generate_sample_firms.py')],
    [sys.executable, str(ROOT / 'exercise_3/src/simulate_serial.py'), '--input', str(ROOT / 'exercise_3/data/firms_sample.csv')],
    [sys.executable, str(ROOT / 'exercise_4/src/benchmark_kmeans.py')],
]

for cmd in commands:
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True)
