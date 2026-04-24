# HPC Unit 3 Final Assignment

This repository contains the four required exercises for the Unit 3 final assignment in High Performance Computing.

## Repository structure

- `exercise_1/` Parallel matrix multiplication
- `exercise_2/` Parallel cell image processing and morphological characterization
- `exercise_3/` Forest fire cellular automaton driven by NASA FIRMS data
- `exercise_4/` Parallel K-Means clustering on the Covertype dataset
- `docs/` IEEE-style report and generated assets
- `scripts/` helper scripts for reproducing the full workflow

## Environment

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

MPI examples require an MPI runtime such as MS-MPI or OpenMPI plus `mpi4py`.

## Exercise 1 - Parallel Matrix Multiplication

Serial baseline, multiprocessing versions by rows, columns, and blocks, plus an MPI implementation and a Strassen-based method.

```bash
python exercise_1/src/benchmark_dense.py
python exercise_1/src/benchmark_sparse.py
mpiexec -n 4 python exercise_1/src/matmul_mpi.py
```

Outputs are written to `exercise_1/results/`.

## Exercise 2 - Parallel Cell Image Processing

The code supports either the DIC-C2DH-HeLa dataset or any folder of grayscale microscopy images.

```bash
python exercise_2/src/inspect_dataset.py --input exercise_2/data/DIC-C2DH-HeLa
python exercise_2/src/process_images.py --input exercise_2/data/DIC-C2DH-HeLa/01 --mode serial
python exercise_2/src/process_images.py --input exercise_2/data/DIC-C2DH-HeLa/01 --mode parallel --workers 4
```

Outputs are written to `exercise_2/results/`.

## Exercise 3 - Forest Fire Cellular Automaton

The repository includes a serial simulator and an MPI domain-decomposed implementation.

```bash
python exercise_3/src/generate_sample_firms.py
python exercise_3/src/simulate_serial.py --input exercise_3/data/firms_sample.csv
mpiexec -n 4 python exercise_3/src/simulate_mpi.py --input exercise_3/data/firms_sample.csv
```

Outputs are written to `exercise_3/results/`.

## Exercise 4 - Parallel K-Means Clustering

The exercise includes a serial NumPy baseline and an MPI version using collective communication to aggregate partial sums and counts.

```bash
python exercise_4/src/run_serial.py --samples 20000 --features 20 --k 5
mpiexec -n 4 python exercise_4/src/run_mpi.py --samples 20000 --features 20 --k 5
```

The helper script below uses either the built-in synthetic generator or the Covertype dataset if it is available locally.

```bash
python exercise_4/src/benchmark_kmeans.py
```

## Full report

The Overleaf-ready LaTeX source is in `docs/report.tex`. After updating local timings if needed:

```bash
cd docs
pdflatex report.tex
pdflatex report.tex
```

The resulting PDF must be stored as `docs/report.pdf`.
