# HPC Unit 3 Final Assignment

This repository contains the solutions for the **High Performance Computing – Unit 3 Final Assignment**. The project is organized into four exercises that compare **serial baselines** against **parallel implementations** using `multiprocessing`, `mpi4py`, or both.

## Objective

The goal of this assignment is to implement, measure, and analyze how parallelism affects performance in four different application domains:

1. Parallel Matrix Multiplication
2. Parallel Cell Image Processing and Morphological Characterization
3. Forest Fire Cellular Automaton Driven by NASA FIRMS Data
4. Parallel K-Means Clustering

For every exercise, the repository is structured so the reviewer can easily identify:

- the **serial baseline**
- the **parallel implementation**
- the **commands used to reproduce experiments**
- the **evidence and outputs** used in the report

## Repository Structure

```text
hpc_final_repo/
├── README.md
├── requirements.txt
├── exercise_1/
│   ├── README.md
│   ├── serial/
│   ├── parallel_mp/
│   ├── parallel_mpi/
│   ├── data/
│   ├── outputs/
│   └── notebooks/
├── exercise_2/
│   ├── README.md
│   ├── serial/
│   ├── parallel_mp/
│   ├── data/
│   ├── outputs/
│   └── notebooks/
├── exercise_3/
│   ├── README.md
│   ├── serial/
│   ├── parallel_mpi/
│   ├── data/
│   ├── outputs/
│   └── notebooks/
├── exercise_4/
│   ├── README.md
│   ├── serial/
│   ├── parallel_mpi/
│   ├── data/
│   ├── outputs/
│   └── notebooks/
├── docs/
│   ├── report.tex
│   └── assets/
└── scripts/
    └── run_examples.md
```

## Software Requirements

Recommended environment:

- Python 3.10+
- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `scikit-learn`
- `scikit-image`
- `opencv-python`
- `mpi4py`
- `jupyter`
- `cellpose` (optional, for Exercise 2)

For MPI-based exercises:

- MPI runtime installed, for example **MS-MPI**, **OpenMPI**, or **MPICH**
- `mpi4py` correctly linked to the local MPI installation

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
# On Windows Git Bash:
# source .venv/Scripts/activate
pip install -r requirements.txt
```

## Reproducibility Notes

- Fix random seeds whenever synthetic data or randomized initialization is used.
- Record hardware information and execution environment in the report.
- Save benchmark tables, figures, logs, and exported outputs inside each exercise folder or `docs/assets/`.
- Keep serial and parallel outputs separate to simplify comparisons.

## Exercise Overview

### Exercise 1 – Parallel Matrix Multiplication

Compare serial dense multiplication against:
- multiprocessing by rows
- multiprocessing by columns
- block or quadrant decomposition
- an MPI version
- a Strassen-inspired advanced approach

Also test at least two sparse matrices and discuss how sparsity changes the behavior.

### Exercise 2 – Parallel Cell Image Processing and Morphological Characterization

Build a serial and parallel pipeline to:
- load cell images
- segment cells
- measure bounding boxes, area, major axis, and minor axis
- summarize results per image
- compare serial vs parallel runtimes

### Exercise 3 – Forest Fire Cellular Automaton Driven by NASA FIRMS Data

Build a forest fire simulation using real hotspot detections:
- serial cellular automaton
- MPI parallel version with domain decomposition
- visualizations of temporal evolution
- runtime comparison and scientific discussion

### Exercise 4 – Parallel K-Means Clustering

Implement K-Means using:
- a serial baseline
- an MPI version with distributed data
- collective communication for local statistics aggregation
- timing, convergence, and scalability analysis

## Expected Outputs

Each exercise should generate:

- source code for serial and parallel implementations
- benchmark logs or tables
- plots or figures when useful
- output files needed by the report

The final report must be saved as:

```text
docs/report.pdf
```

## Suggested Commands

### Exercise 1

```bash
python exercise_1/serial/matmul_serial.py
python exercise_1/parallel_mp/matmul_rows_mp.py
python exercise_1/parallel_mp/matmul_cols_mp.py
python exercise_1/parallel_mp/matmul_blocks_mp.py
mpiexec -n 4 python exercise_1/parallel_mpi/matmul_mpi.py
```

### Exercise 2

```bash
python exercise_2/serial/cell_pipeline_serial.py
python exercise_2/parallel_mp/cell_pipeline_mp.py
```

### Exercise 3

```bash
python exercise_3/serial/forest_fire_serial.py
mpiexec -n 4 python exercise_3/parallel_mpi/forest_fire_mpi.py
```

### Exercise 4

```bash
python exercise_4/serial/kmeans_serial.py
mpiexec -n 4 python exercise_4/parallel_mpi/kmeans_mpi.py
```

## Final Note

This repository is intentionally organized to make the comparison between **serial** and **parallel** solutions straightforward. Every implementation should be accompanied by timing results and a short analysis explaining whether the chosen parallel strategy provides a real benefit and why.
