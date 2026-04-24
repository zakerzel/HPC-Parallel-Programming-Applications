# Exercise 1 – Parallel Matrix Multiplication

## Goal
Compare a serial matrix multiplication baseline against multiple parallel strategies.

## Minimum implementations
- Serial dense baseline
- Multiprocessing by rows
- Multiprocessing by columns
- Multiprocessing by blocks/quadrants
- MPI distributed version
- Strassen-based or hybrid analysis

## Suggested experiments
- Dense sizes: 128, 256, 512, 1024
- Sparse matrices: at least 2 real matrices from SuiteSparse
- Metrics: runtime, speedup, efficiency, correctness checks

## Folder notes
- `serial/`: serial baseline
- `parallel_mp/`: multiprocessing versions
- `parallel_mpi/`: MPI version
- `data/`: saved sparse matrices or download scripts
- `outputs/`: benchmark tables, plots, logs
