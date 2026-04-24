# Quick Execution Notes

This file is a small reminder of the commands to run each exercise.

## Windows + Git Bash

Activate environment:

```bash
source .venv/Scripts/activate
```

## MPI examples

If `mpiexec` is available:

```bash
mpiexec -n 4 python exercise_1/parallel_mpi/matmul_mpi.py
mpiexec -n 4 python exercise_3/parallel_mpi/forest_fire_mpi.py
mpiexec -n 4 python exercise_4/parallel_mpi/kmeans_mpi.py
```

If MPI is not configured yet, implement and test the serial and multiprocessing parts first.
