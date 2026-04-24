# Exercise 4 – Parallel K-Means Clustering

## Goal
Implement serial and MPI-based K-Means on the Covertype dataset.

## Minimum implementations
- Serial baseline
- MPI distributed version using collective communication

## Suggested experiments
- Test multiple values of k
- Test multiple process counts
- Measure runtime per iteration and total runtime
- Compare convergence behavior and clustering stability

## Folder notes
- `serial/`: serial baseline
- `parallel_mpi/`: MPI version
- `data/`: dataset or download instructions
- `outputs/`: plots, tables, timing logs
