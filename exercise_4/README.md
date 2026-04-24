# Exercise 4 - Parallel K-Means Clustering

This exercise contains a serial NumPy baseline and an MPI implementation of K-Means. The default runner uses a reproducible synthetic dataset. If the Covertype dataset is available locally, the helper can load it instead.

## Files

- `src/kmeans_core.py`: shared serial utilities
- `src/run_serial.py`: serial baseline
- `src/run_mpi.py`: MPI collective implementation
- `src/benchmark_kmeans.py`: small local benchmarks and plots

## Run

```bash
python src/run_serial.py --samples 20000 --features 20 --k 5
mpiexec -n 4 python src/run_mpi.py --samples 20000 --features 20 --k 5
python src/benchmark_kmeans.py
```
