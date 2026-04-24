# Exercise 3 - Forest Fire Cellular Automaton Driven by NASA FIRMS Data

This exercise converts hotspot detections into a regular grid and simulates fire propagation in serial and MPI modes.

## Files

- `src/generate_sample_firms.py`: generates a small reproducible FIRMS-like CSV for offline testing
- `src/fire_model.py`: grid creation and cellular automaton rules
- `src/simulate_serial.py`: serial baseline
- `src/simulate_mpi.py`: MPI domain decomposition with ghost-row exchanges

## Run

```bash
python src/generate_sample_firms.py
python src/simulate_serial.py --input data/firms_sample.csv
mpiexec -n 4 python src/simulate_mpi.py --input data/firms_sample.csv
```
