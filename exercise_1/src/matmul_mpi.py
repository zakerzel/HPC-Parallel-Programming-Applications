from __future__ import annotations

import argparse
import time

import numpy as np
from mpi4py import MPI


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        rng = np.random.default_rng(args.seed)
        a = rng.random((args.n, args.n))
        b = rng.random((args.n, args.n))
        row_counts = [args.n // size + (1 if i < args.n % size else 0) for i in range(size)]
        offsets = np.cumsum([0] + row_counts[:-1])
        chunks = [a[offsets[i]: offsets[i] + row_counts[i], :] for i in range(size)]
    else:
        b = None
        chunks = None
        row_counts = None

    local_a = comm.scatter(chunks, root=0)
    b = comm.bcast(b, root=0)

    comm.Barrier()
    start = time.perf_counter()
    local_c = local_a @ b
    gathered = comm.gather(local_c, root=0)
    elapsed = comm.reduce(time.perf_counter() - start, op=MPI.MAX, root=0)

    if rank == 0:
        c = np.vstack(gathered)
        baseline = a @ b
        ok = np.allclose(c, baseline)
        print({'n': args.n, 'processes': size, 'time_s': elapsed, 'correct': ok})


if __name__ == '__main__':
    main()
