import numpy as np
import time


def matmul_serial(A, B):
    return A @ B


if __name__ == "__main__":
    np.random.seed(42)
    n = 256
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)

    start = time.perf_counter()
    C = matmul_serial(A, B)
    elapsed = time.perf_counter() - start

    print(f"Serial matrix multiplication finished for size {n}x{n}")
    print(f"Elapsed time: {elapsed:.6f} seconds")
    print(f"Checksum: {np.sum(C):.6f}")
