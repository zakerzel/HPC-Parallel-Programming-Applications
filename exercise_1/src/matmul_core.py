import math
import time
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import sparse


def random_dense(n: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.random((n, n)), rng.random((n, n))


def serial_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b


def _row_worker(args: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    a_chunk, b = args
    return a_chunk @ b


def parallel_rows(a: np.ndarray, b: np.ndarray, workers: int | None = None) -> np.ndarray:
    workers = workers or max(1, min(cpu_count(), len(a)))
    chunks = np.array_split(a, workers, axis=0)
    with Pool(processes=workers) as pool:
        parts = pool.map(_row_worker, [(chunk, b) for chunk in chunks if len(chunk) > 0])
    return np.vstack(parts)


def _col_worker(args: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    a, b_chunk = args
    return a @ b_chunk


def parallel_cols(a: np.ndarray, b: np.ndarray, workers: int | None = None) -> np.ndarray:
    workers = workers or max(1, min(cpu_count(), b.shape[1]))
    chunks = np.array_split(b, workers, axis=1)
    with Pool(processes=workers) as pool:
        parts = pool.map(_col_worker, [(a, chunk) for chunk in chunks if chunk.shape[1] > 0])
    return np.hstack(parts)


def _block_worker(args: Tuple[np.ndarray, np.ndarray, int, int]) -> Tuple[int, int, np.ndarray]:
    a_rows, b_cols, row_index, col_index = args
    return row_index, col_index, a_rows @ b_cols


def parallel_blocks(a: np.ndarray, b: np.ndarray, workers: int | None = None) -> np.ndarray:
    workers = workers or max(1, cpu_count())
    grid = max(1, int(math.sqrt(workers)))
    row_chunks = np.array_split(a, grid, axis=0)
    col_chunks = np.array_split(b, grid, axis=1)
    tasks = []
    for i, row_chunk in enumerate(row_chunks):
        for j, col_chunk in enumerate(col_chunks):
            tasks.append((row_chunk, col_chunk, i, j))
    with Pool(processes=min(workers, len(tasks))) as pool:
        results = pool.map(_block_worker, tasks)

    block_map: Dict[Tuple[int, int], np.ndarray] = {(i, j): block for i, j, block in results}
    rows = []
    for i in range(grid):
        rows.append(np.hstack([block_map[(i, j)] for j in range(grid)]))
    return np.vstack(rows)


def _next_power_of_two(n: int) -> int:
    return 1 if n == 0 else 2 ** math.ceil(math.log2(n))


def _pad_matrix(m: np.ndarray, size: int) -> np.ndarray:
    out = np.zeros((size, size), dtype=m.dtype)
    out[: m.shape[0], : m.shape[1]] = m
    return out


def strassen(a: np.ndarray, b: np.ndarray, threshold: int = 64) -> np.ndarray:
    assert a.shape[1] == b.shape[0]
    n = max(a.shape + b.shape)
    m = _next_power_of_two(n)
    a_pad = _pad_matrix(a, m)
    b_pad = _pad_matrix(b, m)
    c_pad = _strassen_recursive(a_pad, b_pad, threshold)
    return c_pad[: a.shape[0], : b.shape[1]]


def _strassen_recursive(a: np.ndarray, b: np.ndarray, threshold: int) -> np.ndarray:
    n = a.shape[0]
    if n <= threshold:
        return a @ b

    mid = n // 2
    a11, a12, a21, a22 = a[:mid, :mid], a[:mid, mid:], a[mid:, :mid], a[mid:, mid:]
    b11, b12, b21, b22 = b[:mid, :mid], b[:mid, mid:], b[mid:, :mid], b[mid:, mid:]

    m1 = _strassen_recursive(a11 + a22, b11 + b22, threshold)
    m2 = _strassen_recursive(a21 + a22, b11, threshold)
    m3 = _strassen_recursive(a11, b12 - b22, threshold)
    m4 = _strassen_recursive(a22, b21 - b11, threshold)
    m5 = _strassen_recursive(a11 + a12, b22, threshold)
    m6 = _strassen_recursive(a21 - a11, b11 + b12, threshold)
    m7 = _strassen_recursive(a12 - a22, b21 + b22, threshold)

    c11 = m1 + m4 - m5 + m7
    c12 = m3 + m5
    c21 = m2 + m4
    c22 = m1 - m2 + m3 + m6

    top = np.hstack((c11, c12))
    bottom = np.hstack((c21, c22))
    return np.vstack((top, bottom))


def benchmark_dense(n: int, workers: int = 4, seed: int = 42) -> List[dict]:
    a, b = random_dense(n, seed)
    methods = {
        'serial': lambda x, y: serial_matmul(x, y),
        'parallel_rows': lambda x, y: parallel_rows(x, y, workers),
        'parallel_cols': lambda x, y: parallel_cols(x, y, workers),
        'parallel_blocks': lambda x, y: parallel_blocks(x, y, workers),
        'strassen': lambda x, y: strassen(x, y),
    }
    baseline = serial_matmul(a, b)
    results = []
    for name, func in methods.items():
        start = time.perf_counter()
        c = func(a, b)
        elapsed = time.perf_counter() - start
        ok = np.allclose(baseline, c, atol=1e-8)
        results.append({'method': name, 'n': n, 'workers': workers, 'time_s': elapsed, 'correct': ok})
    return results


def generate_sparse_pair(n: int, density: float, seed: int = 42) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
    rng = np.random.default_rng(seed)
    a = sparse.random(n, n, density=density, format='csr', random_state=rng)
    b = sparse.random(n, n, density=density, format='csr', random_state=rng)
    return a, b


def benchmark_sparse(n: int, density: float, seed: int = 42) -> List[dict]:
    a, b = generate_sparse_pair(n, density, seed)
    dense_a = a.toarray()
    dense_b = b.toarray()
    rows_start = time.perf_counter(); rows = parallel_rows(dense_a, dense_b, 4); rows_t = time.perf_counter() - rows_start
    serial_start = time.perf_counter(); serial = dense_a @ dense_b; serial_t = time.perf_counter() - serial_start
    sparse_start = time.perf_counter(); sparse_prod = a @ b; sparse_t = time.perf_counter() - sparse_start
    return [
        {'method': 'serial_dense_from_sparse', 'n': n, 'density': density, 'time_s': serial_t, 'nnz': int(a.nnz)},
        {'method': 'parallel_rows_dense_from_sparse', 'n': n, 'density': density, 'time_s': rows_t, 'nnz': int(a.nnz), 'correct': np.allclose(serial, rows)},
        {'method': 'scipy_sparse_csr', 'n': n, 'density': density, 'time_s': sparse_t, 'nnz': int(a.nnz), 'correct': np.allclose(serial, sparse_prod.toarray())},
    ]
