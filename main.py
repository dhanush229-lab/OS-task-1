from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from animation import create_animation
from matrix_multiplier import ThreadedMatrixMultiplier


DEFAULT_SIZE = 100
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)
SEED = 42


def build_matrices(size: int) -> tuple[tf.Tensor, tf.Tensor]:
    if size < 1:
        raise ValueError("Matrix size must be at least 1.")
    tf.random.set_seed(SEED)
    a = tf.random.uniform(
        (size, size), minval=0.0, maxval=10.0, dtype=tf.float32, seed=1
    )
    b = tf.random.uniform(
        (size, size), minval=0.0, maxval=10.0, dtype=tf.float32, seed=2
    )
    return a, b


def sequential_python_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Simple sequential reference used only for optional benchmarking."""
    rows, inner = a.shape
    _, cols = b.shape
    result = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            total = 0.0
            for k in range(inner):
                total += float(a[i, k]) * float(b[k, j])
            result[i, j] = total
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Threaded TensorFlow matrix multiplication")
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SIZE,
        help="Square matrix dimension (default: 100)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Worker thread count (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=1000,
        help="Record one real operation for every N scalar multiplications (default: 1000)",
    )
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="Skip GIF generation and display",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Also run the sequential Python benchmark",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.size < 1:
        raise ValueError("--size must be at least 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    print("=" * 62)
    print("THREAD-BASED MATRIX MULTIPLICATION")
    print("TensorFlow + Python ThreadPoolExecutor")
    print("=" * 62)

    size = args.size
    a, b = build_matrices(size)

    print(f"Matrix A         : {size} x {size}")
    print(f"Matrix B         : {size} x {size}")
    print(f"Result           : {size} x {size}")
    print(f"Worker threads   : {args.workers}")
    print(f"Scalar products  : {size ** 3:,}")
    print()

    # Small demonstrations can record every real operation; large runs remain sampled.
    sample_every = 1 if size ** 3 <= 400 else args.sample_every
    multiplier = ThreadedMatrixMultiplier(
        max_workers=args.workers,
        sample_every=sample_every,
    )
    last_report = [0.0]

    def progress(done: int, total: int) -> None:
        now = time.perf_counter()
        if done == total or now - last_report[0] >= 1.0:
            print(f"Completed output cells: {done}/{total} ({done / total:.1%})")
            last_report[0] = now

    start = time.perf_counter()
    threaded_result, cell_results = multiplier.multiply(a, b, progress_callback=progress)
    threaded_time = time.perf_counter() - start

    print("\nThreaded computation completed.")
    print(f"Observed worker threads: {len(multiplier.unique_threads)}")
    print(f"Worker names: {sorted(multiplier.unique_threads)}")
    print(f"Tracked scalar multiplications: {size ** 3:,}")
    print(f"Threaded execution time: {threaded_time:.3f} s")

    # Independent TensorFlow reference implementation.
    tf_start = time.perf_counter()
    reference = tf.matmul(a, b)
    reference_np = reference.numpy()
    tf_time = time.perf_counter() - tf_start

    threaded_np = threaded_result.numpy()
    max_error = float(np.max(np.abs(threaded_np - reference_np)))
    passed = bool(np.allclose(threaded_np, reference_np, rtol=1e-5, atol=1e-2))

    print("\nValidation")
    print("-" * 62)
    print("Reference: TensorFlow tf.matmul()")
    print(f"Maximum absolute error: {max_error:.8f}")
    print(f"Tolerance: rtol=1e-5, atol=1e-2")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    print(f"TensorFlow tf.matmul time: {tf_time:.3f} s")

    if args.benchmark:
        print("\nOptional sequential benchmark")
        seq_start = time.perf_counter()
        sequential = sequential_python_multiply(a.numpy(), b.numpy())
        sequential_time = time.perf_counter() - seq_start
        benchmark_error = float(np.max(np.abs(sequential - reference_np)))
        print(f"Sequential Python time: {sequential_time:.3f} s")
        print(f"Sequential max error vs tf.matmul: {benchmark_error:.8f}")

    animation_path = Path(__file__).parent / "animations" / "matrix_multiplication.gif"
    if args.no_animation:
        print("\nAnimation: skipped (--no-animation)")
    else:
        print("\nGenerating animation from sampled real execution events...")
        create_animation(
            a.numpy(),
            b.numpy(),
            threaded_np,
            multiplier.events,
            cell_results,
            animation_path,
        )
        print(f"Animation saved to: {animation_path}")

    if not passed:
        raise RuntimeError("Threaded result failed validation against tf.matmul().")


if __name__ == "__main__":
    main()
