from __future__ import annotations

import argparse
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import tensorflow as tf

from animation import create_animation
from matrix_multiplier import ThreadedMatrixMultiplier


DEFAULT_ORDER = 100
RANDOM_SEED = 42
DEFAULT_POOL_SIZE = min(8, os.cpu_count() or 1)


def build_matrices(size: int) -> tuple[tf.Tensor, tf.Tensor]:
    """Create the deterministic square input tensors used by the demonstration."""
    if size < 1:
        raise ValueError("Matrix size must be positive.")

    tf.random.set_seed(RANDOM_SEED)
    shape = (size, size)
    left = tf.random.uniform(shape, 0.0, 10.0, dtype=tf.float32, seed=1)
    right = tf.random.uniform(shape, 0.0, 10.0, dtype=tf.float32, seed=2)
    return left, right


def sequential_python_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Optional plain-Python baseline; it is not used by the threaded routine."""
    row_count, shared_count = left.shape
    column_count = right.shape[1]
    baseline = np.zeros((row_count, column_count), dtype=np.float32)

    for row in range(row_count):
        for column in range(column_count):
            running_total = 0.0
            for shared_index in range(shared_count):
                running_total += float(left[row, shared_index]) * float(right[shared_index, column])
            baseline[row, column] = running_total
    return baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multiply square TensorFlow matrices with explicit Python workers."
    )
    parser.add_argument("--size", type=int, default=DEFAULT_ORDER, help="Matrix order (default: 100)")
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_POOL_SIZE,
        help=f"Number of worker threads (default: {DEFAULT_POOL_SIZE})",
    )
    parser.add_argument(
        "--sample-every", type=int, default=1000,
        help="Keep one real scalar-operation event per N operations (default: 1000)",
    )
    parser.add_argument("--no-animation", action="store_true", help="Do not write the event GIF")
    parser.add_argument("--benchmark", action="store_true", help="Run the optional Python baseline")
    return parser.parse_args()


def _show_configuration(order: int, worker_count: int) -> None:
    print("=" * 62)
    print("EXPLICIT-THREAD MATRIX MULTIPLICATION")
    print("TensorFlow scalar products with ThreadPoolExecutor")
    print("=" * 62)
    print(f"Left matrix      : {order} x {order}")
    print(f"Right matrix     : {order} x {order}")
    print(f"Output matrix    : {order} x {order}")
    print(f"Worker threads   : {worker_count}")
    print(f"Scalar products  : {order ** 3:,}\n")


def main() -> None:
    options = parse_args()
    if options.size < 1:
        raise ValueError("--size must be at least 1")
    if options.workers < 1:
        raise ValueError("--workers must be at least 1")

    _show_configuration(options.size, options.workers)
    left_tensor, right_tensor = build_matrices(options.size)

    # A small walkthrough benefits from every genuine scalar event.  Large
    # matrices continue to use the caller's sampling interval.
    sampling_interval = 1 if options.size ** 3 <= 400 else options.sample_every
    engine = ThreadedMatrixMultiplier(options.workers, sampling_interval)
    last_progress_report = 0.0

    def report_cell_completion(finished: int, expected: int) -> None:
        nonlocal last_progress_report
        current_time = perf_counter()
        if finished == expected or current_time - last_progress_report >= 1.0:
            print(f"Completed output cells: {finished}/{expected} ({finished / expected:.1%})")
            last_progress_report = current_time

    started_at = perf_counter()
    threaded_tensor, completed_cells = engine.multiply(
        left_tensor, right_tensor, progress_callback=report_cell_completion
    )
    worker_elapsed = perf_counter() - started_at

    print("\nWorker calculation complete.")
    print(f"Observed worker threads: {len(engine.unique_threads)}")
    print(f"Worker names: {sorted(engine.unique_threads)}")
    print(f"Tracked scalar multiplications: {options.size ** 3:,}")
    print(f"Threaded execution time: {worker_elapsed:.3f} s")

    # This vectorized call is deliberately an independent validation reference.
    reference_started = perf_counter()
    reference_array = tf.matmul(left_tensor, right_tensor).numpy()
    reference_elapsed = perf_counter() - reference_started
    threaded_array = threaded_tensor.numpy()
    largest_difference = float(np.max(np.abs(threaded_array - reference_array)))
    validation_ok = bool(np.allclose(threaded_array, reference_array, rtol=1e-5, atol=1e-2))

    print("\nValidation")
    print("-" * 62)
    print("Reference: TensorFlow tf.matmul()")
    print(f"Maximum absolute error: {largest_difference:.8f}")
    print("Tolerance: rtol=1e-5, atol=1e-2")
    print(f"RESULT: {'PASS' if validation_ok else 'FAIL'}")
    print(f"TensorFlow tf.matmul time: {reference_elapsed:.3f} s")

    if options.benchmark:
        baseline_started = perf_counter()
        baseline = sequential_python_multiply(left_tensor.numpy(), right_tensor.numpy())
        baseline_elapsed = perf_counter() - baseline_started
        baseline_difference = float(np.max(np.abs(baseline - reference_array)))
        print("\nOptional sequential benchmark")
        print(f"Sequential Python time: {baseline_elapsed:.3f} s")
        print(f"Sequential max error vs tf.matmul: {baseline_difference:.8f}")

    if options.no_animation:
        print("\nAnimation: skipped (--no-animation)")
    else:
        gif_path = Path(__file__).parent / "animations" / "matrix_multiplication.gif"
        print("\nBuilding animation from sampled worker events...")
        create_animation(
            left_tensor.numpy(), right_tensor.numpy(), threaded_array,
            engine.events, completed_cells, gif_path,
        )
        print(f"Animation saved to: {gif_path}")

    if not validation_ok:
        raise RuntimeError("The threaded result does not match the TensorFlow reference.")


if __name__ == "__main__":
    main()
