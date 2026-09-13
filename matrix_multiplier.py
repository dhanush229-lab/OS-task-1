from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import tensorflow as tf


@dataclass(frozen=True)
class MultiplicationEvent:
    row: int
    col: int
    k: int
    a_value: float
    b_value: float
    product: float
    partial_sum: float
    thread_name: str
    timestamp: float
    operation_index: int


@dataclass(frozen=True)
class CellResult:
    row: int
    col: int
    value: float
    thread_name: str
    elapsed: float
    scalar_multiplications: int


class ThreadedMatrixMultiplier:
    """Matrix multiplication with explicit Python worker threads and TensorFlow.

    Each output cell C[i, j] is assigned to a worker thread. Inside that worker,
    each scalar multiplication A[i, k] * B[k, j] is executed with tf.multiply().
    """

    def __init__(self, max_workers: Optional[int] = None, sample_every: int = 1000):
        self.max_workers = max_workers or min(8, os.cpu_count() or 1)
        self.sample_every = max(1, sample_every)
        self.unique_threads: set[str] = set()
        self.events: list[MultiplicationEvent] = []
        self._lock = threading.Lock()

    @staticmethod
    def validate_dimensions(a: np.ndarray, b: np.ndarray) -> None:
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("Both matrices must be 2-D.")
        if a.shape[1] != b.shape[0]:
            raise ValueError(
                f"Incompatible dimensions: A is {a.shape}, B is {b.shape}. "
                "A columns must equal B rows."
            )

    def _worker_for_cell(
        self,
        a: np.ndarray,
        b: np.ndarray,
        i: int,
        j: int,
    ) -> tuple[CellResult, list[MultiplicationEvent]]:
        start = time.perf_counter()
        thread_name = threading.current_thread().name
        with self._lock:
            self.unique_threads.add(thread_name)

        a_row = a[i]
        b_col = b[:, j]
        k_count = a.shape[1]
        sampled_events: list[MultiplicationEvent] = []
        cell_sum = 0.0

        for k in range(k_count):
            # These are deliberately scalar operations performed inside the worker.
            a_value = np.float32(a_row[k])
            b_value = np.float32(b_col[k])
            product_tensor = tf.multiply(a_value, b_value)
            product_value = float(product_tensor.numpy())
            cell_sum += product_value

            operation_index = (i * b.shape[1] * k_count) + (j * k_count) + k
            # Preserve final cell values in the event stream for visualization.
            if operation_index % self.sample_every == 0 or k == k_count - 1:
                sampled_events.append(
                    MultiplicationEvent(
                        row=i,
                        col=j,
                        k=k,
                        a_value=float(a_value),
                        b_value=float(b_value),
                        product=product_value,
                        partial_sum=cell_sum,
                        thread_name=thread_name,
                        timestamp=time.perf_counter(),
                        operation_index=operation_index,
                    )
                )

        elapsed = time.perf_counter() - start
        return (
            CellResult(
                row=i,
                col=j,
                value=cell_sum,
                thread_name=thread_name,
                elapsed=elapsed,
                scalar_multiplications=k_count,
            ),
            sampled_events,
        )

    def multiply(
        self,
        a: tf.Tensor | np.ndarray,
        b: tf.Tensor | np.ndarray,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> tuple[tf.Tensor, list[CellResult]]:
        """Compute C = A x B using explicit worker threads.

        The input matrices are converted to NumPy once before threading only to
        avoid expensive TensorFlow tensor indexing inside the million-operation loop.
        The actual scalar multiplication remains TensorFlow tf.multiply() inside
        the worker threads.
        """
        a_np = a.numpy() if isinstance(a, tf.Tensor) else np.asarray(a)
        b_np = b.numpy() if isinstance(b, tf.Tensor) else np.asarray(b)
        self.validate_dimensions(a_np, b_np)

        rows = a_np.shape[0]
        cols = b_np.shape[1]
        result = np.empty((rows, cols), dtype=np.float32)
        cell_results: list[CellResult] = []
        all_events: list[MultiplicationEvent] = []
        completed = 0

        with ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="MatrixWorker",
        ) as executor:
            futures = [
                executor.submit(self._worker_for_cell, a_np, b_np, i, j)
                for i in range(rows)
                for j in range(cols)
            ]

            for future in as_completed(futures):
                cell, sampled_events = future.result()
                result[cell.row, cell.col] = cell.value
                cell_results.append(cell)
                all_events.extend(sampled_events)
                completed += 1
                if progress_callback:
                    progress_callback(completed, rows * cols)

        cell_results.sort(key=lambda item: (item.row, item.col))
        all_events.sort(key=lambda event: event.operation_index)
        self.events = all_events
        return tf.convert_to_tensor(result, dtype=tf.float32), cell_results
