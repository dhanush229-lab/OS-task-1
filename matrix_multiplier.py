from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from time import perf_counter
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
    """Assign output cells to named Python workers.

    A worker calculates one C[row, column] entry at a time.  Its inner loop is
    intentionally scalar and calls ``tf.multiply`` for every pair of elements.
    """

    def __init__(self, max_workers: Optional[int] = None, sample_every: int = 1000):
        available_cores = os.cpu_count() or 1
        self.max_workers = max_workers or min(8, available_cores)
        self.sample_every = max(1, sample_every)
        self.unique_threads: set[str] = set()
        self.events: list[MultiplicationEvent] = []
        self._thread_set_lock = threading.Lock()

    @staticmethod
    def validate_dimensions(left: np.ndarray, right: np.ndarray) -> None:
        """Reject non-matrices and shapes that cannot be multiplied."""
        if left.ndim != 2 or right.ndim != 2:
            raise ValueError("Both inputs must be two-dimensional matrices.")
        if left.shape[1] != right.shape[0]:
            raise ValueError(
                f"Cannot multiply shapes {left.shape} and {right.shape}: "
                "the left width must equal the right height."
            )

    def _worker_for_cell(
        self, left: np.ndarray, right: np.ndarray, row: int, column: int
    ) -> tuple[CellResult, list[MultiplicationEvent]]:
        """Calculate one output location and return its recorded real events."""
        began = perf_counter()
        worker_name = threading.current_thread().name
        with self._thread_set_lock:
            self.unique_threads.add(worker_name)

        left_values = left[row]
        right_values = right[:, column]
        shared_count = left.shape[1]
        output_width = right.shape[1]
        accumulated_value = 0.0
        captured_events: list[MultiplicationEvent] = []

        for shared_index, (left_item, right_item) in enumerate(zip(left_values, right_values)):
            # Keep the TensorFlow scalar operation inside the explicit worker.
            left_scalar = np.float32(left_item)
            right_scalar = np.float32(right_item)
            scalar_product = float(tf.multiply(left_scalar, right_scalar).numpy())
            accumulated_value += scalar_product

            linear_position = ((row * output_width) + column) * shared_count + shared_index
            is_periodic_sample = linear_position % self.sample_every == 0
            is_cell_completion = shared_index == shared_count - 1
            if is_periodic_sample or is_cell_completion:
                captured_events.append(
                    MultiplicationEvent(
                        row=row,
                        col=column,
                        k=shared_index,
                        a_value=float(left_scalar),
                        b_value=float(right_scalar),
                        product=scalar_product,
                        partial_sum=accumulated_value,
                        thread_name=worker_name,
                        timestamp=perf_counter(),
                        operation_index=linear_position,
                    )
                )

        duration = perf_counter() - began
        return (
            CellResult(
                row=row,
                col=column,
                value=accumulated_value,
                thread_name=worker_name,
                elapsed=duration,
                scalar_multiplications=shared_count,
            ),
            captured_events,
        )

    def multiply(
        self,
        a: tf.Tensor | np.ndarray,
        b: tf.Tensor | np.ndarray,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> tuple[tf.Tensor, list[CellResult]]:
        """Return A x B using cell futures and TensorFlow scalar products."""
        left = a.numpy() if isinstance(a, tf.Tensor) else np.asarray(a)
        right = b.numpy() if isinstance(b, tf.Tensor) else np.asarray(b)
        self.validate_dimensions(left, right)

        row_count, column_count = left.shape[0], right.shape[1]
        output = np.empty((row_count, column_count), dtype=np.float32)
        completed_cells: list[CellResult] = []
        recorded_events: list[MultiplicationEvent] = []
        expected_cells = row_count * column_count
        finished_cells = 0

        with ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="MatrixWorker"
        ) as executor:
            submitted_cells = [
                executor.submit(self._worker_for_cell, left, right, row, column)
                for row in range(row_count)
                for column in range(column_count)
            ]
            for completed_future in as_completed(submitted_cells):
                cell, cell_events = completed_future.result()
                output[cell.row, cell.col] = cell.value
                completed_cells.append(cell)
                recorded_events.extend(cell_events)
                finished_cells += 1
                if progress_callback is not None:
                    progress_callback(finished_cells, expected_cells)

        completed_cells.sort(key=lambda cell: (cell.row, cell.col))
        recorded_events.sort(key=lambda event: event.operation_index)
        self.events = recorded_events
        return tf.convert_to_tensor(output, dtype=tf.float32), completed_cells
