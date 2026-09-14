from __future__ import annotations

from pathlib import Path

import matplotlib

# Saving the presentation should not depend on a working desktop Tk setup.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import numpy as np

from matrix_multiplier import CellResult, MultiplicationEvent


def _pick_evenly(events: list[MultiplicationEvent], limit: int) -> list[MultiplicationEvent]:
    """Select a representative subset while retaining only observed events."""
    if len(events) <= limit:
        return events
    selected_indices = np.linspace(0, len(events) - 1, num=limit, dtype=int)
    return [events[index] for index in np.unique(selected_indices)]


def _events_for_animation(
    recorded_events: list[MultiplicationEvent], shared_dimension: int, frame_limit: int
) -> list[MultiplicationEvent]:
    """Give output-cell completions precedence in a bounded-size GIF."""
    if len(recorded_events) <= frame_limit:
        return list(recorded_events)

    terminal_events = [event for event in recorded_events if event.k == shared_dimension - 1]
    partial_events = [event for event in recorded_events if event.k != shared_dimension - 1]
    partial_quota = min(frame_limit // 4, len(partial_events))
    selected = _pick_evenly(terminal_events, frame_limit - partial_quota)
    selected.extend(_pick_evenly(partial_events, partial_quota))
    return sorted(selected, key=lambda event: event.operation_index)


def _operation_summary(event: MultiplicationEvent, current: int, total: int, available_events: int) -> str:
    completion = current / total
    return (
        f"CURRENT OUTPUT CELL:  C[{event.row}, {event.col}]\n\n"
        f"SCALAR MULTIPLICATION: A[{event.row}, {event.k}] x B[{event.k}, {event.col}]\n"
        f"ACTUAL VALUES:         {event.a_value:.4f} x {event.b_value:.4f} = {event.product:.4f}\n"
        f"PARTIAL SUM for C[{event.row}, {event.col}]: {event.partial_sum:.4f}\n\n"
        f"EXECUTING THREAD: {event.thread_name}\n"
        f"PROGRESS: {current} / {total} ({completion:.1%})\n"
        f"Scalar operation index: {event.operation_index + 1:,} | Recorded samples: {available_events:,}"
    )


def create_animation(
    a: np.ndarray,
    b: np.ndarray,
    result: np.ndarray,
    events: list[MultiplicationEvent],
    cell_results: list[CellResult],
    output_path: str | Path,
    fps: int = 6,
    max_frames: int = 400,
) -> None:
    """Render actual multiplication events as a compact explanatory GIF."""
    del cell_results  # A final event for each cell contains its completed value.
    if not events:
        raise ValueError("No execution events were recorded; animation is unavailable.")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    displayed_events = _events_for_animation(events, a.shape[1], max_frames)
    worker_names = sorted({event.thread_name for event in events})
    worker_totals = {
        worker: sum(event.thread_name == worker for event in events)
        for worker in worker_names
    }

    figure = plt.figure(figsize=(10, 6), dpi=80)
    layout = GridSpec(2, 3, figure=figure, height_ratios=(1.15, 0.85))
    matrix_a_axis = figure.add_subplot(layout[0, 0])
    matrix_b_axis = figure.add_subplot(layout[0, 1])
    matrix_c_axis = figure.add_subplot(layout[0, 2])
    details_axis = figure.add_subplot(layout[1, :2])
    workers_axis = figure.add_subplot(layout[1, 2])

    left_image = matrix_a_axis.imshow(a, aspect="auto", cmap="viridis")
    right_image = matrix_b_axis.imshow(b, aspect="auto", cmap="viridis")
    visible_result = np.zeros_like(result, dtype=np.float32)
    result_image = matrix_c_axis.imshow(
        visible_result,
        aspect="auto",
        cmap="magma",
        vmin=0,
        vmax=max(1.0, float(np.max(result))),
    )
    for axis, heading in (
        (matrix_a_axis, f"Matrix A ({a.shape[0]} x {a.shape[1]})"),
        (matrix_b_axis, f"Matrix B ({b.shape[0]} x {b.shape[1]})"),
        (matrix_c_axis, "Result C - progressively constructed"),
    ):
        axis.set_title(heading, fontweight="bold")
        axis.set_xlabel("Column")
        axis.set_ylabel("Row")
    for image, axis in ((left_image, matrix_a_axis), (right_image, matrix_b_axis), (result_image, matrix_c_axis)):
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    row_marker = Rectangle((-0.5, -0.5), a.shape[1], 1, fill=False, edgecolor="#ffcc33", linewidth=3)
    column_marker = Rectangle((-0.5, -0.5), 1, b.shape[0], fill=False, edgecolor="#ffcc33", linewidth=3)
    result_marker = Rectangle((-0.5, -0.5), 1, 1, fill=False, edgecolor="#4ee1ff", linewidth=3)
    matrix_a_axis.add_patch(row_marker)
    matrix_b_axis.add_patch(column_marker)
    matrix_c_axis.add_patch(result_marker)

    details_axis.set_axis_off()
    details_text = details_axis.text(
        0.02, 0.94, "", transform=details_axis.transAxes, va="top", fontsize=11,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.7", "fc": "#f5f7fa", "ec": "#4472c4"},
    )

    worker_positions = np.arange(len(worker_names))
    worker_bars = workers_axis.barh(worker_positions, np.zeros(len(worker_names)), color="#4c78a8")
    workers_axis.set_yticks(worker_positions, worker_names, fontsize=9)
    workers_axis.invert_yaxis()
    workers_axis.set_xlim(0, max(1, max(worker_totals.values(), default=1)))
    workers_axis.set_xlabel("Observed sampled operations")
    workers_axis.set_title("Recorded worker activity", fontweight="bold")
    figure.suptitle("Thread-Based Matrix Multiplication - Real TensorFlow Worker Events", fontsize=14, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.93))

    def draw_frame(frame_number: int):
        current_event = displayed_events[frame_number]
        visible_result[current_event.row, current_event.col] = current_event.partial_sum
        result_image.set_data(visible_result)
        row_marker.set_y(current_event.row - 0.5)
        column_marker.set_x(current_event.col - 0.5)
        result_marker.set_xy((current_event.col - 0.5, current_event.row - 0.5))

        # Calculate from the prefix each time: animation writers can redraw frame 0.
        prefix = displayed_events[: frame_number + 1]
        for bar, worker in zip(worker_bars, worker_names):
            seen_count = sum(event.thread_name == worker for event in prefix)
            bar.set_width(seen_count)
            bar.set_color("#f28e2b" if worker == current_event.thread_name else "#4c78a8")

        details_text.set_text(
            _operation_summary(current_event, frame_number + 1, len(displayed_events), len(events))
        )
        return (result_image, row_marker, column_marker, result_marker, details_text, *worker_bars)

    animation = FuncAnimation(
        figure, draw_frame, frames=len(displayed_events), interval=1000 / fps,
        blit=False, repeat=False,
    )
    animation.save(destination, writer=PillowWriter(fps=fps), dpi=60)
    # ``animation`` stays referenced until saving is complete; opening a window is optional.
    plt.show()
