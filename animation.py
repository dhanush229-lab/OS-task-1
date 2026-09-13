from __future__ import annotations

from pathlib import Path

import matplotlib
# GIF output must also work on machines without a functional desktop Tk install.
# Displaying a window is optional; saving the real-event animation is not.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import numpy as np

from matrix_multiplier import CellResult, MultiplicationEvent


def _evenly_spaced(items: list[MultiplicationEvent], count: int) -> list[MultiplicationEvent]:
    """Return representative, ordered real events without manufacturing data."""
    if count >= len(items):
        return items
    positions = np.linspace(0, len(items) - 1, count, dtype=int)
    return [items[position] for position in np.unique(positions)]


def _select_frame_events(events: list[MultiplicationEvent], inner: int, max_frames: int) -> list[MultiplicationEvent]:
    """Keep a large animation compact while favouring completed-cell events."""
    if len(events) <= max_frames:
        return list(events)
    completed = [event for event in events if event.k == inner - 1]
    intermediate = [event for event in events if event.k != inner - 1]
    detail_count = min(max_frames // 4, len(intermediate))
    selected = _evenly_spaced(completed, max_frames - detail_count)
    selected.extend(_evenly_spaced(intermediate, detail_count))
    return sorted(selected, key=lambda event: event.operation_index)


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
    """Create a polished, data-faithful animation from real worker events."""
    del cell_results  # Terminal MultiplicationEvents represent completed cells.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not events:
        raise ValueError("No execution events were recorded; cannot create animation.")

    frame_events = _select_frame_events(events, a.shape[1], max_frames)
    thread_names = sorted({event.thread_name for event in events})
    totals = {name: sum(event.thread_name == name for event in events) for name in thread_names}

    # A compact canvas keeps the GIF practical to render for a 100-400 frame run.
    fig = plt.figure(figsize=(10, 6), dpi=80)
    grid = GridSpec(2, 3, figure=fig, height_ratios=(1.15, 0.85))
    ax_a, ax_b, ax_c = (fig.add_subplot(grid[0, index]) for index in range(3))
    ax_info = fig.add_subplot(grid[1, :2])
    ax_threads = fig.add_subplot(grid[1, 2])

    a_image = ax_a.imshow(a, aspect="auto", cmap="viridis")
    b_image = ax_b.imshow(b, aspect="auto", cmap="viridis")
    displayed = np.zeros_like(result, dtype=np.float32)
    im_c = ax_c.imshow(
        displayed, aspect="auto", cmap="magma", vmin=0, vmax=max(1.0, float(np.max(result)))
    )
    for axis, title in ((ax_a, f"Matrix A ({a.shape[0]} x {a.shape[1]})"), (ax_b, f"Matrix B ({b.shape[0]} x {b.shape[1]})"), (ax_c, "Result C - progressively constructed")):
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("Column")
        axis.set_ylabel("Row")
    fig.colorbar(a_image, ax=ax_a, fraction=0.046, pad=0.04)
    fig.colorbar(b_image, ax=ax_b, fraction=0.046, pad=0.04)
    fig.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.04)

    row_highlight = Rectangle((-0.5, -0.5), a.shape[1], 1, fill=False, edgecolor="#ffcc33", linewidth=3)
    column_highlight = Rectangle((-0.5, -0.5), 1, b.shape[0], fill=False, edgecolor="#ffcc33", linewidth=3)
    cell_highlight = Rectangle((-0.5, -0.5), 1, 1, fill=False, edgecolor="#4ee1ff", linewidth=3)
    ax_a.add_patch(row_highlight)
    ax_b.add_patch(column_highlight)
    ax_c.add_patch(cell_highlight)

    ax_info.set_axis_off()
    info_text = ax_info.text(0.02, 0.94, "", transform=ax_info.transAxes, va="top", fontsize=11, family="monospace", bbox={"boxstyle": "round,pad=0.7", "fc": "#f5f7fa", "ec": "#4472c4"})

    positions = np.arange(len(thread_names))
    bars = ax_threads.barh(positions, np.zeros(len(thread_names)), color="#4c78a8")
    ax_threads.set_yticks(positions, thread_names, fontsize=9)
    ax_threads.invert_yaxis()
    ax_threads.set_xlim(0, max(1, max(totals.values(), default=1)))
    ax_threads.set_xlabel("Observed sampled operations")
    ax_threads.set_title("Recorded worker activity", fontweight="bold")
    fig.suptitle("Thread-Based Matrix Multiplication - Real TensorFlow Worker Events", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    def update(frame_index: int):
        event = frame_events[frame_index]
        # Terminal events hold final cell sums; all other values are real partial sums.
        displayed[event.row, event.col] = event.partial_sum
        im_c.set_data(displayed)
        row_highlight.set_y(event.row - 0.5)
        column_highlight.set_x(event.col - 0.5)
        cell_highlight.set_xy((event.col - 0.5, event.row - 0.5))

        # FuncAnimation may draw its first frame more than once while saving.
        # Rebuild from the selected real-event prefix rather than accumulating
        # mutable state, so the chart always reports the true count.
        counts = {
            name: sum(prior.thread_name == name for prior in frame_events[: frame_index + 1])
            for name in thread_names
        }
        for bar, name in zip(bars, thread_names):
            bar.set_width(counts[name])
            bar.set_color("#f28e2b" if name == event.thread_name else "#4c78a8")

        progress = (frame_index + 1) / len(frame_events)
        info_text.set_text(
            f"CURRENT OUTPUT CELL:  C[{event.row}, {event.col}]\n\n"
            f"SCALAR MULTIPLICATION: A[{event.row}, {event.k}] x B[{event.k}, {event.col}]\n"
            f"ACTUAL VALUES:         {event.a_value:.4f} x {event.b_value:.4f} = {event.product:.4f}\n"
            f"PARTIAL SUM for C[{event.row}, {event.col}]: {event.partial_sum:.4f}\n\n"
            f"EXECUTING THREAD: {event.thread_name}\n"
            f"PROGRESS: {frame_index + 1} / {len(frame_events)} ({progress:.1%})\n"
            f"Scalar operation index: {event.operation_index + 1:,} | Recorded samples: {len(events):,}"
        )
        return (im_c, row_highlight, column_highlight, cell_highlight, info_text, *bars)

    animation = FuncAnimation(fig, update, frames=len(frame_events), interval=1000 / fps, blit=False, repeat=False)
    animation.save(output_path, writer=PillowWriter(fps=fps), dpi=60)
    # Show only after saving, and retain the FuncAnimation through the save.
    plt.show()
