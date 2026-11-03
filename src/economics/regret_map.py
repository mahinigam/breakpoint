"""Regret map: fractional profit loss by scenario and cutoff."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_regret_map(
    regret_matrix: np.ndarray,
    thresholds: np.ndarray,
    scenarios: list[str],
    output_dir: str | Path,
    minimax_threshold: float | None = None,
    filename: str = "regret_map.png",
) -> np.ndarray:
    """Plot a scenario x threshold regret heatmap plus the worst-case curve.

    Parameters
    ----------
    regret_matrix:
        Shape ``(n_scenarios, n_thresholds)`` of fractional regret.
    thresholds:
        Cutoffs corresponding to the columns.
    scenarios:
        Row labels.
    minimax_threshold:
        If provided, the closest threshold is highlighted.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    regret_matrix = np.asarray(regret_matrix, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)

    n_ticks = min(14, len(thresholds))
    index = np.unique(np.linspace(0, len(thresholds) - 1, n_ticks, dtype=int))
    worst_case = np.max(regret_matrix, axis=0)
    minimax_index = int(np.argmin(worst_case))
    if minimax_threshold is not None:
        nearest = int(np.argmin(np.abs(thresholds - minimax_threshold)))
        index = np.unique(np.append(index, nearest))
    plot_matrix = regret_matrix[:, index]
    plot_thresholds = thresholds[index]

    fig = plt.figure(figsize=(14, 9.5))
    grid = fig.add_gridspec(2, 1, height_ratios=[2.1, 1.0], hspace=0.55)

    ax_heat = fig.add_subplot(grid[0])
    sns.heatmap(
        plot_matrix,
        annot=True,
        fmt=".1%",
        cmap="YlOrRd",
        xticklabels=[f"{value:.3g}" for value in plot_thresholds],
        yticklabels=[name.title() for name in scenarios],
        ax=ax_heat,
        cbar_kws={"label": "Fractional regret"},
    )
    ax_heat.set_title("Regret map: profit left on the table")
    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("Economic scenario")

    if minimax_threshold is not None:
        nearest = int(np.argmin(np.abs(thresholds - minimax_threshold)))
        position = int(np.where(index == nearest)[0][0])
        ax_heat.axvline(position + 0.5, color="#111111", linewidth=2.5)
        ax_heat.set_title(
            "Regret map: profit left on the table "
            f"(black line = minimax cutoff {minimax_threshold:.4g})"
        )

    ax_line = fig.add_subplot(grid[1])
    ax_line.plot(thresholds, worst_case * 100, color="#111111", linewidth=2)
    ax_line.axvline(
        thresholds[minimax_index],
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        label=f"Minimax cutoff = {thresholds[minimax_index]:.4f}",
    )
    ax_line.set_title(
        f"Worst-case regret across scenarios "
        f"({worst_case[minimax_index] * 100:.1f}% at the minimax cutoff)"
    )
    ax_line.set_xlabel("Probability cutoff")
    ax_line.set_ylabel("Worst-case regret (%)")
    ax_line.grid(True, linestyle="--", alpha=0.4)
    ax_line.legend(frameon=False)

    fig.subplots_adjust(left=0.13, right=0.95, top=0.93, bottom=0.09)
    fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return regret_matrix
