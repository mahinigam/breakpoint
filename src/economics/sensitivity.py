"""Continuous cost-ratio sensitivity sweep.

The sweep answers a reviewer's natural question: what happens when the economy
moves continuously between the four named scenarios? For each cost ratio
``r = loss / margin`` we compare the profit captured by the F1, KS, binary
minimax and three-action robust policies against the best possible cutoff.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.economics.profit_curve import ScoreDistribution
from src.policy.actions import action_profit_at


def run_sensitivity_sweep(
    distribution: ScoreDistribution,
    grid: np.ndarray,
    f1_threshold: float,
    ks_threshold: float,
    minimax_threshold: float,
    action_grow_cutoff: float,
    action_restrict_cutoff: float,
    action_config: dict[str, dict[str, Any]],
    cost_ratio_config: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compute captured-profit curves for every policy across ``r``."""
    r_values = np.linspace(
        float(cost_ratio_config["min"]),
        float(cost_ratio_config["max"]),
        int(cost_ratio_config["points"]),
    )
    base_margin = 100.0

    rows: list[dict[str, float]] = []
    for r_value in r_values:
        margin = base_margin
        loss = float(r_value) * margin
        curve = distribution.profit_curve(grid, margin, loss)
        oracle = float(np.max(curve))
        optimal_cutoff = float(grid[int(np.argmax(curve))])

        f1_profit = distribution.profit_at(f1_threshold, margin, loss)
        ks_profit = distribution.profit_at(ks_threshold, margin, loss)
        minimax_profit = distribution.profit_at(minimax_threshold, margin, loss)
        action_profit = action_profit_at(
            distribution,
            action_grow_cutoff,
            action_restrict_cutoff,
            margin,
            loss,
            action_config,
        )

        rows.append(
            {
                "cost_ratio": float(r_value),
                "optimal_cutoff": optimal_cutoff,
                "oracle_profit": oracle,
                "f1_captured": f1_profit / oracle,
                "ks_captured": ks_profit / oracle,
                "binary_minimax_captured": minimax_profit / oracle,
                "action_policy_captured": action_profit / oracle,
                "binary_minimax_regret": (oracle - minimax_profit) / oracle,
                "action_policy_regret": (oracle - action_profit) / oracle,
            }
        )

    frame = pd.DataFrame(rows)

    # Minimax cutoff over the continuous sweep (binary policy).
    binary_regret = np.maximum(
        (frame["oracle_profit"].to_numpy()[:, None]
         - np.array(
             [
                 distribution.profit_curve(grid, 100.0, r * 100.0)
                 for r in r_values
             ]
         ))
        / frame["oracle_profit"].to_numpy()[:, None],
        0.0,
    )
    binary_worst = np.max(binary_regret, axis=0)
    binary_sweep_index = int(np.argmin(binary_worst))
    binary_sweep_cutoff = float(grid[binary_sweep_index])

    # Worst-case regret of the three-action policy over the sweep.
    action_regret_values = np.maximum(frame["action_policy_regret"].to_numpy(), 0.0)
    summary = {
        "binary_sweep_minimax_cutoff": binary_sweep_cutoff,
        "binary_sweep_worst_case_regret": float(binary_worst[binary_sweep_index]),
        "action_sweep_worst_case_regret": float(np.max(action_regret_values)),
        "action_sweep_worst_case_cost_ratio": float(
            frame.loc[int(np.argmax(action_regret_values)), "cost_ratio"]
        ),
        "binary_minimax_worst_case_cost_ratio": float(
            r_values[int(np.argmax(frame["binary_minimax_regret"].to_numpy()))]
        ),
    }
    return frame, summary


def plot_sensitivity_sweep(
    frame: pd.DataFrame,
    output_dir: str | Path,
    summary: dict[str, float] | None = None,
    filename: str = "sensitivity_sweep.png",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(
        frame["cost_ratio"],
        frame["f1_captured"] * 100,
        linestyle="--",
        label="F1 cutoff",
        color="#1f77b4",
    )
    ax.plot(
        frame["cost_ratio"],
        frame["ks_captured"] * 100,
        linestyle="-.",
        label="KS cutoff",
        color="#ff7f0e",
    )
    ax.plot(
        frame["cost_ratio"],
        frame["binary_minimax_captured"] * 100,
        label="Binary minimax",
        color="#2ca02c",
        linewidth=2,
    )
    ax.plot(
        frame["cost_ratio"],
        frame["action_policy_captured"] * 100,
        label="Three-action robust",
        color="#111111",
        linewidth=2.5,
    )
    ax.set_title("Sensitivity sweep: profit captured across cost ratios")
    ax.set_xlabel("Cost ratio r = expected loss / margin")
    ax.set_ylabel("% of optimal profit captured")
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=False, loc="lower left")

    if summary:
        ax.text(
            0.99,
            0.02,
            (
                f"worst case: binary {summary['binary_sweep_worst_case_regret']:.1%} | "
                f"three-action {summary['action_sweep_worst_case_regret']:.1%}"
            ),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#444444",
        )

    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=300)
    plt.close(fig)
