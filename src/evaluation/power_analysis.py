"""Simulation-based power analysis for an offline champion/challenger test.

The simulation draws accounts from the **population-weighted** score/label
distribution, so the result is expressed in live accounts rather than in the
20x-weighted modelling sample. The champion is the three-action policy and the
challenger is the binary minimax policy under the stress scenario.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.config import (
    ACTION_ECONOMICS,
    ECONOMIC_SCENARIOS,
    FIGURES_DIR,
    NEGATIVE_WEIGHT,
    PATHS,
    REPORTS_DIR,
    SEED,
    ensure_directories,
    scenario_loss,
)
from src.evaluation.bootstrap import action_profit_vector, binary_profit_vector
from src.reporting import environment_metadata, load_json, update_json, write_json


def _profit_distribution(
    profit: np.ndarray, population_weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse per-account profit into a small categorical distribution."""
    values, inverse = np.unique(profit, return_inverse=True)
    probabilities = np.bincount(
        inverse, weights=population_weight, minlength=len(values)
    )
    probabilities = probabilities / probabilities.sum()
    return values, probabilities


def simulate_power(
    profit_champion: np.ndarray,
    profit_challenger: np.ndarray,
    population_weight: np.ndarray,
    n_accounts: int,
    n_sims: int = 500,
    alpha: float = 0.05,
    seed: int = SEED,
) -> dict[str, float]:
    """Estimate power for a two-arm test with independent samples."""
    rng = np.random.default_rng(seed)
    champion_values, champion_probs = _profit_distribution(
        profit_champion, population_weight
    )
    challenger_values, challenger_probs = _profit_distribution(
        profit_challenger, population_weight
    )

    critical = float(stats.norm.ppf(1.0 - alpha))
    successes = 0
    observed_differences = np.empty(n_sims, dtype=float)
    for index in range(n_sims):
        champion = rng.choice(
            champion_values, size=n_accounts, p=champion_probs, replace=True
        )
        challenger = rng.choice(
            challenger_values, size=n_accounts, p=challenger_probs, replace=True
        )
        difference = float(champion.mean() - challenger.mean())
        observed_differences[index] = difference
        standard_error = float(
            np.sqrt(
                champion.var(ddof=1) / n_accounts
                + challenger.var(ddof=1) / n_accounts
            )
        )
        if standard_error > 0 and difference / standard_error > critical:
            successes += 1

    return {
        "n_accounts_per_arm": int(n_accounts),
        "power": float(successes / n_sims),
        "mean_difference": float(observed_differences.mean()),
        "std_difference": float(observed_differences.std(ddof=1)),
        "alpha": float(alpha),
        "n_sims": int(n_sims),
    }


def _required_sample_size(
    rows: list[dict[str, float]], target_power: float = 0.80
) -> int | None:
    for previous, current in zip(rows, rows[1:]):
        if previous["power"] < target_power <= current["power"]:
            fraction = (target_power - previous["power"]) / (
                current["power"] - previous["power"]
            )
            return int(
                round(
                    previous["n_accounts_per_arm"]
                    + fraction
                    * (
                        current["n_accounts_per_arm"]
                        - previous["n_accounts_per_arm"]
                    )
                )
            )
    if rows and rows[-1]["power"] >= target_power:
        return int(rows[-1]["n_accounts_per_arm"])
    return None


def run_power_analysis(n_sims: int = 500) -> dict[str, Any]:
    ensure_directories()
    frame = pd.read_parquet(PATHS["calibrated_predictions"])
    policy = load_json(REPORTS_DIR / "policy.json")
    if not policy:
        raise FileNotFoundError("reports/policy.json not found. Run decision analysis first.")

    y_true = frame["target"].to_numpy(dtype=int)
    y_prob = frame["lgb_prob_calibrated"].to_numpy(dtype=float)
    weights = np.where(y_true == 0, float(NEGATIVE_WEIGHT), 1.0)
    population_weight = weights / weights.sum()

    scenario = ECONOMIC_SCENARIOS["stress"]
    margin = float(scenario["margin"])
    loss = scenario_loss(scenario)

    binary_threshold = float(policy["binary_minimax"]["threshold"])
    grow_cutoff = float(policy["action_policy"]["grow_cutoff"])
    restrict_cutoff = float(policy["action_policy"]["restrict_cutoff"])

    binary_profit = binary_profit_vector(
        y_true, y_prob, binary_threshold, margin, loss
    )
    action_profit = action_profit_vector(
        y_true,
        y_prob,
        grow_cutoff,
        restrict_cutoff,
        margin,
        loss,
        ACTION_ECONOMICS,
    )

    sample_sizes = [1_000, 2_500, 5_000, 10_000, 25_000, 50_000]
    rows = [
        simulate_power(
            action_profit,
            binary_profit,
            population_weight,
            n_accounts=n,
            n_sims=n_sims,
            seed=SEED + index,
        )
        for index, n in enumerate(sample_sizes)
    ]
    required = _required_sample_size(rows)

    payload = {
        "champion": "three_action_robust",
        "challenger": "binary_minimax",
        "scenario": "stress",
        "binary_threshold": binary_threshold,
        "grow_cutoff": grow_cutoff,
        "restrict_cutoff": restrict_cutoff,
        "sample_sizes_per_arm": sample_sizes,
        "results": rows,
        "required_n_per_arm_80pct": required,
        "interpretation": (
            "Two-arm test with independent draws from the population-weighted "
            "score/label distribution; one-sided alpha = 0.05."
        ),
        "environment": environment_metadata(),
    }
    write_json(REPORTS_DIR / "power_analysis.json", payload)
    update_json(REPORTS_DIR / "metrics.json", "power_analysis", payload)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        [row["n_accounts_per_arm"] for row in rows],
        [row["power"] * 100 for row in rows],
        marker="o",
        linewidth=2,
        color="#1f77b4",
    )
    ax.axhline(80, color="#d62728", linestyle="--", label="80% power")
    if required is not None:
        ax.axvline(
            required,
            color="#2ca02c",
            linestyle=":",
            label=f"Required n ≈ {required:,} per arm",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Accounts per arm (log scale)")
    ax.set_ylabel("Simulated power (%)")
    ax.set_ylim(0, 105)
    ax.set_title(
        "Power to detect the three-action profit advantage (stress scenario)"
    )
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "power_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    for row in rows:
        print(
            f"  {row['n_accounts_per_arm']:,} per arm -> "
            f"{row['power']*100:.1f}% power"
        )
    if required is not None:
        print(f"Required sample size for 80% power: ~{required:,} per arm")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sims", type=int, default=500)
    args = parser.parse_args()
    run_power_analysis(n_sims=args.sims)


if __name__ == "__main__":
    main()
