"""Paired bootstrap for the profit impact of the decision policy.

The bootstrap is class-stratified so every replicate preserves the observed
number of defaults and non-defaults, and carries the 20x population weight.
It reports the action-policy advantage over the binary minimax policy, plus
the legacy minimax-vs-F1 comparison.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
from src.reporting import environment_metadata, load_json, update_json


@dataclass(frozen=True)
class ProfitBootstrapResult:
    observed_difference: float
    mean_difference: float
    ci_low: float
    ci_high: float
    p_value: float
    probability_positive: float
    n_boot: int
    replicates: np.ndarray | None = None

    def to_dict(self) -> dict[str, float | int]:
        return {
            "observed_difference": self.observed_difference,
            "mean_difference": self.mean_difference,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_value": self.p_value,
            "probability_positive": self.probability_positive,
            "n_boot": self.n_boot,
        }


def binary_profit_vector(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    margin: float,
    loss: float,
) -> np.ndarray:
    """Unweighted per-account profit for the binary approve/decline policy."""
    approved = (y_prob <= threshold).astype(float)
    return approved * (margin - y_true * (margin + loss))


def action_profit_vector(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    grow_cutoff: float,
    restrict_cutoff: float,
    margin: float,
    loss: float,
    action_config: dict[str, dict[str, float]],
) -> np.ndarray:
    """Unweighted per-account profit for the three-action policy."""
    y_true = np.asarray(y_true, dtype=float)
    grow = y_prob <= grow_cutoff
    hold = (y_prob > grow_cutoff) & (y_prob <= restrict_cutoff)
    restrict = y_prob > restrict_cutoff

    grow_margin = margin * float(action_config["grow"]["margin_multiplier"])
    grow_loss = loss * float(action_config["grow"]["exposure_multiplier"])
    hold_margin = margin * float(action_config["hold"]["margin_multiplier"])
    hold_loss = loss * float(action_config["hold"]["exposure_multiplier"])
    restrict_margin = margin * float(action_config["restrict"]["margin_multiplier"])
    restrict_loss = loss * float(action_config["restrict"]["exposure_multiplier"])

    profit = np.zeros(len(y_true), dtype=float)
    profit[grow] = grow_margin - y_true[grow] * (grow_margin + grow_loss)
    profit[hold] = hold_margin - y_true[hold] * (hold_margin + hold_loss)
    profit[restrict] = (
        restrict_margin - y_true[restrict] * (restrict_margin + restrict_loss)
    )
    return profit


def paired_profit_bootstrap(
    y_true: np.ndarray,
    profit_a: np.ndarray,
    profit_b: np.ndarray,
    weights: np.ndarray,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = SEED,
) -> ProfitBootstrapResult:
    """Class-stratified paired bootstrap for ``profit_b - profit_a``."""
    y_true = np.asarray(y_true).astype(int)
    profit_a = np.asarray(profit_a, dtype=float)
    profit_b = np.asarray(profit_b, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if not (len(y_true) == len(profit_a) == len(profit_b) == len(weights)):
        raise ValueError("All arrays must have the same length")

    difference = profit_b - profit_a
    observed = float(np.sum(weights * difference))

    rng = np.random.default_rng(seed)
    positives = np.flatnonzero(y_true == 1)
    negatives = np.flatnonzero(y_true == 0)
    if len(positives) == 0 or len(negatives) == 0:
        raise ValueError("Bootstrap requires both classes")

    replicates = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        pos = rng.choice(positives, size=len(positives), replace=True)
        neg = rng.choice(negatives, size=len(negatives), replace=True)
        sample = np.concatenate([pos, neg])
        replicates[index] = float(np.sum(weights[sample] * difference[sample]))

    alpha = (1.0 - confidence) / 2.0
    ci_low, ci_high = np.quantile(replicates, [alpha, 1.0 - alpha])
    p_value = 2.0 * min(
        float(np.mean(replicates <= 0.0)),
        float(np.mean(replicates >= 0.0)),
    )
    return ProfitBootstrapResult(
        observed_difference=observed,
        mean_difference=float(np.mean(replicates)),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float(min(1.0, p_value)),
        probability_positive=float(np.mean(replicates > 0.0)),
        n_boot=int(n_boot),
        replicates=replicates,
    )


def _load_inputs() -> tuple[pd.DataFrame, dict[str, Any]]:
    oof_path = PATHS["calibrated_predictions"]
    policy_path = REPORTS_DIR / "policy.json"
    if not oof_path.exists():
        raise FileNotFoundError(f"{oof_path} not found. Run training + calibration first.")
    if not policy_path.exists():
        raise FileNotFoundError(f"{policy_path} not found. Run decision analysis first.")
    return pd.read_parquet(oof_path), load_json(policy_path)


def _plot_distribution(
    values: np.ndarray,
    observed: float,
    ci_low: float,
    ci_high: float,
    title: str,
    filename: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.hist(values / 1e6, bins=60, color="#4c78a8", alpha=0.8)
    ax.axvline(observed / 1e6, color="#111111", linewidth=2, label="Observed")
    ax.axvline(ci_low / 1e6, color="#d62728", linestyle="--", label="95% CI")
    ax.axvline(ci_high / 1e6, color="#d62728", linestyle="--")
    ax.axvline(0.0, color="#777777", linestyle=":", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Profit difference ($ millions)")
    ax.set_ylabel("Bootstrap replicates")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_bootstrap(n_boot: int = 1000) -> dict[str, Any]:
    ensure_directories()
    frame, policy = _load_inputs()
    y_true = frame["target"].to_numpy(dtype=int)
    y_prob = frame["lgb_prob_calibrated"].to_numpy(dtype=float)
    weights = np.where(y_true == 0, float(NEGATIVE_WEIGHT), 1.0)

    binary_threshold = float(policy["binary_minimax"]["threshold"])
    grow_cutoff = float(policy["action_policy"]["grow_cutoff"])
    restrict_cutoff = float(policy["action_policy"]["restrict_cutoff"])
    f1_threshold = float(policy["heuristics"]["f1"])

    results: dict[str, Any] = {}
    stress = ECONOMIC_SCENARIOS["stress"]
    for scenario_name, scenario in ECONOMIC_SCENARIOS.items():
        margin = float(scenario["margin"])
        loss = scenario_loss(scenario)
        binary = binary_profit_vector(
            y_true, y_prob, binary_threshold, margin, loss
        )
        action = action_profit_vector(
            y_true,
            y_prob,
            grow_cutoff,
            restrict_cutoff,
            margin,
            loss,
            ACTION_ECONOMICS,
        )
        action_result = paired_profit_bootstrap(
            y_true, binary, action, weights, n_boot=n_boot
        )
        f1 = binary_profit_vector(y_true, y_prob, f1_threshold, margin, loss)
        f1_result = paired_profit_bootstrap(
            y_true, f1, binary, weights, n_boot=n_boot
        )
        results[scenario_name] = {
            "action_vs_binary": action_result.to_dict(),
            "binary_vs_f1": f1_result.to_dict(),
        }
        if scenario_name == "stress":
            if action_result.replicates is None:
                raise RuntimeError("Bootstrap replicates were not retained")
            _plot_distribution(
                action_result.replicates,
                action_result.observed_difference,
                action_result.ci_low,
                action_result.ci_high,
                "Action policy advantage over binary minimax (stress)",
                "bootstrap_distribution.png",
            )

    payload = {
        "n_boot": int(n_boot),
        "binary_threshold": binary_threshold,
        "grow_cutoff": grow_cutoff,
        "restrict_cutoff": restrict_cutoff,
        "f1_threshold": f1_threshold,
        "results": results,
        "environment": environment_metadata(),
    }
    update_json(REPORTS_DIR / "metrics.json", "bootstrap", payload)
    from src.reporting import write_json

    write_json(REPORTS_DIR / "bootstrap.json", payload)

    for name, result in results.items():
        action_result = result["action_vs_binary"]
        print(
            f"{name:12s} action-vs-binary "
            f"${action_result['observed_difference']/1e6:,.1f}M "
            f"[{action_result['ci_low']/1e6:,.1f}, "
            f"{action_result['ci_high']/1e6:,.1f}] "
            f"p={action_result['p_value']:.3f}"
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()
    run_bootstrap(n_boot=args.n_boot)


if __name__ == "__main__":
    main()
