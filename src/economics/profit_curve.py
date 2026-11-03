"""Expected-profit calculations for credit decisions.

The central object is :class:`ScoreDistribution`, which stores a sorted score
vector and cumulative weighted good/bad counts. Any threshold query then costs
``O(log n)`` instead of ``O(n)``, which makes large grid searches (and the
three-action policy search) tractable.

Decision semantics: an account is approved when ``score <= threshold``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

SCENARIO_COLORS = {
    "conservative": "#1f77b4",
    "moderate": "#2ca02c",
    "aggressive": "#ff7f0e",
    "stress": "#d62728",
}


@dataclass(frozen=True)
class ScoreDistribution:
    """Sorted scores with cumulative population-weighted label counts."""

    scores: np.ndarray
    cum_good: np.ndarray
    cum_bad: np.ndarray
    cum_weight: np.ndarray
    cum_weighted_pd: np.ndarray
    total_good: float
    total_bad: float
    total_weight: float
    total_weighted_pd: float

    @classmethod
    def from_labels(
        cls,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        negative_weight: float,
    ) -> "ScoreDistribution":
        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob, dtype=float)
        if len(y_true) != len(y_prob):
            raise ValueError("y_true and y_prob must have the same length")
        if not np.isfinite(y_prob).all():
            raise ValueError("y_prob contains non-finite values")

        order = np.argsort(y_prob, kind="mergesort")
        scores = y_prob[order]
        labels = y_true[order]

        weights = np.where(labels == 0, float(negative_weight), 1.0)
        good = np.where(labels == 0, float(negative_weight), 0.0)
        bad = np.where(labels == 1, 1.0, 0.0)
        weighted_pd = weights * scores

        zero = np.array([0.0])
        return cls(
            scores=scores,
            cum_good=np.concatenate([zero, np.cumsum(good)]),
            cum_bad=np.concatenate([zero, np.cumsum(bad)]),
            cum_weight=np.concatenate([zero, np.cumsum(weights)]),
            cum_weighted_pd=np.concatenate([zero, np.cumsum(weighted_pd)]),
            total_good=float(good.sum()),
            total_bad=float(bad.sum()),
            total_weight=float(weights.sum()),
            total_weighted_pd=float(weighted_pd.sum()),
        )

    def row_weights(self) -> np.ndarray:
        """Per-row population weights, aligned with ``scores``."""
        return np.diff(self.cum_weight)

    def row_good(self) -> np.ndarray:
        return np.diff(self.cum_good)

    def row_bad(self) -> np.ndarray:
        return np.diff(self.cum_bad)

    def _index(self, threshold: float | np.ndarray) -> np.ndarray:
        return np.searchsorted(self.scores, threshold, side="right")

    def counts_at(self, threshold: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (weighted good, weighted bad) approved at ``threshold``."""
        idx = self._index(threshold)
        return self.cum_good[idx], self.cum_bad[idx]

    def weight_at(self, threshold: float | np.ndarray) -> np.ndarray:
        """Population weight approved at ``threshold``."""
        return self.cum_weight[self._index(threshold)]

    def weighted_pd_at(self, threshold: float | np.ndarray) -> np.ndarray:
        """Sum of population weight * calibrated PD approved at ``threshold``."""
        return self.cum_weighted_pd[self._index(threshold)]

    def profit_at(self, threshold: float, margin: float, loss: float) -> float:
        good, bad = self.counts_at(threshold)
        return float(margin * good - loss * bad)

    def profit_curve(
        self, thresholds: np.ndarray, margin: float, loss: float
    ) -> np.ndarray:
        good, bad = self.counts_at(np.asarray(thresholds, dtype=float))
        return margin * good - loss * bad

    def expected_loss_at(self, threshold: float, loss: float) -> float:
        return float(loss * self.weighted_pd_at(threshold))

    def approved_share_at(self, threshold: float) -> float:
        return float(self.weight_at(threshold) / self.total_weight)

    def bad_rate_at(self, threshold: float) -> float:
        """Population-weighted default rate among approved accounts."""
        weight = self.weight_at(threshold)
        if weight <= 0:
            return 0.0
        _, bad = self.counts_at(threshold)
        # Convert the weighted bad count back to a probability: bad count is
        # already the weighted number of defaults; the approved weight is the
        # population weight of the approved segment.
        return float(bad / weight)


def compute_profit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    margin: float,
    loss: float,
    neg_weight: float,
) -> float:
    """Expected profit at one threshold (backwards-compatible helper)."""
    distribution = ScoreDistribution.from_labels(y_true, y_prob, neg_weight)
    return distribution.profit_at(threshold, margin, loss)


def threshold_grid(
    scores: np.ndarray,
    log_min: float = 1e-5,
    log_max: float = 0.99,
    log_points: int = 400,
    quantile_points: int = 200,
    include: Iterable[float] = (),
) -> np.ndarray:
    """A grid that covers both the extreme low-PD region and the bulk of scores."""
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        raise ValueError("scores is empty")

    log_grid = np.logspace(np.log10(log_min), np.log10(log_max), log_points)
    quantiles = np.quantile(scores, np.linspace(0.001, 0.999, quantile_points))
    extras = np.asarray(list(include), dtype=float)
    grid = np.unique(np.concatenate([log_grid, quantiles, extras]))
    grid = grid[(grid > 0) & (grid < 1)]
    if len(grid) < 2:
        raise ValueError("threshold grid collapsed to fewer than two points")
    return grid


def plot_profit_curves(scenario_profits: dict, output_dir: str | Path) -> None:
    """Plot profit vs cutoff for each economic scenario."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in scenario_profits.items():
        thresholds = np.asarray(data["thresholds"])
        profits = np.asarray(data["profits"])
        color = SCENARIO_COLORS.get(name.lower(), "#333333")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(thresholds, profits / 1e6, label=f"{name.title()} scenario", color=color, linewidth=2)
        ax.axvline(
            data["empirical_opt"],
            color="#111111",
            linestyle="--",
            linewidth=1.5,
            label=f"Empirical optimum ({data['empirical_opt']:.4f})",
        )
        ax.set_title(
            f"Expected profit: {name.title()} scenario "
            f"(r = {data['loss'] / data['margin']:.1f})"
        )
        ax.set_xlabel("Probability cutoff (approve if PD <= cutoff)")
        ax.set_ylabel("Expected profit ($ millions)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best", frameon=False)

        # The economically interesting region sits below 10% PD, where the
        # linear axis compresses the peak into a few pixels.
        zoom_mask = thresholds <= 0.1
        if zoom_mask.sum() >= 2:
            inset = ax.inset_axes([0.56, 0.10, 0.40, 0.32])
            inset.plot(
                thresholds[zoom_mask],
                profits[zoom_mask] / 1e6,
                color=color,
                linewidth=1.8,
            )
            inset.axvline(
                data["empirical_opt"],
                color="#111111",
                linestyle="--",
                linewidth=1.2,
            )
            inset.set_xlim(0, 0.1)
            inset.set_title("Low-PD zoom", fontsize=9)
            inset.tick_params(labelsize=8)
            inset.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        filename = f"profit_curve_{name.lower().replace(' ', '_')}.png"
        fig.savefig(output_dir / filename, dpi=300)
        plt.close(fig)
