"""Three-action credit-line policy: Grow / Hold / Restrict.

The policy maps a calibrated probability of default (PD) to an action using
two cutoffs:

    PD <= grow_cutoff                  -> grow
    grow_cutoff < PD <= restrict_cutoff -> hold
    PD > restrict_cutoff               -> restrict (exit)

For a scenario with base margin ``M``, base loss ``L = lgd * exposure`` and
action multipliers ``(alpha_a, beta_a)``:

    margin_a = alpha_a * M
    loss_a   = beta_a * L
    profit_a = margin_a * good - loss_a * bad

The robust policy minimizes the worst-case fractional regret relative to the
best policy in the same class for each scenario, subject to portfolio
guardrails (grow share, hold share, expected-loss increase).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.economics.profit_curve import ScoreDistribution

ACTION_ORDER = ("grow", "hold", "restrict")


@dataclass(frozen=True)
class ActionEconomics:
    name: str
    label: str
    margin: float
    loss: float
    exposure: float
    margin_multiplier: float
    exposure_multiplier: float

    @classmethod
    def from_scenario(
        cls,
        scenario: dict[str, Any],
        action_name: str,
        action_config: dict[str, Any],
    ) -> "ActionEconomics":
        base_margin = float(scenario["margin"])
        base_loss = float(scenario["lgd"]) * float(scenario["exposure"])
        base_exposure = float(scenario["exposure"])
        margin_multiplier = float(action_config["margin_multiplier"])
        exposure_multiplier = float(action_config["exposure_multiplier"])
        return cls(
            name=action_name,
            label=str(action_config.get("label", action_name.title())),
            margin=base_margin * margin_multiplier,
            loss=base_loss * exposure_multiplier,
            exposure=base_exposure * exposure_multiplier,
            margin_multiplier=margin_multiplier,
            exposure_multiplier=exposure_multiplier,
        )


@dataclass
class PolicyOutcome:
    threshold: float
    profit: float
    regret: float
    expected_loss: float
    exposure: float
    action_mix: dict[str, float]
    bad_rate_by_action: dict[str, float]
    counts: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": float(self.threshold),
            "profit": float(self.profit),
            "regret": float(self.regret),
            "expected_loss": float(self.expected_loss),
            "exposure": float(self.exposure),
            "action_mix": {k: float(v) for k, v in self.action_mix.items()},
            "bad_rate_by_action": {
                k: float(v) for k, v in self.bad_rate_by_action.items()
            },
            "counts": {k: float(v) for k, v in self.counts.items()},
        }


@dataclass
class ActionPolicyOutcome:
    grow_cutoff: float
    restrict_cutoff: float
    profit: float
    regret: float
    expected_loss: float
    exposure: float
    action_mix: dict[str, float]
    bad_rate_by_action: dict[str, float]
    counts: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "grow_cutoff": float(self.grow_cutoff),
            "restrict_cutoff": float(self.restrict_cutoff),
            "profit": float(self.profit),
            "regret": float(self.regret),
            "expected_loss": float(self.expected_loss),
            "exposure": float(self.exposure),
            "action_mix": {k: float(v) for k, v in self.action_mix.items()},
            "bad_rate_by_action": {
                k: float(v) for k, v in self.bad_rate_by_action.items()
            },
            "counts": {k: float(v) for k, v in self.counts.items()},
        }


@dataclass
class ScenarioPolicyMatrix:
    scenario: str
    actions: dict[str, ActionEconomics]
    profit: np.ndarray
    expected_loss: np.ndarray
    exposure: np.ndarray
    grow_share: np.ndarray
    hold_share: np.ndarray
    oracle_profit: float = 0.0
    best_pair: tuple[float, float] = (0.0, 0.0)

    @property
    def regret(self) -> np.ndarray:
        if self.oracle_profit <= 0:
            return np.zeros_like(self.profit)
        return np.where(
            np.isfinite(self.profit),
            (self.oracle_profit - self.profit) / self.oracle_profit,
            np.inf,
        )


@dataclass
class RobustActionPolicy:
    grow_cutoff: float
    restrict_cutoff: float
    worst_case_regret: float
    scenarios: dict[str, ScenarioPolicyMatrix] = field(default_factory=dict)
    outcomes: dict[str, ActionPolicyOutcome] = field(default_factory=dict)
    binary_baseline: dict[str, PolicyOutcome] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "grow_cutoff": float(self.grow_cutoff),
            "restrict_cutoff": float(self.restrict_cutoff),
            "worst_case_regret": float(self.worst_case_regret),
            "scenarios": {
                name: {
                    "actions": {
                        action: {
                            "label": econ.label,
                            "margin": float(econ.margin),
                            "loss": float(econ.loss),
                            "exposure": float(econ.exposure),
                            "margin_multiplier": float(econ.margin_multiplier),
                            "exposure_multiplier": float(econ.exposure_multiplier),
                        }
                        for action, econ in matrix.actions.items()
                    },
                    "oracle_profit": float(matrix.oracle_profit),
                    "best_pair": [float(matrix.best_pair[0]), float(matrix.best_pair[1])],
                }
                for name, matrix in self.scenarios.items()
            },
            "outcomes": {
                name: outcome.to_dict() for name, outcome in self.outcomes.items()
            },
            "binary_baseline": {
                name: outcome.to_dict()
                for name, outcome in self.binary_baseline.items()
            },
        }


def build_action_economics(
    scenario: dict[str, Any],
    action_config: dict[str, dict[str, Any]],
) -> dict[str, ActionEconomics]:
    return {
        name: ActionEconomics.from_scenario(scenario, name, action_config[name])
        for name in ACTION_ORDER
    }


def _matrix_terms(
    distribution: ScoreDistribution, grid: np.ndarray
) -> dict[str, np.ndarray]:
    good, bad = distribution.counts_at(grid)
    weight = distribution.weight_at(grid)
    weighted_pd = distribution.weighted_pd_at(grid)
    return {
        "good": np.asarray(good),
        "bad": np.asarray(bad),
        "weight": np.asarray(weight),
        "weighted_pd": np.asarray(weighted_pd),
    }


def build_scenario_matrix(
    distribution: ScoreDistribution,
    scenario_name: str,
    scenario: dict[str, Any],
    action_config: dict[str, dict[str, Any]],
    grid: np.ndarray,
    constraints: dict[str, float],
    baseline_expected_loss: float,
) -> ScenarioPolicyMatrix:
    """Build the profit/expected-loss/exposure matrices for one scenario."""
    actions = build_action_economics(scenario, action_config)
    terms = _matrix_terms(distribution, grid)

    grow = actions["grow"]
    hold = actions["hold"]
    restrict = actions["restrict"]

    good_1 = terms["good"][:, None]
    bad_1 = terms["bad"][:, None]
    weight_1 = terms["weight"][:, None]
    wpd_1 = terms["weighted_pd"][:, None]

    good_2 = terms["good"][None, :]
    bad_2 = terms["bad"][None, :]
    weight_2 = terms["weight"][None, :]
    wpd_2 = terms["weighted_pd"][None, :]

    total_good = distribution.total_good
    total_bad = distribution.total_bad
    total_weight = distribution.total_weight
    total_wpd = distribution.total_weighted_pd

    profit = (
        grow.margin * good_1
        - grow.loss * bad_1
        + hold.margin * (good_2 - good_1)
        - hold.loss * (bad_2 - bad_1)
        + restrict.margin * (total_good - good_2)
        - restrict.loss * (total_bad - bad_2)
    )

    expected_loss = (
        grow.loss * wpd_1
        + hold.loss * (wpd_2 - wpd_1)
        + restrict.loss * (total_wpd - wpd_2)
    )

    exposure = (
        grow.exposure * weight_1
        + hold.exposure * (weight_2 - weight_1)
        + restrict.exposure * (total_weight - weight_2)
    )

    grow_share = weight_1 / total_weight
    hold_share = (weight_2 - weight_1) / total_weight

    valid = grid[:, None] <= grid[None, :]
    valid &= grow_share <= float(constraints["max_grow_share"])
    valid &= hold_share >= float(constraints["min_hold_share"])
    if baseline_expected_loss > 0:
        max_loss = baseline_expected_loss * (
            1.0 + float(constraints["max_expected_loss_increase"])
        )
        valid &= expected_loss <= max_loss

    profit = np.where(valid, profit, -np.inf)
    expected_loss = np.where(valid, expected_loss, np.inf)
    exposure = np.where(valid, exposure, np.inf)

    matrix = ScenarioPolicyMatrix(
        scenario=scenario_name,
        actions=actions,
        profit=profit,
        expected_loss=expected_loss,
        exposure=exposure,
        grow_share=np.broadcast_to(grow_share, profit.shape).copy(),
        hold_share=np.broadcast_to(hold_share, profit.shape).copy(),
    )
    if np.isfinite(profit).any():
        flat_index = int(np.nanargmax(profit))
        row, col = np.unravel_index(flat_index, profit.shape)
        matrix.oracle_profit = float(profit[row, col])
        matrix.best_pair = (float(grid[row]), float(grid[col]))
    return matrix


def _outcome_from_cutoffs(
    distribution: ScoreDistribution,
    scenario: dict[str, Any],
    actions: dict[str, ActionEconomics],
    grow_cutoff: float,
    restrict_cutoff: float,
    oracle_profit: float,
) -> ActionPolicyOutcome:
    good_grow, bad_grow = distribution.counts_at(grow_cutoff)
    good_restrict, bad_restrict = distribution.counts_at(restrict_cutoff)
    weight_grow = distribution.weight_at(grow_cutoff)
    weight_restrict = distribution.weight_at(restrict_cutoff)

    good_hold = good_restrict - good_grow
    bad_hold = bad_restrict - bad_grow
    weight_hold = weight_restrict - weight_grow

    total_good = distribution.total_good
    total_bad = distribution.total_bad
    total_weight = distribution.total_weight

    grow = actions["grow"]
    hold = actions["hold"]
    restrict = actions["restrict"]

    profit = (
        grow.margin * good_grow
        - grow.loss * bad_grow
        + hold.margin * good_hold
        - hold.loss * bad_hold
        + restrict.margin * (total_good - good_restrict)
        - restrict.loss * (total_bad - bad_restrict)
    )

    wpd_grow = distribution.weighted_pd_at(grow_cutoff)
    wpd_restrict = distribution.weighted_pd_at(restrict_cutoff)
    expected_loss = (
        grow.loss * wpd_grow
        + hold.loss * (wpd_restrict - wpd_grow)
        + restrict.loss * (distribution.total_weighted_pd - wpd_restrict)
    )

    weight_grow_total = distribution.weight_at(grow_cutoff)
    weight_restrict_total = distribution.weight_at(restrict_cutoff)
    exposure = (
        grow.exposure * weight_grow_total
        + hold.exposure * (weight_restrict_total - weight_grow_total)
        + restrict.exposure * (total_weight - weight_restrict_total)
    )

    def _share(value: float) -> float:
        return float(value / total_weight) if total_weight else 0.0

    def _bad_rate(bad: float, weight: float) -> float:
        return float(bad / weight) if weight > 0 else 0.0

    regret = (
        float((oracle_profit - profit) / oracle_profit)
        if oracle_profit > 0
        else 0.0
    )

    return ActionPolicyOutcome(
        grow_cutoff=float(grow_cutoff),
        restrict_cutoff=float(restrict_cutoff),
        profit=float(profit),
        regret=regret,
        expected_loss=float(expected_loss),
        exposure=float(exposure),
        action_mix={
            "grow": _share(weight_grow),
            "hold": _share(weight_hold),
            "restrict": _share(total_weight - weight_restrict),
        },
        bad_rate_by_action={
            "grow": _bad_rate(bad_grow, weight_grow),
            "hold": _bad_rate(bad_hold, weight_hold),
            "restrict": _bad_rate(
                total_bad - bad_restrict, total_weight - weight_restrict
            ),
        },
        counts={
            "good_grow": float(good_grow),
            "bad_grow": float(bad_grow),
            "good_hold": float(good_hold),
            "bad_hold": float(bad_hold),
            "good_restrict": float(total_good - good_restrict),
            "bad_restrict": float(total_bad - bad_restrict),
        },
    )


def action_profit_at(
    distribution: ScoreDistribution,
    grow_cutoff: float,
    restrict_cutoff: float,
    margin: float,
    loss: float,
    action_config: dict[str, dict[str, Any]],
) -> float:
    """Profit of the three-action policy for an arbitrary margin/loss pair."""
    good_grow, bad_grow = distribution.counts_at(grow_cutoff)
    good_restrict, bad_restrict = distribution.counts_at(restrict_cutoff)

    grow_margin = margin * float(action_config["grow"]["margin_multiplier"])
    grow_loss = loss * float(action_config["grow"]["exposure_multiplier"])
    hold_margin = margin * float(action_config["hold"]["margin_multiplier"])
    hold_loss = loss * float(action_config["hold"]["exposure_multiplier"])
    restrict_margin = margin * float(
        action_config["restrict"]["margin_multiplier"]
    )
    restrict_loss = loss * float(
        action_config["restrict"]["exposure_multiplier"]
    )

    return float(
        grow_margin * good_grow
        - grow_loss * bad_grow
        + hold_margin * (good_restrict - good_grow)
        - hold_loss * (bad_restrict - bad_grow)
        + restrict_margin * (distribution.total_good - good_restrict)
        - restrict_loss * (distribution.total_bad - bad_restrict)
    )


def binary_outcome(
    distribution: ScoreDistribution,
    threshold: float,
    margin: float,
    loss: float,
    exposure_value: float,
) -> PolicyOutcome:
    """Evaluate the current binary approve/decline policy."""
    good, bad = distribution.counts_at(threshold)
    weight = distribution.weight_at(threshold)
    profit = float(margin * good - loss * bad)
    expected_loss = float(loss * distribution.weighted_pd_at(threshold))
    exposure = float(exposure_value * weight)
    total_weight = distribution.total_weight
    action_mix = {
        "approve": float(weight / total_weight),
        "decline": float(1.0 - weight / total_weight),
    }
    bad_rate = float(bad / weight) if weight > 0 else 0.0
    return PolicyOutcome(
        threshold=float(threshold),
        profit=profit,
        regret=0.0,
        expected_loss=expected_loss,
        exposure=exposure,
        action_mix=action_mix,
        bad_rate_by_action={"approve": bad_rate},
        counts={"good": float(good), "bad": float(bad)},
    )


def optimize_robust_action_policy(
    distribution: ScoreDistribution,
    scenarios: dict[str, dict[str, Any]],
    action_config: dict[str, dict[str, Any]],
    constraints: dict[str, float],
    grid: np.ndarray,
    baseline_expected_loss: dict[str, float],
) -> RobustActionPolicy:
    """Find cutoffs that minimize worst-case regret across scenarios."""
    matrices: dict[str, ScenarioPolicyMatrix] = {}
    regret_arrays: list[np.ndarray] = []

    for name, scenario in scenarios.items():
        matrix = build_scenario_matrix(
            distribution,
            name,
            scenario,
            action_config,
            grid,
            constraints,
            baseline_expected_loss=float(baseline_expected_loss[name]),
        )
        if not np.isfinite(matrix.profit).any():
            raise RuntimeError(
                f"No feasible three-action policy for scenario '{name}'. "
                "Relax config/economic_scenarios.yaml constraints."
            )
        matrices[name] = matrix
        regret_arrays.append(matrix.regret)

    worst_case = np.max(np.stack(regret_arrays), axis=0)
    flat_index = int(np.nanargmin(worst_case))
    row, col = np.unravel_index(flat_index, worst_case.shape)
    grow_cutoff = float(grid[row])
    restrict_cutoff = float(grid[col])

    policy = RobustActionPolicy(
        grow_cutoff=grow_cutoff,
        restrict_cutoff=restrict_cutoff,
        worst_case_regret=float(worst_case[row, col]),
        scenarios=matrices,
    )

    for name, matrix in matrices.items():
        policy.outcomes[name] = _outcome_from_cutoffs(
            distribution,
            scenarios[name],
            matrix.actions,
            grow_cutoff,
            restrict_cutoff,
            oracle_profit=matrix.oracle_profit,
        )
    return policy


def binary_minimax_threshold(
    distribution: ScoreDistribution,
    scenarios: dict[str, dict[str, Any]],
    grid: np.ndarray,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Minimax threshold for the current binary approve/decline policy."""
    regret_arrays: list[np.ndarray] = []
    profits: dict[str, np.ndarray] = {}
    for name, scenario in scenarios.items():
        margin = float(scenario["margin"])
        loss = float(scenario["lgd"]) * float(scenario["exposure"])
        curve = distribution.profit_curve(grid, margin, loss)
        profits[name] = curve
        best = float(np.max(curve))
        regret_arrays.append(
            (best - curve) / best if best > 0 else np.zeros_like(curve)
        )

    worst_case = np.max(np.stack(regret_arrays), axis=0)
    index = int(np.nanargmin(worst_case))
    threshold = float(grid[index])
    regrets = {
        name: float(regret_arrays[i][index]) for i, name in enumerate(scenarios)
    }
    absolute = {
        name: float(profits[name][index]) for name in scenarios
    }
    return threshold, regrets, absolute
