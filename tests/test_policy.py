import numpy as np
import pytest

from src.economics.profit_curve import ScoreDistribution, threshold_grid
from src.policy.actions import (
    ACTION_ORDER,
    action_profit_at,
    binary_minimax_threshold,
    build_action_economics,
    optimize_robust_action_policy,
)


def _toy_distribution():
    y = np.array([0, 0, 1, 0, 1, 0, 0, 0, 1, 0])
    p = np.array([0.01, 0.02, 0.03, 0.2, 0.25, 0.4, 0.6, 0.7, 0.8, 0.95])
    return y, p, ScoreDistribution.from_labels(y, p, negative_weight=20)


def _toy_scenarios():
    return {
        "low": {"margin": 100.0, "lgd": 0.2, "exposure": 100.0},
        "high": {"margin": 100.0, "lgd": 0.9, "exposure": 500.0},
    }


TOY_ACTIONS = {
    "grow": {"margin_multiplier": 1.1, "exposure_multiplier": 1.5},
    "hold": {"margin_multiplier": 1.0, "exposure_multiplier": 1.0},
    "restrict": {"margin_multiplier": 0.0, "exposure_multiplier": 0.0},
}


def test_score_distribution_threshold_is_inclusive():
    y, p, dist = _toy_distribution()
    good, bad = dist.counts_at(0.2)
    # 0.01, 0.02, 0.03 and 0.2 are at or below the threshold.
    assert bad == 1.0  # the 0.03 default
    assert good == 60.0  # three goods, each weighted 20x


def test_score_distribution_profit_matches_direct_formula():
    y, p, dist = _toy_distribution()
    mask = p <= 0.25
    expected = 100.0 * 20 * (y[mask] == 0).sum() - 500.0 * (y[mask] == 1).sum()
    assert dist.profit_at(0.25, 100.0, 500.0) == expected


def test_score_distribution_weighted_pd_and_share():
    y, p, dist = _toy_distribution()
    mask = p <= 0.25
    weights = np.where(y[mask] == 0, 20.0, 1.0)
    assert dist.weighted_pd_at(0.25) == pytest.approx((weights * p[mask]).sum())
    assert dist.approved_share_at(0.25) == pytest.approx(
        weights.sum() / dist.total_weight
    )


def test_threshold_grid_covers_low_and_high_scores():
    _, p, _ = _toy_distribution()
    grid = threshold_grid(p, log_points=40, quantile_points=20)
    assert grid.min() < 0.01
    assert grid.max() > 0.9
    assert np.all(np.diff(grid) > 0)


def test_action_economics_multipliers():
    actions = build_action_economics(_toy_scenarios()["high"], TOY_ACTIONS)
    assert actions["grow"].margin == pytest.approx(110.0)
    assert actions["grow"].loss == pytest.approx(0.9 * 500.0 * 1.5)
    assert actions["restrict"].margin == 0.0
    assert actions["restrict"].loss == 0.0
    assert set(actions) == set(ACTION_ORDER)


def test_action_policy_profit_matches_manual_segments():
    _, _, dist = _toy_distribution()
    profit = action_profit_at(
        dist, 0.05, 0.5, margin=100.0, loss=250.0, action_config=TOY_ACTIONS
    )
    good_grow, bad_grow = dist.counts_at(0.05)
    good_restrict, bad_restrict = dist.counts_at(0.5)
    expected = (
        110.0 * good_grow
        - 250.0 * 1.5 * bad_grow
        + 100.0 * (good_restrict - good_grow)
        - 250.0 * (bad_restrict - bad_grow)
    )
    assert profit == pytest.approx(expected)


def test_robust_action_policy_respects_constraints_and_mix():
    _, _, dist = _toy_distribution()
    grid = threshold_grid(dist.scores, log_points=80, quantile_points=40)
    scenarios = _toy_scenarios()
    baseline_loss = {
        name: scenario["lgd"] * scenario["exposure"] * dist.weighted_pd_at(0.5)
        for name, scenario in scenarios.items()
    }
    policy = optimize_robust_action_policy(
        dist,
        scenarios,
        TOY_ACTIONS,
        {
            "max_grow_share": 0.8,
            "min_hold_share": 0.05,
            "max_expected_loss_increase": 10.0,
        },
        grid,
        baseline_expected_loss=baseline_loss,
    )
    assert policy.grow_cutoff <= policy.restrict_cutoff
    assert policy.worst_case_regret >= 0
    for name, outcome in policy.outcomes.items():
        assert outcome.action_mix["grow"] <= 0.8 + 1e-9
        assert outcome.action_mix["hold"] >= 0.05 - 1e-9
        assert sum(outcome.action_mix.values()) == pytest.approx(1.0)
        assert outcome.profit > 0
        assert 0 <= outcome.regret <= 1


def test_binary_minimax_threshold_is_on_grid():
    _, _, dist = _toy_distribution()
    grid = threshold_grid(dist.scores, log_points=60, quantile_points=30)
    threshold, regrets, profits = binary_minimax_threshold(
        dist, _toy_scenarios(), grid
    )
    assert threshold in grid
    assert set(regrets) == {"low", "high"}
    assert set(profits) == {"low", "high"}
    assert all(value >= 0 for value in regrets.values())
