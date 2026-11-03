import numpy as np
import pytest

from src.evaluation.delong import (
    bootstrap_auc_difference,
    calc_pvalue,
    compute_midrank,
    delong_roc_test,
    fast_delong,
)


def _synthetic_scores(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    # Model A is weaker than model B but both are informative.
    latent = y * 1.2 + rng.normal(0, 1.0, size=n)
    pred_a = 1.0 / (1.0 + np.exp(-0.8 * latent + rng.normal(0, 0.9, size=n)))
    pred_b = 1.0 / (1.0 + np.exp(-1.1 * latent + rng.normal(0, 0.9, size=n)))
    return y, pred_a, pred_b


def test_compute_midrank_handles_ties():
    ranks = compute_midrank(np.array([0.1, 0.2, 0.2, 0.3]))
    assert np.allclose(ranks, [1.0, 2.5, 2.5, 4.0])


def test_fast_delong_is_symmetric_and_positive():
    y, pred_a, pred_b = _synthetic_scores()
    order = np.argsort(y)[::-1]
    preds = np.vstack([pred_a[order], pred_b[order]])
    sigma = fast_delong(preds, int(y.sum()))
    assert sigma.shape == (2, 2)
    assert np.allclose(sigma, sigma.T)
    assert np.all(np.diag(sigma) > 0)


def test_delong_detects_real_auc_difference():
    y, pred_a, pred_b = _synthetic_scores(n=6000, seed=1)
    p_value = delong_roc_test(y, pred_b, pred_a)
    assert 0.0 <= p_value <= 1.0
    assert p_value < 0.01


def test_delong_null_is_not_significant():
    y, pred_a, _ = _synthetic_scores(n=6000, seed=2)
    p_value = delong_roc_test(y, pred_a, pred_a.copy())
    assert p_value > 0.05


def test_calc_pvalue_zero_variance_is_conservative():
    p_value, z_value = calc_pvalue(
        np.array([0.7, 0.7]), np.zeros((2, 2), dtype=float)
    )
    assert p_value == 1.0
    assert z_value == 0.0


def test_bootstrap_auc_difference_returns_sensible_interval():
    y, pred_a, pred_b = _synthetic_scores(n=3000, seed=3)
    result = bootstrap_auc_difference(
        y, pred_a, pred_b, n_boot=120, seed=7
    )
    assert result.observed_difference > 0
    assert result.ci_low <= result.observed_difference <= result.ci_high
    assert 0.0 <= result.p_value <= 1.0
    assert result.n_boot == 120
    assert "ci_low" in result.to_dict()


def test_bootstrap_auc_difference_rejects_bad_input():
    with pytest.raises(ValueError):
        bootstrap_auc_difference(
            np.array([0, 1]), np.array([0.1]), np.array([0.2])
        )
    with pytest.raises(ValueError):
        bootstrap_auc_difference(
            np.array([0, 1]),
            np.array([0.1, 0.2]),
            np.array([0.2, 0.3]),
            sample_weight=np.array([1.0, 0.0]),
        )
