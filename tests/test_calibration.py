import numpy as np
import pytest

from src.models.calibration import calibration_bins, compute_ece


def test_calibration_bins_weights_sum_to_one():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 2000)
    p = np.clip(rng.beta(2, 8, 2000), 1e-6, 1 - 1e-6)
    weights = np.where(y == 0, 20.0, 1.0)

    bins = calibration_bins(y, p, weights, n_bins=10, strategy="quantile")
    assert not bins.empty
    assert bins["weight_share"].sum() == pytest.approx(1.0)
    assert (bins["count"] > 0).all()
    assert (bins["upper"] >= bins["lower"]).all()


def test_compute_ece_is_lower_for_calibrated_predictions():
    rng = np.random.default_rng(11)
    y = rng.integers(0, 2, 5000)
    # Overconfident predictions: rank-informative but systematically too high.
    p = np.where(y == 1, 0.95, 0.40)
    p = np.clip(p + rng.normal(0, 0.02, len(p)), 1e-6, 1 - 1e-6)
    weights = np.where(y == 0, 20.0, 1.0)

    raw_ece = compute_ece(y, p, weights, n_bins=10, strategy="quantile")
    calibrated = np.where(y == 1, 0.95, 0.05)
    calibrated_ece = compute_ece(
        y, calibrated, weights, n_bins=10, strategy="quantile"
    )
    assert raw_ece > calibrated_ece


def test_calibration_bins_handles_constant_scores():
    y = np.array([0, 1, 0, 1])
    p = np.full(4, 0.3)
    weights = np.where(y == 0, 20.0, 1.0)
    bins = calibration_bins(y, p, weights, n_bins=10, strategy="quantile")
    assert len(bins) >= 1
    assert bins["weight_share"].sum() == pytest.approx(1.0)


def test_compute_ece_rejects_bad_strategy():
    with pytest.raises(ValueError):
        compute_ece(
            np.array([0, 1]),
            np.array([0.2, 0.8]),
            np.array([20.0, 1.0]),
            strategy="not-a-strategy",
        )
