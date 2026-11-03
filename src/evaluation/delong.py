"""Corrected DeLong test for paired ROC-AUC differences.

The original implementation in this repository omitted the normalisation of
the placement values, which inflated the covariance matrix and produced
p-values near 1 even for clearly different models. This module implements the
standard fast DeLong algorithm (Sun & Xu, 2014) with the correct scaling:

    V10_i = (# negatives below positive i) / n
    V01_j = (# positives above negative j) / m
    S     = cov(V10) / m + cov(V01) / n

The DeLong test treats the sample as unweighted. Because the Amex sample uses
a 20x weight on negatives, the decision-relevant uncertainty is estimated with
a paired, class-stratified bootstrap that carries the sample weights. Both are
reported; neither is silently substituted for the other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def compute_midrank(x: np.ndarray) -> np.ndarray:
    """Return 1-based midranks for ties in ``x``."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1)
        i = j
    result = np.empty(n, dtype=float)
    result[order] = ranks + 1.0
    return result


def fast_delong(
    predictions_sorted_transposed: np.ndarray, label_1_count: int
) -> np.ndarray:
    """Covariance matrix of AUC estimates using the fast DeLong algorithm.

    Parameters
    ----------
    predictions_sorted_transposed:
        Shape ``(n_models, n_samples)``. Rows are model scores. Columns must be
        sorted with positives first.
    label_1_count:
        Number of positive examples, ``m``.
    """
    predictions_sorted_transposed = np.asarray(
        predictions_sorted_transposed, dtype=float
    )
    if predictions_sorted_transposed.ndim == 1:
        predictions_sorted_transposed = predictions_sorted_transposed[None, :]

    m = int(label_1_count)
    n = predictions_sorted_transposed.shape[1] - m
    if m <= 0 or n <= 0:
        raise ValueError("DeLong requires at least one positive and one negative")

    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)
    for row in range(k):
        tz[row, :] = compute_midrank(predictions_sorted_transposed[row, :])
        tx[row, :] = compute_midrank(positive_examples[row, :])
        ty[row, :] = compute_midrank(negative_examples[row, :])

    # Placement values. The division is the correction the old code missed.
    v10 = (tz[:, :m] - tx) / n
    v01 = (ty - tz[:, m:]) / m

    # np.cov returns a scalar for a single model; force 2-D for the caller.
    s10 = np.atleast_2d(np.cov(v10, rowvar=True))
    s01 = np.atleast_2d(np.cov(v01, rowvar=True))
    return s10 / m + s01 / n


def calc_pvalue(aucs: np.ndarray, sigma: np.ndarray) -> tuple[float, float]:
    """Two-sided p-value and z-statistic for a difference of two AUCs."""
    aucs = np.asarray(aucs, dtype=float).ravel()
    if aucs.size != 2:
        raise ValueError("calc_pvalue expects exactly two AUCs")
    contrast = np.array([[1.0, -1.0]])
    variance = float((contrast @ sigma @ contrast.T)[0, 0])
    if variance <= 0:
        # Identical predictions can produce a zero variance; the observed
        # difference is then also zero and the test is undefined. Return a
        # conservative p-value of 1 rather than dividing by zero.
        return 1.0, 0.0
    z_value = float(abs(aucs[1] - aucs[0]) / np.sqrt(variance))
    p_value = float(2.0 * (1.0 - stats.norm.cdf(z_value)))
    return p_value, z_value


def delong_roc_test(
    ground_truth: np.ndarray,
    predictions_one: np.ndarray,
    predictions_two: np.ndarray,
) -> float:
    """Return the two-sided DeLong p-value for two paired AUCs.

    The test is unweighted. Use :func:`bootstrap_auc_difference` for the
    sample-weighted uncertainty used in the Amex analysis.
    """
    ground_truth = np.asarray(ground_truth).astype(int).ravel()
    predictions_one = np.asarray(predictions_one, dtype=float).ravel()
    predictions_two = np.asarray(predictions_two, dtype=float).ravel()
    if not (
        len(ground_truth) == len(predictions_one) == len(predictions_two)
    ):
        raise ValueError("ground_truth and predictions must have equal length")
    if set(np.unique(ground_truth)) - {0, 1}:
        raise ValueError("ground_truth must be binary")

    # Sort positives first; ties are stable.
    order = np.argsort(ground_truth, kind="mergesort")[::-1]
    y_sorted = ground_truth[order]
    preds = np.vstack([predictions_one[order], predictions_two[order]])

    auc_one = roc_auc_score(y_sorted, preds[0])
    auc_two = roc_auc_score(y_sorted, preds[1])
    sigma = fast_delong(preds, int(y_sorted.sum()))
    p_value, _ = calc_pvalue(np.array([auc_one, auc_two]), sigma)
    return p_value


@dataclass(frozen=True)
class BootstrapResult:
    observed_difference: float
    ci_low: float
    ci_high: float
    p_value: float
    n_boot: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "observed_difference": self.observed_difference,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_value": self.p_value,
            "n_boot": self.n_boot,
        }


def bootstrap_auc_difference(
    ground_truth: np.ndarray,
    predictions_one: np.ndarray,
    predictions_two: np.ndarray,
    sample_weight: np.ndarray | None = None,
    n_boot: int = 500,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Paired, class-stratified bootstrap for the AUC difference.

    Resampling is stratified by class so every replicate has the same number of
    positives and negatives as the original sample. Sample weights are carried
    with the resampled rows. The p-value is the two-sided bootstrap
    percentile p-value for a difference of zero.
    """
    y = np.asarray(ground_truth).astype(int).ravel()
    pred_a = np.asarray(predictions_one, dtype=float).ravel()
    pred_b = np.asarray(predictions_two, dtype=float).ravel()
    if not (len(y) == len(pred_a) == len(pred_b)):
        raise ValueError("ground_truth and predictions must have equal length")
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("ground_truth must be binary")
    if sample_weight is None:
        weight = np.ones(len(y), dtype=float)
    else:
        weight = np.asarray(sample_weight, dtype=float).ravel()
        if len(weight) != len(y):
            raise ValueError("sample_weight must match ground_truth length")
        if (weight <= 0).any():
            raise ValueError("sample_weight must be strictly positive")

    rng = np.random.default_rng(seed)
    positive_idx = np.flatnonzero(y == 1)
    negative_idx = np.flatnonzero(y == 0)
    if len(positive_idx) == 0 or len(negative_idx) == 0:
        raise ValueError("bootstrap requires both classes")

    observed = float(
        roc_auc_score(y, pred_b, sample_weight=weight)
        - roc_auc_score(y, pred_a, sample_weight=weight)
    )

    differences = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        pos = rng.choice(positive_idx, size=len(positive_idx), replace=True)
        neg = rng.choice(negative_idx, size=len(negative_idx), replace=True)
        idx = np.concatenate([pos, neg])
        y_boot = y[idx]
        w_boot = weight[idx]
        differences[i] = roc_auc_score(
            y_boot, pred_b[idx], sample_weight=w_boot
        ) - roc_auc_score(y_boot, pred_a[idx], sample_weight=w_boot)

    alpha = (1.0 - confidence) / 2.0
    ci_low, ci_high = np.quantile(differences, [alpha, 1.0 - alpha])
    p_value = 2.0 * min(
        float(np.mean(differences <= 0.0)),
        float(np.mean(differences >= 0.0)),
    )
    p_value = min(1.0, p_value)

    return BootstrapResult(
        observed_difference=observed,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float(p_value),
        n_boot=int(n_boot),
    )
