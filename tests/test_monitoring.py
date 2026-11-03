import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_flags import evaluate_drift_triggers
from src.monitoring.psi import (
    action_mix,
    calculate_psi,
    categorical_psi,
    compute_feature_drift,
    status_for,
)


def _drift_frame(n=600):
    rng = np.random.default_rng(7)
    reference = pd.DataFrame(
        {
            "num": rng.normal(0.0, 1.0, n),
            "cat": np.where(rng.random(n) < 0.8, "A", "B"),
            "missing": np.where(rng.random(n) < 0.05, np.nan, 1.0),
        }
    )
    current = pd.DataFrame(
        {
            "num": rng.normal(0.8, 1.0, n),
            "cat": np.where(rng.random(n) < 0.5, "A", "B"),
            "missing": np.where(rng.random(n) < 0.30, np.nan, 1.0),
        }
    )
    return reference, current


def test_weighted_psi_responds_to_weights():
    values = np.array([0.01, 0.01, 0.5, 0.5])
    unweighted = calculate_psi(values, values)
    weighted = calculate_psi(
        values,
        values,
        expected_weight=np.array([20.0, 20.0, 1.0, 1.0]),
        actual_weight=np.array([20.0, 20.0, 1.0, 1.0]),
    )
    assert unweighted == pytest.approx(0.0)
    assert weighted == pytest.approx(0.0)


def test_categorical_psi_detects_shift_and_handles_missing():
    reference = pd.Series(["A"] * 90 + ["B"] * 10)
    same = pd.Series(["A"] * 90 + ["B"] * 10)
    shifted = pd.Series(["A"] * 40 + ["B"] * 60)
    assert categorical_psi(reference, same) == pytest.approx(0.0)
    assert categorical_psi(reference, shifted) > 0.1

    with_missing = pd.Series(["A"] * 50 + [None] * 50)
    assert np.isfinite(categorical_psi(reference, with_missing))


def test_compute_feature_drift_status_and_missingness():
    reference, current = _drift_frame()
    drift = compute_feature_drift(
        reference, current, ["num", "cat", "missing"], ["cat"], bins=10
    ).set_index("feature")

    assert drift.loc["num", "status"] in {"yellow", "red"}
    assert drift.loc["num", "psi"] > 0.1
    assert drift.loc["cat", "status"] in {"yellow", "red"}
    assert drift.loc["missing", "missing_delta"] > 0.2
    assert drift.loc["missing", "status"] == "red"


def test_compute_feature_drift_marks_insufficient_data():
    reference = pd.DataFrame({"x": [np.nan] * 500})
    current = pd.DataFrame({"x": [np.nan] * 500})
    drift = compute_feature_drift(reference, current, ["x"], [])
    assert drift.iloc[0]["status"] == "insufficient_data"


def test_status_thresholds():
    assert status_for(0.01, 0.0, True) == "green"
    assert status_for(0.15, 0.0, True) == "yellow"
    assert status_for(0.30, 0.0, True) == "red"
    assert status_for(0.01, 0.08, True) == "yellow"
    assert status_for(0.01, 0.20, True) == "red"
    assert status_for(0.01, 0.0, False) == "insufficient_data"


def test_action_mix_uses_population_weights():
    scores = np.array([0.01, 0.02, 0.5, 0.9])
    weights = np.array([20.0, 20.0, 1.0, 1.0])
    mix = action_mix(scores, 0.1, 0.8, weights=weights)
    assert mix["grow"] == pytest.approx(40.0 / 42.0)
    assert mix["hold"] == pytest.approx(1.0 / 42.0)
    assert mix["restrict"] == pytest.approx(1.0 / 42.0)
    assert sum(mix.values()) == pytest.approx(1.0)


def _psi_frame(rows):
    return pd.DataFrame(rows, columns=["feature", "family", "psi"])


def test_drift_triggers_quiet_when_nothing_changed():
    frame = _psi_frame([(f"P_{index}", "Payment", 0.01) for index in range(200)])
    result = evaluate_drift_triggers(
        frame,
        score_psi=0.01,
        action_mix_delta={"grow": 0.001, "hold": -0.001, "restrict": 0.0},
    )
    assert result["retrain_review"] is False
    assert result["reasons"] == []


def test_drift_triggers_global_and_critical_families():
    rows = [(f"P_{index}", "Payment", 0.01) for index in range(170)]
    rows += [(f"D_{index}", "Delinquency", 0.40) for index in range(3)]
    rows += [(f"B_{index}", "Balance", 0.40) for index in range(27)]
    result = evaluate_drift_triggers(_psi_frame(rows))
    assert result["retrain_review"] is True
    assert result["features_above_threshold"] == 30
    assert any("Global drift" in reason for reason in result["reasons"])
    assert any("Critical family drift" in reason for reason in result["reasons"])


def test_drift_triggers_score_and_action_mix():
    frame = _psi_frame([(f"P_{index}", "Payment", 0.01) for index in range(200)])
    score_result = evaluate_drift_triggers(frame, score_psi=0.30)
    assert score_result["retrain_review"] is True
    assert any("Score drift" in reason for reason in score_result["reasons"])

    mix_result = evaluate_drift_triggers(
        frame, action_mix_delta={"grow": 0.08, "hold": -0.04, "restrict": -0.04}
    )
    assert mix_result["retrain_review"] is True
    assert any("Action-mix drift" in reason for reason in mix_result["reasons"])


def test_drift_triggers_empty_frame_is_quiet():
    result = evaluate_drift_triggers(pd.DataFrame(columns=["feature", "family", "psi"]))
    assert result["retrain_review"] is False
    assert result["features_above_threshold"] == 0
