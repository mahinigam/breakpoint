import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"


def _load(name):
    path = REPORTS / name
    if not path.exists():
        pytest.skip(f"{path} is not present")
    return json.loads(path.read_text(encoding="utf-8"))


def test_metrics_are_plausible_and_population_weighted():
    metrics = _load("metrics.json")
    training = metrics["training"]
    evaluation = metrics["evaluation"]
    monitoring = metrics["monitoring"]

    assert training["dataset_rows"] == 458913
    assert training["n_features"] == 200
    assert training["negative_weight"] == 20.0
    assert 0.9 < evaluation["calibrated_metrics"]["AUC"] < 1.0
    assert 0.0 < training["posterior_default_rate"] < 0.05
    assert training["posterior_default_rate"] != pytest.approx(0.2589, abs=0.001)

    assert monitoring["reference_default_rate"] < 0.05
    assert monitoring["current_default_rate"] < 0.05
    assert monitoring["score_psi"] >= 0.0
    assert isinstance(monitoring["retrain_trigger"]["retrain_review"], bool)


def test_policy_respects_guardrails_and_beats_binary():
    policy = _load("policy.json")
    action = policy["action_policy"]
    constraints = action["constraints"]

    assert action["grow_cutoff"] < action["restrict_cutoff"]
    assert action["worst_case_regret"] < policy["binary_minimax"]["max_regret"]

    for scenario in action["scenarios"].values():
        mix = scenario["action_outcome"]["action_mix"]
        assert sum(mix.values()) == pytest.approx(1.0, abs=1e-9)
        assert mix["grow"] <= constraints["max_grow_share"] + 1e-9
        assert mix["hold"] >= constraints["min_hold_share"] - 1e-9
        assert scenario["expected_loss_change"] <= constraints[
            "max_expected_loss_increase"
        ] + 1e-9
        assert scenario["action_outcome"]["profit"] > scenario["binary_profit"]


def test_bootstrap_intervals_bracket_observed_differences():
    bootstrap = _load("bootstrap.json")
    assert bootstrap["n_boot"] >= 100
    for result in bootstrap["results"].values():
        for comparison in result.values():
            assert comparison["ci_low"] <= comparison["observed_difference"]
            assert comparison["observed_difference"] <= comparison["ci_high"]
            assert 0.0 <= comparison["p_value"] <= 1.0


def test_power_analysis_is_internally_consistent():
    power = _load("power_analysis.json")
    assert power["required_n_per_arm_80pct"] > 0
    sizes = [row["n_accounts_per_arm"] for row in power["results"]]
    values = [row["power"] for row in power["results"]]
    assert sizes == sorted(sizes)
    assert values == sorted(values)
    assert max(values) > 0.80


def test_explainability_uses_the_saved_model_contract():
    explain = _load("explainability.json")
    assert explain["n_features"] == 200
    assert explain["sample_rows"] > 0
    assert "lgbm_model.joblib" in explain["explainer"]
    assert len(explain["top_features"]) > 0
    assert len(explain["top_families"]) > 0
