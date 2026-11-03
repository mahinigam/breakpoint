from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import ARTIFACTS_DIR, REPORTS_DIR
from src.scoring.artifacts import ArtifactBundle, ScoringContractError
from src.scoring.score import read_input


pytestmark = pytest.mark.skipif(
    not (ARTIFACTS_DIR / "lgbm_model.joblib").exists()
    or not (REPORTS_DIR / "policy.json").exists(),
    reason="scoring artifacts are not present",
)


@pytest.fixture(scope="module")
def bundle():
    return ArtifactBundle.load()


@pytest.fixture(scope="module")
def demo_frame():
    path = Path("data/demo/scoring_input.csv")
    if not path.exists():
        pytest.skip("demo scoring input is not present")
    return pd.read_csv(path)


def test_bundle_contract_is_internally_consistent(bundle):
    assert len(bundle.features) == 200
    assert len(bundle.categorical) == 2
    assert set(bundle.categorical).issubset(bundle.features)
    assert bundle.grow_cutoff < bundle.restrict_cutoff
    assert 0.0 < bundle.grow_cutoff < 1.0
    assert 0.0 < bundle.restrict_cutoff < 1.0


def test_score_frame_returns_actions_consistent_with_cutoffs(bundle, demo_frame):
    scored = bundle.score_frame(demo_frame.head(25))
    assert len(scored) == 25
    assert set(scored["action"]).issubset({"Grow", "Hold", "Restrict"})
    assert scored["probability_of_default"].between(0.0, 1.0).all()

    grow = scored["probability_of_default"] <= bundle.grow_cutoff
    restrict = scored["probability_of_default"] > bundle.restrict_cutoff
    assert (scored.loc[grow, "action"] == "Grow").all()
    assert (scored.loc[restrict, "action"] == "Restrict").all()
    hold = ~grow & ~restrict
    assert (scored.loc[hold, "action"] == "Hold").all()


def test_expected_profit_is_finite_and_uses_action(bundle, demo_frame):
    scored = bundle.score_frame(demo_frame.head(50))
    assert np.isfinite(scored["expected_profit"]).all()
    assert (scored["expected_margin"] >= 0).all()
    assert (scored["expected_loss"] >= 0).all()
    by_action = scored.groupby("action")["expected_profit"].mean()
    assert by_action.notna().all()


def test_missing_feature_raises_clear_error(bundle, demo_frame):
    with pytest.raises(ScoringContractError, match="missing model features"):
        bundle.score_frame(demo_frame.drop(columns=[bundle.features[0]]).head(2))


def test_unknown_scenario_is_rejected():
    with pytest.raises(ScoringContractError, match="Unknown scenario"):
        ArtifactBundle.load(scenario_name="not-a-scenario")


def test_read_input_supports_csv_and_parquet(tmp_path, demo_frame):
    csv_path = tmp_path / "rows.csv"
    demo_frame.head(3).to_csv(csv_path, index=False)
    csv = read_input(csv_path)
    assert len(csv) == 3

    parquet_path = tmp_path / "rows.parquet"
    demo_frame.head(3).to_parquet(parquet_path, index=False)
    parquet = read_input(parquet_path)
    assert len(parquet) == 3

    with pytest.raises(ValueError, match=".parquet or .csv"):
        read_input(tmp_path / "rows.json")
