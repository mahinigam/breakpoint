"""Load the persisted champion model and its inference contract.

The bundle is deliberately strict: feature order, category levels, and policy
thresholds are all loaded from artifacts rather than inferred at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import (
    ACTION_ECONOMICS,
    ARTIFACTS_DIR,
    ECONOMIC_SCENARIOS,
    REPORTS_DIR,
)
from src.models.preprocessing import apply_categoricals
from src.policy.actions import build_action_economics
from src.reporting import load_json


class ScoringContractError(ValueError):
    """Raised when an input frame does not satisfy the model contract."""


def _calibrate(raw: np.ndarray, calibrator: dict[str, Any]) -> np.ndarray:
    method = calibrator.get("method", "isotonic")
    if method == "isotonic":
        calibrated = calibrator["isotonic"].predict(raw)
    else:
        calibrated = calibrator["platt"].predict_proba(raw.reshape(-1, 1))[:, 1]
    return np.clip(np.asarray(calibrated, dtype=float), 0.0, 1.0)


@dataclass(frozen=True)
class ArtifactBundle:
    model: Any
    calibrator: dict[str, Any]
    features: list[str]
    categorical: list[str]
    levels: dict[str, list[str]]
    policy: dict[str, Any]
    scenario_name: str = "moderate"

    @classmethod
    def load(
        cls,
        artifacts_dir: Path = ARTIFACTS_DIR,
        policy_path: Path | None = None,
        scenario_name: str = "moderate",
    ) -> "ArtifactBundle":
        required = {
            "model": artifacts_dir / "lgbm_model.joblib",
            "calibrator": artifacts_dir / "calibrator.joblib",
            "feature_list": artifacts_dir / "feature_list.json",
            "categorical_levels": artifacts_dir / "categorical_levels.json",
        }
        missing = [str(path) for path in required.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing scoring artifacts: "
                + ", ".join(missing)
                + ". Run `make train` and `make calibrate` first."
            )
        policy_path = policy_path or (REPORTS_DIR / "policy.json")
        if not policy_path.exists():
            raise FileNotFoundError(
                f"Missing policy file {policy_path}. Run `make decide` first."
            )

        feature_contract = load_json(required["feature_list"])
        features = list(feature_contract["all_features"])
        categorical = list(feature_contract["categorical_features"])
        levels = load_json(required["categorical_levels"])
        model = joblib.load(required["model"])
        calibrator = joblib.load(required["calibrator"])
        policy = load_json(policy_path)

        expected_features = getattr(model, "n_features_in_", len(features))
        if int(expected_features) != len(features):
            raise ScoringContractError(
                f"Model expects {expected_features} features but the contract lists "
                f"{len(features)}."
            )
        missing_levels = set(categorical) - set(levels)
        if missing_levels:
            raise ScoringContractError(
                f"Missing category levels for: {sorted(missing_levels)}"
            )
        if scenario_name not in ECONOMIC_SCENARIOS:
            raise ScoringContractError(
                f"Unknown scenario {scenario_name!r}; choose from "
                f"{sorted(ECONOMIC_SCENARIOS)}"
            )
        return cls(
            model=model,
            calibrator=calibrator,
            features=features,
            categorical=categorical,
            levels=levels,
            policy=policy,
            scenario_name=scenario_name,
        )

    @property
    def grow_cutoff(self) -> float:
        return float(self.policy["action_policy"]["grow_cutoff"])

    @property
    def restrict_cutoff(self) -> float:
        return float(self.policy["action_policy"]["restrict_cutoff"])

    def validate(self, frame: pd.DataFrame) -> None:
        missing = [feature for feature in self.features if feature not in frame.columns]
        if missing:
            raise ScoringContractError(
                "Input is missing model features: " + ", ".join(sorted(missing))
            )

    def score_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Score a customer-level feature frame and assign policy actions."""
        self.validate(frame)
        x_frame = apply_categoricals(
            frame[self.features], self.categorical, self.levels
        )
        raw = np.asarray(self.model.predict_proba(x_frame)[:, 1], dtype=float)
        calibrated = _calibrate(raw, self.calibrator)

        grow = calibrated <= self.grow_cutoff
        restrict = calibrated > self.restrict_cutoff
        hold = ~grow & ~restrict
        action = np.where(grow, "Grow", np.where(hold, "Hold", "Restrict"))

        scenario = ECONOMIC_SCENARIOS[self.scenario_name]
        economics = build_action_economics(scenario, ACTION_ECONOMICS)
        margins = np.select(
            [grow, hold, restrict],
            [
                economics["grow"].margin,
                economics["hold"].margin,
                economics["restrict"].margin,
            ],
            default=economics["hold"].margin,
        )
        losses = np.select(
            [grow, hold, restrict],
            [
                economics["grow"].loss,
                economics["hold"].loss,
                economics["restrict"].loss,
            ],
            default=economics["hold"].loss,
        )

        output = pd.DataFrame(
            {
                "customer_ID": (
                    frame["customer_ID"].to_numpy()
                    if "customer_ID" in frame.columns
                    else np.arange(len(frame))
                ),
                "probability_of_default": calibrated,
                "raw_model_score": raw,
                "action": action,
                "policy_band": action,
                "grow_cutoff": self.grow_cutoff,
                "restrict_cutoff": self.restrict_cutoff,
                "scenario": self.scenario_name,
                "expected_margin": (1.0 - calibrated) * margins,
                "expected_loss": calibrated * losses,
                "expected_profit": (1.0 - calibrated) * margins
                - calibrated * losses,
                "exposure_multiplier": np.select(
                    [grow, hold, restrict],
                    [
                        ACTION_ECONOMICS["grow"]["exposure_multiplier"],
                        ACTION_ECONOMICS["hold"]["exposure_multiplier"],
                        ACTION_ECONOMICS["restrict"]["exposure_multiplier"],
                    ],
                    default=1.0,
                ),
            }
        )
        output["expected_profit"] = (
            output["expected_profit"].astype(float).round(6)
        )
        output["probability_of_default"] = output["probability_of_default"].round(8)
        output["raw_model_score"] = output["raw_model_score"].round(8)
        return output
