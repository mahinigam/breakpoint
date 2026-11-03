"""Central configuration for the Breakpoint pipeline.

All magic numbers (seed, sampling weight, economic assumptions) live in
``config/*.yaml`` so that the same values are used by training, evaluation,
decisioning, monitoring and the Tableau export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return loaded


_settings = _load_yaml(ROOT_DIR / "config" / "settings.yaml")
_economics = _load_yaml(ROOT_DIR / "config" / "economic_scenarios.yaml")

SEED: int = int(_settings["seed"])
NEGATIVE_WEIGHT: int = int(_settings["negative_weight"])
TRAIN_VAL_SPLIT_RATIO: float = float(_settings["train_val_split_ratio"])
CV_FOLDS: int = int(_settings["cv_folds"])
EARLY_STOPPING_ROUNDS: int = int(_settings.get("early_stopping_rounds", 75))
BOOTSTRAP_AUC_REPS: int = int(_settings.get("bootstrap_auc_reps", 300))
CALIBRATION_METHOD: str = str(_settings.get("calibration_method", "isotonic"))
CALIBRATION_FOLDS: int = int(_settings.get("calibration_folds", 5))
CALIBRATION_BINS: int = int(_settings.get("calibration_bins", 12))
MODEL_CONFIG: dict[str, Any] = _settings.get("model", {})

ECONOMIC_SCENARIOS: dict[str, dict[str, Any]] = _economics["scenarios"]
ACTION_ECONOMICS: dict[str, dict[str, Any]] = _economics["actions"]
POLICY_CONSTRAINTS: dict[str, float] = _economics["constraints"]
THRESHOLD_GRID_CONFIG: dict[str, Any] = _economics["threshold_grid"]
COST_RATIO_SWEEP: dict[str, float] = _economics["cost_ratio_sweep"]

PATHS: dict[str, Path] = {
    key: ROOT_DIR / value for key, value in _settings["paths"].items()
}

ARTIFACTS_DIR = PATHS["artifacts"]
REPORTS_DIR = PATHS["reports"]
FIGURES_DIR = PATHS["figures"]
DASHBOARD_DATA_DIR = PATHS["dashboard_data"]


def ensure_directories() -> None:
    """Create every output directory used by the pipeline."""
    for path in (
        ARTIFACTS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        DASHBOARD_DATA_DIR,
        PATHS["processed_features"].parent,
        PATHS["parquet_train_data"].parent,
        PATHS["demo"],
        ROOT_DIR / "data" / "raw",
    ):
        path.mkdir(parents=True, exist_ok=True)


def scenario_loss(scenario: dict[str, Any]) -> float:
    """Expected loss per default for a scenario."""
    return float(scenario["lgd"]) * float(scenario["exposure"])


def scenario_cost_ratio(scenario: dict[str, Any]) -> float:
    """Cost ratio r = expected loss / margin."""
    return scenario_loss(scenario) / float(scenario["margin"])
