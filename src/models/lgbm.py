"""LightGBM configuration.

Hyperparameters are explicit and versioned in ``config/settings.yaml``. The
repository does not claim an Optuna search: the model card documents exactly
what was run.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb

from src.config import SEED, _settings

DEFAULT_PARAMS: dict[str, Any] = {
    **_settings["model"]["lgbm"],
    "random_state": SEED,
    "n_jobs": int(_settings["model"]["lgbm"].get("n_jobs", -1)),
    "verbose": int(_settings["model"]["lgbm"].get("verbose", -1)),
}


def get_lgbm_model(params: dict[str, Any] | None = None) -> lgb.LGBMClassifier:
    """Return an unfitted LightGBM classifier with the pinned parameters."""
    merged = {**DEFAULT_PARAMS, **(params or {})}
    return lgb.LGBMClassifier(**merged)
