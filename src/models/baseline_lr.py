"""Logistic-regression baseline pipeline."""

from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import SEED, _settings


def get_lr_pipeline(random_state: int = SEED) -> Pipeline:
    """Numeric-only fallback pipeline.

    Training uses :func:`src.models.preprocessing.build_lr_pipeline`, which
    also handles the categorical columns. This helper is kept for small
    numeric-only experiments and tests.
    """
    lr_config = _settings["model"]["lr"]
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=float(lr_config["C"]),
                    max_iter=int(lr_config["max_iter"]),
                    random_state=random_state,
                ),
            ),
        ]
    )
