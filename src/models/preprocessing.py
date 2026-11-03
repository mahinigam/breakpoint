"""Shared feature preparation for training, scoring, and explainability.

The model consumes one row per customer. Two columns are categorical
(``D_63`` and ``D_64``); everything else is numeric. The category levels are
saved alongside the model so inference cannot silently re-code them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import PATHS

ID_COLUMNS = ("customer_ID", "S_2", "target")
CATEGORICAL_FEATURES = ("D_63", "D_64")


def load_training_frame() -> pd.DataFrame:
    features = pd.read_parquet(PATHS["processed_features"])
    labels = pd.read_parquet(PATHS["parquet_train_labels"])
    if "target" in features.columns:
        features = features.drop(columns=["target"])
    frame = features.merge(labels, on="customer_ID", how="inner", validate="one_to_one")
    if len(frame) != len(features):
        raise ValueError(
            f"Feature/label merge changed row count: {len(features)} -> {len(frame)}"
        )
    return frame


def feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Return (all_features, numeric_features, categorical_features)."""
    all_features = [
        column for column in frame.columns if column not in ID_COLUMNS
    ]
    categorical = [column for column in CATEGORICAL_FEATURES if column in all_features]
    numeric = [
        column
        for column in all_features
        if column not in categorical and pd.api.types.is_numeric_dtype(frame[column])
    ]
    unsupported = set(all_features) - set(numeric) - set(categorical)
    if unsupported:
        raise ValueError(
            "Unsupported feature dtypes for columns: "
            + ", ".join(sorted(unsupported))
        )
    return all_features, numeric, categorical


def category_levels(
    frame: pd.DataFrame, categorical: list[str]
) -> dict[str, list[str]]:
    levels: dict[str, list[str]] = {}
    for column in categorical:
        values = frame[column].dropna().astype(str).unique().tolist()
        levels[column] = sorted(values)
    return levels


def apply_categoricals(
    frame: pd.DataFrame,
    categorical: list[str],
    levels: dict[str, list[str]],
) -> pd.DataFrame:
    """Cast categorical columns to the exact training categories."""
    frame = frame.copy()
    for column in categorical:
        expected = levels.get(column, [])
        if not expected:
            continue
        frame[column] = pd.Categorical(
            frame[column].astype("string"), categories=expected
        )
    return frame


def build_lr_pipeline(
    numeric: list[str], categorical: list[str], random_state: int
) -> Pipeline:
    """Logistic-regression baseline with imputation, scaling and one-hot encoding."""
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    transformers: list[tuple[str, Any, list[str]]] = [
        ("numeric", numeric_pipeline, numeric)
    ]
    if categorical:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical))

    return Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers, remainder="drop")),
            (
                "lr",
                LogisticRegression(
                    C=0.1,
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )
