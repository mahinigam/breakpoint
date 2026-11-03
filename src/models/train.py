"""Train the LightGBM champion and logistic-regression baseline.

Outputs
-------
``data/processed/oof_predictions.parquet``
    Out-of-fold raw probabilities for both models.
``artifacts/lgbm_model.joblib``
    Final LightGBM model trained on all rows.
``artifacts/lr_model.joblib``
    Final logistic-regression baseline.
``artifacts/feature_list.json``
    Exact feature contract used at inference.
``artifacts/categorical_levels.json``
    Category levels for ``D_63`` / ``D_64``.
``artifacts/model_metadata.json``
    Parameters, fold results and environment metadata.
``reports/metrics.json``
    The ``training`` section consumed by the docs renderer.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config import (
    ARTIFACTS_DIR,
    BOOTSTRAP_AUC_REPS,
    CV_FOLDS,
    EARLY_STOPPING_ROUNDS,
    NEGATIVE_WEIGHT,
    PATHS,
    REPORTS_DIR,
    SEED,
    ensure_directories,
)
from src.evaluation.delong import bootstrap_auc_difference, delong_roc_test
from src.evaluation.metrics import evaluate_model
from src.models.lgbm import get_lgbm_model
from src.models.preprocessing import (
    apply_categoricals,
    build_lr_pipeline,
    category_levels,
    feature_columns,
    load_training_frame,
)
from src.reporting import environment_metadata, write_json


def _sample_frame(frame: pd.DataFrame, sample_frac: float) -> pd.DataFrame:
    if sample_frac >= 1.0:
        return frame
    if not 0.0 < sample_frac <= 1.0:
        raise ValueError("sample_frac must be in (0, 1]")
    sampled, _ = train_test_split(
        frame,
        train_size=sample_frac,
        stratify=frame["target"],
        random_state=SEED,
    )
    return sampled.reset_index(drop=True)


def _metrics(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in evaluate_model(
            y, probabilities, negative_weight=NEGATIVE_WEIGHT
        ).items()
    }


def train(
    sample_frac: float = 1.0,
    folds: int = CV_FOLDS,
    bootstrap_reps: int = BOOTSTRAP_AUC_REPS,
) -> dict[str, Any]:
    ensure_directories()
    frame = _sample_frame(load_training_frame(), sample_frac)
    all_features, numeric, categorical = feature_columns(frame)
    levels = category_levels(frame, categorical)

    y = frame["target"].to_numpy(dtype=int)
    weights = np.where(y == 0, float(NEGATIVE_WEIGHT), 1.0)

    # Two views: strings for the LR one-hot pipeline, categories for LightGBM.
    x_lr = frame[all_features].copy()
    x_lgb = apply_categoricals(frame[all_features], categorical, levels)

    numeric_values = x_lr[numeric].to_numpy(dtype=float)
    if np.isinf(numeric_values).any():
        raise ValueError("Numeric features contain infinite values")

    print(
        f"Training on {len(frame):,} customers, {len(all_features)} features "
        f"({len(numeric)} numeric, {len(categorical)} categorical)"
    )

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    lr_oof = np.zeros(len(frame), dtype=float)
    lgb_oof = np.zeros(len(frame), dtype=float)
    fold_rows: list[dict[str, Any]] = []
    best_iterations: list[int] = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(x_lgb, y), start=1):
        print(f"Fold {fold}/{folds}")
        y_train, y_valid = y[train_idx], y[valid_idx]
        w_train, w_valid = weights[train_idx], weights[valid_idx]

        lr = build_lr_pipeline(numeric, categorical, random_state=SEED)
        lr.fit(
            x_lr.iloc[train_idx],
            y_train,
            lr__sample_weight=w_train,
        )
        lr_oof[valid_idx] = lr.predict_proba(x_lr.iloc[valid_idx])[:, 1]

        model = get_lgbm_model()
        model.fit(
            x_lgb.iloc[train_idx],
            y_train,
            sample_weight=w_train,
            eval_X=x_lgb.iloc[valid_idx],
            eval_y=y_valid,
            eval_sample_weight=[w_valid],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        lgb_oof[valid_idx] = model.predict_proba(x_lgb.iloc[valid_idx])[:, 1]
        best_iteration = int(model.best_iteration_ or model.n_estimators)
        best_iterations.append(best_iteration)

        lr_fold = _metrics(y_valid, lr_oof[valid_idx])
        lgb_fold = _metrics(y_valid, lgb_oof[valid_idx])
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": int(len(train_idx)),
                "valid_rows": int(len(valid_idx)),
                "best_iteration": best_iteration,
                "lr_auc": lr_fold["AUC"],
                "lr_ks": lr_fold["KS"],
                "lr_amex": lr_fold["Amex Metric"],
                "lgb_auc": lgb_fold["AUC"],
                "lgb_ks": lgb_fold["KS"],
                "lgb_amex": lgb_fold["Amex Metric"],
            }
        )

    print("Evaluating out-of-fold predictions")
    lr_metrics = _metrics(y, lr_oof)
    lgb_metrics = _metrics(y, lgb_oof)

    print("Corrected DeLong test (unweighted sample)")
    delong_p = float(delong_roc_test(y, lgb_oof, lr_oof))
    print(f"  p = {delong_p:.4e}")

    bootstrap_result = None
    if bootstrap_reps > 0:
        print(f"Weighted paired bootstrap ({bootstrap_reps} replicates)")
        bootstrap_result = bootstrap_auc_difference(
            y,
            lr_oof,
            lgb_oof,
            sample_weight=weights,
            n_boot=bootstrap_reps,
            seed=SEED,
        )
        print(
            "  AUC difference 95% CI: "
            f"[{bootstrap_result.ci_low:+.4f}, {bootstrap_result.ci_high:+.4f}], "
            f"p = {bootstrap_result.p_value:.4f}"
        )

    final_iterations = max(50, int(round(float(np.mean(best_iterations)))))
    print(f"Fitting final LightGBM with {final_iterations} trees on all rows")
    final_model = get_lgbm_model({"n_estimators": final_iterations})
    final_model.fit(x_lgb, y, sample_weight=weights)

    final_lr = build_lr_pipeline(numeric, categorical, random_state=SEED)
    final_lr.fit(x_lr, y, lr__sample_weight=weights)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, ARTIFACTS_DIR / "lgbm_model.joblib")
    joblib.dump(final_lr, ARTIFACTS_DIR / "lr_model.joblib")
    write_json(
        ARTIFACTS_DIR / "feature_list.json",
        {
            "all_features": all_features,
            "numeric_features": numeric,
            "categorical_features": categorical,
        },
    )
    write_json(ARTIFACTS_DIR / "categorical_levels.json", levels)

    oof = pd.DataFrame(
        {
            "customer_ID": frame["customer_ID"].to_numpy(),
            "target": y,
            "lr_prob": lr_oof,
            "lgb_prob": lgb_oof,
        }
    )
    oof.to_parquet(PATHS["oof_predictions"], index=False)

    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(REPORTS_DIR / "training_folds.csv", index=False)

    metadata = {
        "dataset_rows": int(len(frame)),
        "dataset_sample_frac": float(sample_frac),
        "unweighted_default_rate": float(y.mean()),
        "n_features": int(len(all_features)),
        "n_numeric_features": int(len(numeric)),
        "n_categorical_features": int(len(categorical)),
        "categorical_features": categorical,
        "folds": int(folds),
        "negative_weight": float(NEGATIVE_WEIGHT),
        "early_stopping_rounds": int(EARLY_STOPPING_ROUNDS),
        "best_iterations": best_iterations,
        "final_n_estimators": int(final_iterations),
        "lgbm_params": {
            key: value
            for key, value in final_model.get_params().items()
            if key
            in {
                "n_estimators",
                "learning_rate",
                "max_depth",
                "num_leaves",
                "subsample",
                "colsample_bytree",
                "min_child_samples",
                "reg_alpha",
                "reg_lambda",
            }
        },
        "oof_metrics": {"logistic_regression": lr_metrics, "lightgbm": lgb_metrics},
        "fold_metrics": fold_rows,
        "delong_p_value_unweighted": delong_p,
        "bootstrap_auc_difference": (
            bootstrap_result.to_dict() if bootstrap_result is not None else None
        ),
        "environment": environment_metadata(),
    }
    write_json(ARTIFACTS_DIR / "model_metadata.json", metadata)
    write_json(
        REPORTS_DIR / "metrics.json",
        {
            "training": {
                **metadata,
                "posterior_default_rate": float(
                    np.average(y, weights=weights)
                ),
            }
        },
    )
    print(f"Saved artifacts to {ARTIFACTS_DIR}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-frac", type=float, default=1.0)
    parser.add_argument("--folds", type=int, default=CV_FOLDS)
    parser.add_argument(
        "--bootstrap-reps", type=int, default=BOOTSTRAP_AUC_REPS
    )
    args = parser.parse_args()
    train(
        sample_frac=args.sample_frac,
        folds=args.folds,
        bootstrap_reps=args.bootstrap_reps,
    )


if __name__ == "__main__":
    main()
