"""Cross-fitted probability calibration.

The calibration map is evaluated out of fold: for each fold, the calibrator is
fit on the other folds and applied to the held-out fold. The final calibrator
(saved for scoring) is then fit on all OOF rows. This avoids the common but
misleading practice of fitting and evaluating a calibrator on the same rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import (
    ARTIFACTS_DIR,
    CALIBRATION_BINS,
    CALIBRATION_FOLDS,
    CALIBRATION_METHOD,
    FIGURES_DIR,
    NEGATIVE_WEIGHT,
    PATHS,
    REPORTS_DIR,
    SEED,
    ensure_directories,
)
from src.reporting import environment_metadata, update_json, write_json


def _weights(y_true: np.ndarray) -> np.ndarray:
    return np.where(y_true == 0, float(NEGATIVE_WEIGHT), 1.0)


def calibration_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    weights: np.ndarray,
    n_bins: int = CALIBRATION_BINS,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """Return weighted calibration bins with non-empty edges only."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if strategy == "quantile":
        edges = np.quantile(y_prob, np.linspace(0.0, 1.0, n_bins + 1))
        edges = np.unique(edges)
    elif strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'")

    if len(edges) < 2:
        edges = np.array([0.0, 1.0])
    bin_ids = np.digitize(y_prob, edges[1:-1], right=True)

    rows: list[dict[str, float]] = []
    for bin_id in range(len(edges) - 1):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        bin_weight = float(weights[mask].sum())
        rows.append(
            {
                "bin": bin_id,
                "lower": float(edges[bin_id]),
                "upper": float(edges[bin_id + 1]),
                "count": int(mask.sum()),
                "weight": bin_weight,
                "mean_predicted": float(
                    np.average(y_prob[mask], weights=weights[mask])
                ),
                "observed": float(
                    np.average(y_true[mask], weights=weights[mask])
                ),
                "weight_share": float(bin_weight / weights.sum()),
            }
        )
    return pd.DataFrame(rows)


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    weights: np.ndarray,
    n_bins: int = CALIBRATION_BINS,
    strategy: str = "quantile",
) -> float:
    bins = calibration_bins(
        y_true, y_prob, weights, n_bins=n_bins, strategy=strategy
    )
    if bins.empty:
        return 0.0
    return float(
        (bins["weight_share"] * (bins["mean_predicted"] - bins["observed"]).abs()).sum()
    )


def _metrics(
    y_true: np.ndarray, y_prob: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(y_true, y_prob, sample_weight=weights)),
        "brier": float(brier_score_loss(y_true, y_prob, sample_weight=weights)),
        "log_loss": float(log_loss(y_true, y_prob, sample_weight=weights)),
        "ece_quantile": compute_ece(
            y_true, y_prob, weights, strategy="quantile"
        ),
        "ece_uniform": compute_ece(
            y_true, y_prob, weights, strategy="uniform"
        ),
    }


def _fit_isotonic(
    raw: np.ndarray, y_true: np.ndarray, weights: np.ndarray
) -> IsotonicRegression:
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    model.fit(raw, y_true, sample_weight=weights)
    return model


def _fit_platt(
    raw: np.ndarray, y_true: np.ndarray, weights: np.ndarray
) -> LogisticRegression:
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(raw.reshape(-1, 1), y_true, sample_weight=weights)
    return model


def _predict(model: Any, raw: np.ndarray, method: str) -> np.ndarray:
    if method == "isotonic":
        return np.asarray(model.predict(raw), dtype=float)
    return np.asarray(model.predict_proba(raw.reshape(-1, 1))[:, 1], dtype=float)


def plot_reliability(
    y_true: np.ndarray,
    methods: dict[str, np.ndarray],
    weights: np.ndarray,
    output_dir: Path,
    filename: str,
    n_bins: int = CALIBRATION_BINS,
    log_x: bool = False,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", label="Perfect calibration")
    for color, (name, probabilities) in zip(colors, methods.items()):
        bins = calibration_bins(
            y_true, probabilities, weights, n_bins=n_bins, strategy="quantile"
        )
        if bins.empty:
            continue
        ece = compute_ece(y_true, probabilities, weights, strategy="quantile")
        ax.plot(
            bins["mean_predicted"],
            bins["observed"],
            marker="o",
            linewidth=1.8,
            color=color,
            label=f"{name} (ECE {ece:.3%})",
        )
    if log_x:
        ax.set_xscale("log")
        ax.set_xlim(1e-5, 1.0)
    else:
        ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.25, float(max(bins["observed"].max(), 0.05)) * 1.2))
    ax.set_xlabel("Mean predicted probability of default")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Reliability diagram (population-weighted, quantile bins)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def calibrate() -> dict[str, Any]:
    ensure_directories()
    oof_path = PATHS["oof_predictions"]
    if not oof_path.exists():
        raise FileNotFoundError(
            f"{oof_path} not found. Run `python -m src.models.train` first."
        )
    frame = pd.read_parquet(oof_path)
    y_true = frame["target"].to_numpy(dtype=int)
    raw = frame["lgb_prob"].to_numpy(dtype=float)
    weights = _weights(y_true)

    splitter = StratifiedKFold(
        n_splits=CALIBRATION_FOLDS, shuffle=True, random_state=SEED
    )
    platt_oof = np.zeros(len(frame), dtype=float)
    isotonic_oof = np.zeros(len(frame), dtype=float)

    for train_idx, valid_idx in splitter.split(raw.reshape(-1, 1), y_true):
        iso = _fit_isotonic(raw[train_idx], y_true[train_idx], weights[train_idx])
        isotonic_oof[valid_idx] = _predict(iso, raw[valid_idx], "isotonic")
        platt = _fit_platt(raw[train_idx], y_true[train_idx], weights[train_idx])
        platt_oof[valid_idx] = _predict(platt, raw[valid_idx], "platt")

    metrics = {
        "raw": _metrics(y_true, raw, weights),
        "platt": _metrics(y_true, platt_oof, weights),
        "isotonic": _metrics(y_true, isotonic_oof, weights),
    }

    chosen = CALIBRATION_METHOD
    if chosen not in {"isotonic", "platt"}:
        raise ValueError("calibration_method must be 'isotonic' or 'platt'")
    calibrated = isotonic_oof if chosen == "isotonic" else platt_oof

    # Final calibrators for scoring. Isotonic is the production default.
    final_isotonic = _fit_isotonic(raw, y_true, weights)
    final_platt = _fit_platt(raw, y_true, weights)
    joblib.dump(
        {
            "method": chosen,
            "isotonic": final_isotonic,
            "platt": final_platt,
        },
        ARTIFACTS_DIR / "calibrator.joblib",
    )

    output = frame.copy()
    output["platt_prob"] = platt_oof
    output["isotonic_prob"] = isotonic_oof
    output["lgb_prob_calibrated"] = calibrated
    output.to_parquet(PATHS["calibrated_predictions"], index=False)

    bins_frames = []
    for name, probabilities in {
        "raw": raw,
        "platt": platt_oof,
        "isotonic": isotonic_oof,
    }.items():
        bins = calibration_bins(y_true, probabilities, weights)
        bins["method"] = name
        bins_frames.append(bins)
    pd.concat(bins_frames, ignore_index=True).to_csv(
        REPORTS_DIR / "calibration_bins.csv", index=False
    )

    plot_reliability(
        y_true,
        {
            "Raw LightGBM": raw,
            "Platt": platt_oof,
            "Isotonic": isotonic_oof,
        },
        weights,
        FIGURES_DIR,
        "calibration_weighted.png",
    )
    plot_reliability(
        y_true,
        {"Raw LightGBM": raw, "Isotonic": isotonic_oof},
        weights,
        FIGURES_DIR,
        "calibration_low_pd.png",
        log_x=True,
    )

    metadata = {
        "method": chosen,
        "folds": CALIBRATION_FOLDS,
        "bins": CALIBRATION_BINS,
        "metrics": metrics,
        "selected_metrics": metrics[chosen],
        "environment": environment_metadata(),
    }
    write_json(ARTIFACTS_DIR / "calibration_metadata.json", metadata)
    update_json(REPORTS_DIR / "metrics.json", "calibration", metadata)
    print(f"Selected calibrator: {chosen}")
    for name, values in metrics.items():
        print(
            f"  {name:9s} AUC {values['auc']:.4f}  Brier {values['brier']:.6f}  "
            f"ECE(q) {values['ece_quantile']:.4%}"
        )
    print(f"Saved calibrated OOF to {PATHS['calibrated_predictions']}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    calibrate()


if __name__ == "__main__":
    main()
