"""Customer-level feature and score drift monitoring.

The old implementation compared raw statement-level columns from a 1% sample
of the *test* statements. That is not the population the model scores, and it
did not use the model's feature contract. This module monitors the exact 200
customer-level features consumed by the champion model, plus the calibrated
score and the resulting action mix.

The default split is temporal: the earliest 80% of accounts by statement date
form the reference window, and the latest 20% form the current window. A random
split is available as a stability check. Neither split is a substitute for a
true out-of-time labelled validation set; the report says so.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    ARTIFACTS_DIR,
    DASHBOARD_DATA_DIR,
    FIGURES_DIR,
    NEGATIVE_WEIGHT,
    PATHS,
    REPORTS_DIR,
    SEED,
    ensure_directories,
)
from src.monitoring.drift_flags import evaluate_drift_triggers
from src.reporting import environment_metadata, load_json, write_json

PSI_YELLOW = 0.10
PSI_RED = 0.25
MIN_NON_MISSING = 100


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
    epsilon: float = 1e-6,
    expected_weight: np.ndarray | None = None,
    actual_weight: np.ndarray | None = None,
) -> float:
    """Population Stability Index for numeric arrays.

    Bins are quantiles of the reference distribution. Missing values must be
    handled by the caller; they are reported separately. Weights are optional
    and are used for the population-weighted score distribution.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected_mask = np.isfinite(expected)
    actual_mask = np.isfinite(actual)
    expected = expected[expected_mask]
    actual = actual[actual_mask]
    if expected_weight is not None:
        expected_weight = np.asarray(expected_weight, dtype=float)[expected_mask]
    if actual_weight is not None:
        actual_weight = np.asarray(actual_weight, dtype=float)[actual_mask]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    edges = np.unique(np.quantile(expected, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf

    expected_share = np.histogram(
        expected, bins=edges, weights=expected_weight
    )[0].astype(float)
    actual_share = np.histogram(
        actual, bins=edges, weights=actual_weight
    )[0].astype(float)
    expected_share /= expected_share.sum()
    actual_share /= actual_share.sum()
    expected_share = np.clip(expected_share, epsilon, None)
    actual_share = np.clip(actual_share, epsilon, None)
    return float(
        np.sum((actual_share - expected_share) * np.log(actual_share / expected_share))
    )


def categorical_psi(
    expected: pd.Series,
    actual: pd.Series,
    epsilon: float = 1e-6,
    expected_weight: np.ndarray | None = None,
    actual_weight: np.ndarray | None = None,
) -> float:
    """PSI for a categorical feature, treating missing as a level."""
    expected = expected.astype("string").fillna("<missing>")
    actual = actual.astype("string").fillna("<missing>")
    levels = sorted(set(expected.unique()) | set(actual.unique()))
    if not levels:
        return 0.0

    def _shares(values: pd.Series, weights: np.ndarray | None) -> np.ndarray:
        categories = pd.Categorical(values, categories=levels)
        if weights is None:
            weights = np.ones(len(values), dtype=float)
        weighted = np.bincount(
            categories.codes, weights=np.asarray(weights, dtype=float), minlength=len(levels)
        )
        total = weighted.sum()
        return weighted / total if total else np.zeros(len(levels), dtype=float)

    expected_values = np.clip(_shares(expected, expected_weight), epsilon, None)
    actual_values = np.clip(_shares(actual, actual_weight), epsilon, None)
    return float(
        np.sum((actual_values - expected_values) * np.log(actual_values / expected_values))
    )


def family_for(feature: str) -> str:
    for prefix, family in (
        ("D_", "Delinquency"),
        ("S_", "Spend"),
        ("P_", "Payment"),
        ("B_", "Balance"),
        ("R_", "Risk"),
    ):
        if feature.startswith(prefix):
            return family
    return "Engineered/Other"


def status_for(psi: float, missing_delta: float, enough_data: bool) -> str:
    if not enough_data:
        return "insufficient_data"
    if psi > PSI_RED or abs(missing_delta) > 0.10:
        return "red"
    if psi > PSI_YELLOW or abs(missing_delta) > 0.05:
        return "yellow"
    return "green"


def compute_feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    categorical: list[str],
    bins: int = 10,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in features:
        reference_values = reference[feature]
        current_values = current[feature]
        missing_reference = float(reference_values.isna().mean())
        missing_current = float(current_values.isna().mean())
        missing_delta = missing_current - missing_reference

        if feature in categorical:
            psi = categorical_psi(reference_values, current_values)
            reference_count = int(reference_values.notna().sum())
            current_count = int(current_values.notna().sum())
        else:
            reference_numeric = pd.to_numeric(reference_values, errors="coerce")
            current_numeric = pd.to_numeric(current_values, errors="coerce")
            reference_count = int(reference_numeric.notna().sum())
            current_count = int(current_numeric.notna().sum())
            psi = calculate_psi(
                reference_numeric.to_numpy(dtype=float),
                current_numeric.to_numpy(dtype=float),
                bins=bins,
            )

        enough_data = min(reference_count, current_count) >= MIN_NON_MISSING
        rows.append(
            {
                "feature": feature,
                "family": family_for(feature),
                "psi": psi,
                "status": status_for(psi, missing_delta, enough_data),
                "missing_reference": missing_reference,
                "missing_current": missing_current,
                "missing_delta": missing_delta,
                "reference_non_missing": reference_count,
                "current_non_missing": current_count,
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def action_mix(
    scores: np.ndarray,
    grow_cutoff: float,
    restrict_cutoff: float,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    scores = np.asarray(scores, dtype=float)
    if weights is None:
        weights = np.ones(len(scores), dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    grow = float(weights[scores <= grow_cutoff].sum() / total)
    restrict = float(weights[scores > restrict_cutoff].sum() / total)
    hold = float(max(0.0, 1.0 - grow - restrict))
    return {"grow": grow, "hold": hold, "restrict": restrict}


def _plot_psi(drift: pd.DataFrame, output_dir: Path) -> None:
    top = drift.head(20).iloc[::-1]
    colors = {"red": "#d62728", "yellow": "#ff9f1c", "green": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh(
        top["feature"],
        top["psi"],
        color=[colors.get(status, "#777777") for status in top["status"]],
    )
    ax.axvline(PSI_YELLOW, color="#ff9f1c", linestyle="--", linewidth=1.2, label="Yellow 0.10")
    ax.axvline(PSI_RED, color="#d62728", linestyle="--", linewidth=1.2, label="Red 0.25")
    ax.set_xlabel("Population Stability Index (reference vs current)")
    ax.set_title("Top 20 drifting model features")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, axis="x", linestyle="--", alpha=0.25)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "psi_top_features.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_score_drift(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    reference_weight: np.ndarray,
    current_weight: np.ndarray,
    output_dir: Path,
) -> None:
    positive = np.concatenate([reference_scores, current_scores])
    positive = positive[positive > 0]
    lower = max(float(positive.min()) if len(positive) else 1e-6, 1e-6)
    bins = np.logspace(np.log10(lower), 0.0, 50)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(
        reference_scores,
        bins=bins,
        weights=reference_weight,
        density=True,
        alpha=0.45,
        label="Reference window",
        color="#4c78a8",
    )
    ax.hist(
        current_scores,
        bins=bins,
        weights=current_weight,
        density=True,
        alpha=0.45,
        label="Current window",
        color="#f58518",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Calibrated probability of default (log scale)")
    ax.set_ylabel("Density")
    ax.set_title("Score distribution drift (out-of-fold calibrated scores)")
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "score_drift.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_drift(split: str = "temporal", bins: int = 10) -> dict[str, Any]:
    ensure_directories()
    feature_contract = load_json(ARTIFACTS_DIR / "feature_list.json")
    features = feature_contract["all_features"]
    categorical = feature_contract["categorical_features"]

    columns = ["customer_ID", "S_2", *features]
    frame = pd.read_parquet(PATHS["processed_features"], columns=columns)
    scores = pd.read_parquet(
        PATHS["calibrated_predictions"],
        columns=["customer_ID", "target", "lgb_prob_calibrated"],
    )
    frame = frame.merge(scores, on="customer_ID", how="inner", validate="one_to_one")
    frame["population_weight"] = np.where(
        frame["target"].to_numpy() == 0, float(NEGATIVE_WEIGHT), 1.0
    )

    if split == "temporal":
        cutoff = frame["S_2"].quantile(0.80)
        reference = frame[frame["S_2"] <= cutoff]
        current = frame[frame["S_2"] > cutoff]
        split_description = (
            "Earliest 80% of accounts by statement date vs latest 20% "
            f"(cutoff {cutoff})"
        )
    elif split == "random":
        reference, current = train_test_split(
            frame, test_size=0.20, random_state=SEED, stratify=frame["target"]
        )
        split_description = "Random 80/20 split stratified by target"
    else:
        raise ValueError("split must be 'temporal' or 'random'")

    if len(reference) == 0 or len(current) == 0:
        raise ValueError("Drift split produced an empty window")

    drift = compute_feature_drift(
        reference, current, features, categorical, bins=bins
    )
    reference_scores = reference["lgb_prob_calibrated"].to_numpy(dtype=float)
    current_scores = current["lgb_prob_calibrated"].to_numpy(dtype=float)
    reference_weight = reference["population_weight"].to_numpy(dtype=float)
    current_weight = current["population_weight"].to_numpy(dtype=float)
    score_psi = calculate_psi(
        reference_scores,
        current_scores,
        bins=10,
        expected_weight=reference_weight,
        actual_weight=current_weight,
    )
    policy = load_json(REPORTS_DIR / "policy.json")
    grow_cutoff = float(policy["action_policy"]["grow_cutoff"])
    restrict_cutoff = float(policy["action_policy"]["restrict_cutoff"])
    reference_mix = action_mix(
        reference_scores,
        grow_cutoff,
        restrict_cutoff,
        weights=reference_weight,
    )
    current_mix = action_mix(
        current_scores,
        grow_cutoff,
        restrict_cutoff,
        weights=current_weight,
    )
    mix_delta = {
        action: current_mix[action] - reference_mix[action]
        for action in reference_mix
    }

    trigger = evaluate_drift_triggers(
        drift,
        score_psi=score_psi,
        action_mix_delta=mix_delta,
        psi_threshold=PSI_RED,
    )

    drift.to_csv(REPORTS_DIR / "drift_features.csv", index=False)
    drift.to_csv(DASHBOARD_DATA_DIR / "drift.csv", index=False)
    _plot_psi(drift, FIGURES_DIR)
    _plot_score_drift(
        reference_scores,
        current_scores,
        reference_weight,
        current_weight,
        FIGURES_DIR,
    )

    reference_rate = float(
        np.average(reference["target"].to_numpy(), weights=reference_weight)
    )
    current_rate = float(
        np.average(current["target"].to_numpy(), weights=current_weight)
    )
    status_counts = drift["status"].value_counts().to_dict()
    summary = {
        "generated_at": environment_metadata()["generated_at"],
        "split": split,
        "split_description": split_description,
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "reference_default_rate": reference_rate,
        "current_default_rate": current_rate,
        "default_rate_delta": current_rate - reference_rate,
        "reference_default_rate_unweighted": float(reference["target"].mean()),
        "current_default_rate_unweighted": float(current["target"].mean()),
        "feature_psi_weighting": (
            "unweighted (observed account-level feature distributions); "
            "score PSI and action mix are population-weighted with the 20x "
            "negative-class design weight"
        ),
        "n_features": int(len(features)),
        "status_counts": {key: int(value) for key, value in status_counts.items()},
        "top_features": drift.head(10)[
            ["feature", "family", "psi", "status", "missing_delta"]
        ].to_dict(orient="records"),
        "score_psi": score_psi,
        "score_status": (
            "red" if score_psi > PSI_RED else "yellow" if score_psi > PSI_YELLOW else "green"
        ),
        "action_mix_reference": reference_mix,
        "action_mix_current": current_mix,
        "action_mix_delta": mix_delta,
        "retrain_trigger": trigger,
        "caveat": (
            "This is a monitoring exercise on observed data, not a labelled "
            "out-of-time validation set. The test labels are unavailable; the "
            "current window is a temporal or random holdout from the training "
            "population. Score drift uses out-of-fold calibrated scores."
        ),
        "environment": environment_metadata(),
    }
    write_json(REPORTS_DIR / "drift_summary.json", summary)

    from src.reporting import update_json

    update_json(REPORTS_DIR / "metrics.json", "monitoring", summary)

    print(f"Drift split: {split_description}")
    print(
        f"  reference {len(reference):,} rows, current {len(current):,} rows; "
        f"score PSI {score_psi:.4f}"
    )
    print(f"  status counts: {status_counts}")
    if trigger["retrain_review"]:
        print("  RETRAIN REVIEW TRIGGERED")
        for reason in trigger["reasons"]:
            print(f"    - {reason}")
    else:
        print("  No retrain trigger.")
    print(f"Saved drift report to {REPORTS_DIR / 'drift_summary.json'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["temporal", "random"], default="temporal")
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()
    run_drift(split=args.split, bins=args.bins)


if __name__ == "__main__":
    main()
