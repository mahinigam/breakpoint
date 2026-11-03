"""SHAP explainability and reason codes for the saved champion model.

The previous version trained a *proxy* model on a 10% sample and explained the
proxy. This version loads the exact persisted LightGBM model, feature contract,
category levels, and calibrator, so the explanations describe the model that is
actually deployed.

SHAP values are in the model's raw output space (log-odds / margin), not in
probability units. Probability-of-default values in the reason-code table come
from the persisted calibrator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import (
    ARTIFACTS_DIR,
    FIGURES_DIR,
    PATHS,
    REPORTS_DIR,
    SEED,
    ensure_directories,
)
from src.models.preprocessing import apply_categoricals
from src.monitoring.psi import family_for
from src.reporting import environment_metadata, load_json, update_json, write_json

FAMILY_LABELS = {
    "Delinquency": "Delinquency (D)",
    "Spend": "Spend (S)",
    "Payment": "Payment (P)",
    "Balance": "Balance (B)",
    "Risk": "Risk (R)",
    "Engineered/Other": "Engineered / Other",
}


def _positive_class_shap(values: Any) -> np.ndarray:
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    return values


def _calibrate(raw: np.ndarray, calibrator: dict[str, Any]) -> np.ndarray:
    method = calibrator.get("method", "isotonic")
    if method == "isotonic":
        return np.asarray(calibrator["isotonic"].predict(raw), dtype=float)
    return np.asarray(
        calibrator["platt"].predict_proba(raw.reshape(-1, 1))[:, 1], dtype=float
    )


def _action(pd_value: float, grow_cutoff: float, restrict_cutoff: float) -> str:
    if pd_value <= grow_cutoff:
        return "Grow"
    if pd_value <= restrict_cutoff:
        return "Hold"
    return "Restrict"


def _reason_text(feature: str, shap_value: float, feature_value: Any) -> str:
    family = FAMILY_LABELS.get(family_for(feature), "Engineered / Other")
    direction = "increased" if shap_value >= 0 else "decreased"
    try:
        value_text = f"{float(feature_value):.4g}"
    except (TypeError, ValueError):
        value_text = str(feature_value)
    return (
        f"{feature} ({family}) {direction} risk "
        f"[SHAP {shap_value:+.3f}; value {value_text}]"
    )


def _plot_family_importance(
    family_importance: pd.DataFrame, output_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.barh(
        family_importance["family_label"],
        family_importance["mean_abs_shap"],
        color="#1f77b4",
    )
    ax.set_xlabel("Mean absolute SHAP value (raw model output)")
    ax.set_title("Feature-family contribution to model risk")
    ax.grid(True, axis="x", linestyle="--", alpha=0.25)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "shap_family_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_top_risk_contributions(
    row: pd.Series,
    shap_row: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
) -> None:
    order = np.argsort(shap_row)[::-1]
    top = [index for index in order if shap_row[index] > 0][:10]
    if not top:
        top = list(order[:10])
    labels = [feature_names[index] for index in top][::-1]
    values = [shap_row[index] for index in top][::-1]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(labels, values, color="#d62728")
    ax.set_xlabel("SHAP value (raw model output, positive = higher risk)")
    ax.set_title(f"Top risk contributors for {row['customer_ID']}")
    ax.grid(True, axis="x", linestyle="--", alpha=0.25)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / "shap_top_risk_contributions.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def run_explainability(n_rows: int = 5000, n_reasons: int = 25) -> dict[str, Any]:
    ensure_directories()
    model = joblib.load(ARTIFACTS_DIR / "lgbm_model.joblib")
    calibrator = joblib.load(ARTIFACTS_DIR / "calibrator.joblib")
    feature_contract = load_json(ARTIFACTS_DIR / "feature_list.json")
    levels = load_json(ARTIFACTS_DIR / "categorical_levels.json")
    policy = load_json(REPORTS_DIR / "policy.json")
    features = feature_contract["all_features"]
    categorical = feature_contract["categorical_features"]
    grow_cutoff = float(policy["action_policy"]["grow_cutoff"])
    restrict_cutoff = float(policy["action_policy"]["restrict_cutoff"])

    frame = pd.read_parquet(
        PATHS["processed_features"], columns=["customer_ID", *features]
    )
    if len(frame) > n_rows:
        frame = frame.sample(n=n_rows, random_state=SEED).reset_index(drop=True)
    x_frame = apply_categoricals(frame[features], categorical, levels)

    print(f"Computing SHAP values for {len(frame):,} customers")
    explainer = shap.TreeExplainer(model)
    shap_values = _positive_class_shap(explainer.shap_values(x_frame))
    if shap_values.shape != x_frame.shape:
        raise ValueError(
            f"SHAP shape {shap_values.shape} does not match features {x_frame.shape}"
        )

    mean_abs = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame(
        {
            "feature": features,
            "family": [family_for(feature) for feature in features],
            "mean_abs_shap": mean_abs,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    feature_importance["rank"] = np.arange(1, len(feature_importance) + 1)
    feature_importance["share_of_total"] = (
        feature_importance["mean_abs_shap"] / feature_importance["mean_abs_shap"].sum()
    )
    feature_importance.to_csv(REPORTS_DIR / "shap_feature_importance.csv", index=False)

    family_importance = (
        feature_importance.groupby("family", as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)
    )
    family_importance["family_label"] = family_importance["family"].map(
        lambda name: FAMILY_LABELS.get(name, name)
    )
    family_importance["share_of_total"] = (
        family_importance["mean_abs_shap"] / family_importance["mean_abs_shap"].sum()
    )
    family_importance.to_csv(REPORTS_DIR / "shap_family_importance.csv", index=False)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        x_frame,
        max_display=15,
        show=False,
        plot_size=None,
    )
    plt.title("SHAP beeswarm: top 15 model features")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()
    _plot_family_importance(family_importance, FIGURES_DIR)

    raw = np.asarray(model.predict_proba(x_frame)[:, 1], dtype=float)
    calibrated = _calibrate(raw, calibrator)
    # Reason codes should cover the whole Restrict band, not only the most
    # extreme 1.0-PD accounts. A regulator or reviewer needs to see what a
    # representative restricted account looks like, not just the tail.
    restrict_indices = np.flatnonzero(calibrated > restrict_cutoff)
    if len(restrict_indices) >= n_reasons:
        ordered = restrict_indices[np.argsort(calibrated[restrict_indices])]
        positions = np.linspace(0, len(ordered) - 1, n_reasons).astype(int)
        order = ordered[positions]
        # Rank 1 remains the highest-risk account; the selected rows still span
        # the whole Restrict band.
        order = order[np.argsort(calibrated[order])[::-1]]
    else:
        order = np.argsort(calibrated)[::-1][:n_reasons]

    rows: list[dict[str, Any]] = []
    for rank, index in enumerate(order, start=1):
        shap_row = shap_values[index]
        positive = np.argsort(shap_row)[::-1]
        positive = [position for position in positive if shap_row[position] > 0][:3]
        if not positive:
            positive = list(np.argsort(np.abs(shap_row))[::-1][:3])
        row: dict[str, Any] = {
            "rank": rank,
            "customer_ID": frame.iloc[index]["customer_ID"],
            "calibrated_pd": float(calibrated[index]),
            "action": _action(
                float(calibrated[index]), grow_cutoff, restrict_cutoff
            ),
        }
        for reason_index, feature_index in enumerate(positive, start=1):
            feature = features[feature_index]
            value = x_frame.iloc[index][feature]
            row[f"reason_{reason_index}_feature"] = feature
            row[f"reason_{reason_index}_family"] = family_for(feature)
            row[f"reason_{reason_index}_shap"] = float(shap_row[feature_index])
            row[f"reason_{reason_index}_value"] = (
                float(value) if pd.api.types.is_number(value) else str(value)
            )
            row[f"reason_{reason_index}_text"] = _reason_text(
                feature, float(shap_row[feature_index]), value
            )
        rows.append(row)

    reasons = pd.DataFrame(rows)
    reasons.to_csv(REPORTS_DIR / "reason_codes.csv", index=False)

    top_index = int(np.argmax(calibrated))
    _plot_top_risk_contributions(
        frame.iloc[top_index],
        shap_values[top_index],
        features,
        FIGURES_DIR,
    )

    payload = {
        "sample_rows": int(len(frame)),
        "n_features": int(len(features)),
        "top_features": feature_importance.head(15).to_dict(orient="records"),
        "top_families": family_importance.to_dict(orient="records"),
        "top_reason_codes": reasons.head(10).to_dict(orient="records"),
        "explainer": "shap.TreeExplainer on artifacts/lgbm_model.joblib",
        "shap_space": "raw LightGBM output (log-odds / margin)",
        "caveat": (
            "SHAP values explain the final model, which was fit on all labelled "
            "rows. The sample is illustrative, not an out-of-time validation set."
        ),
        "environment": environment_metadata(),
    }
    write_json(REPORTS_DIR / "explainability.json", payload)
    update_json(REPORTS_DIR / "metrics.json", "explainability", payload)

    print("Top features by mean absolute SHAP:")
    for row in feature_importance.head(8).itertuples(index=False):
        print(f"  {row.feature:24s} {row.mean_abs_shap:8.4f}  ({row.family})")
    print(f"Saved explainability report to {REPORTS_DIR / 'explainability.json'}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--reasons", type=int, default=25)
    args = parser.parse_args()
    run_explainability(n_rows=args.rows, n_reasons=args.reasons)


if __name__ == "__main__":
    main()
