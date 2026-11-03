"""Retrain trigger logic for the monitoring report.

The trigger is deliberately simple and auditable:

* more than 10% of model features have PSI > 0.25, or
* at least 3 delinquency/balance features have PSI > 0.25, or
* the calibrated score PSI exceeds 0.25, or
* any action's population share moves by more than 5 percentage points.

These are review triggers, not automatic retraining. A reviewer should inspect
the feature-level report and the score/action drift before changing the model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import REPORTS_DIR

PSI_RED = 0.25
ACTION_SHARE_TOLERANCE = 0.05


def evaluate_drift_triggers(
    psi_df: pd.DataFrame,
    score_psi: float | None = None,
    action_mix_delta: dict[str, float] | None = None,
    psi_threshold: float = PSI_RED,
) -> dict[str, object]:
    """Return a structured retrain-review decision with explicit reasons."""
    if psi_df.empty:
        return {
            "retrain_review": False,
            "reasons": ["No feature-level PSI results were supplied."],
            "features_above_threshold": 0,
            "pct_features_above_threshold": 0.0,
            "delinquency_balance_above_threshold": 0,
        }

    total_features = len(psi_df)
    drifted = psi_df[psi_df["psi"] > psi_threshold]
    pct_drifted = 100.0 * len(drifted) / total_features
    family = psi_df.get("family", pd.Series([""] * total_features))
    critical = drifted[
        family.loc[drifted.index].isin(["Delinquency", "Balance"])
        | drifted["feature"].astype(str).str.startswith(("D_", "B_"))
    ]

    reasons: list[str] = []
    if pct_drifted > 10.0:
        reasons.append(
            f"Global drift: {pct_drifted:.1f}% of model features "
            f"({len(drifted)}/{total_features}) have PSI > {psi_threshold}."
        )
    if len(critical) >= 3:
        reasons.append(
            f"Critical family drift: {len(critical)} delinquency/balance features "
            f"have PSI > {psi_threshold}."
        )
    if score_psi is not None and score_psi > psi_threshold:
        reasons.append(
            f"Score drift: calibrated score PSI is {score_psi:.3f}, above "
            f"{psi_threshold}."
        )
    if action_mix_delta:
        largest = max(action_mix_delta.items(), key=lambda item: abs(item[1]))
        if abs(largest[1]) > ACTION_SHARE_TOLERANCE:
            reasons.append(
                f"Action-mix drift: {largest[0]} share moved "
                f"{largest[1]:+.1%} points, above the "
                f"{ACTION_SHARE_TOLERANCE:.0%} tolerance."
            )

    return {
        "retrain_review": bool(reasons),
        "reasons": reasons,
        "features_above_threshold": int(len(drifted)),
        "pct_features_above_threshold": float(pct_drifted),
        "delinquency_balance_above_threshold": int(len(critical)),
        "score_psi": score_psi,
        "max_action_share_delta": (
            float(max(abs(value) for value in action_mix_delta.values()))
            if action_mix_delta
            else 0.0
        ),
        "thresholds": {
            "feature_psi_red": psi_threshold,
            "global_pct_features": 10.0,
            "critical_family_count": 3,
            "action_share_tolerance": ACTION_SHARE_TOLERANCE,
        },
    }


def generate_drift_flags(
    psi_df: pd.DataFrame, psi_threshold: float = PSI_RED
) -> tuple[bool, list[str]]:
    """Backwards-compatible wrapper returning ``(retrain_review, reasons)``."""
    result = evaluate_drift_triggers(psi_df, psi_threshold=psi_threshold)
    return bool(result["retrain_review"]), list(result["reasons"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=REPORTS_DIR / "drift_features.csv"
    )
    parser.add_argument(
        "--summary", type=Path, default=REPORTS_DIR / "drift_summary.json"
    )
    args = parser.parse_args()

    if not args.features.exists() or not args.summary.exists():
        raise FileNotFoundError(
            "Run `python -m src.monitoring.psi` first to generate the drift report."
        )
    psi_df = pd.read_csv(args.features)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    result = evaluate_drift_triggers(
        psi_df,
        score_psi=summary.get("score_psi"),
        action_mix_delta=summary.get("action_mix_delta"),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
