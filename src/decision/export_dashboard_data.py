"""Export real, aggregate Tableau-ready CSVs from canonical reports.

This replaces the old exporter that recomputed a uniform 99-point threshold
sweep from raw probabilities and hard-coded the minimax cutoff at 0.05. Every
number here comes from the generated decision, bootstrap, power, monitoring,
and explainability reports.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import DASHBOARD_DATA_DIR, REPORTS_DIR, ensure_directories
from src.reporting import environment_metadata, load_json, write_json


def _read_csv(name: str) -> pd.DataFrame:
    path = REPORTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the full pipeline before exporting Tableau data."
        )
    return pd.read_csv(path)


def _percent(series: pd.Series) -> pd.Series:
    return (series.astype(float) * 100.0).round(4)


def _threshold_curve(policy: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve = _read_csv("profit_curves.csv")
    curve["profit_millions"] = (curve["profit"] / 1e6).round(4)
    curve["oracle_profit_millions"] = (curve["oracle_profit"] / 1e6).round(4)
    curve["regret_pct"] = _percent(curve["regret"])
    curve["cost_ratio"] = curve["cost_ratio"].astype(float).round(3)
    curve["scenario_label"] = curve["scenario"].str.title()
    curve = curve.rename(columns={"threshold": "probability_cutoff"})

    binary_threshold = float(policy["binary_minimax"]["threshold"])
    grow_cutoff = float(policy["action_policy"]["grow_cutoff"])
    restrict_cutoff = float(policy["action_policy"]["restrict_cutoff"])
    heuristics = policy["heuristics"]
    markers = {
        "F1-optimal": float(heuristics["f1"]),
        "KS-optimal": float(heuristics["ks"]),
        "Top 4% reject": float(heuristics["top_4pct"]),
        "Binary minimax": binary_threshold,
        "Action grow cutoff": grow_cutoff,
        "Action restrict cutoff": restrict_cutoff,
    }

    marker_rows: list[dict[str, Any]] = []
    for scenario, group in curve.groupby("scenario"):
        group = group.sort_values("probability_cutoff")
        for label, threshold in markers.items():
            position = int(
                np.argmin(np.abs(group["probability_cutoff"].to_numpy() - threshold))
            )
            row = group.iloc[position]
            marker_rows.append(
                {
                    "scenario": scenario,
                    "scenario_label": scenario.title(),
                    "cost_ratio": float(row["cost_ratio"]),
                    "policy": label,
                    "probability_cutoff": float(threshold),
                    "nearest_curve_cutoff": float(row["probability_cutoff"]),
                    "profit_millions": float(row["profit_millions"]),
                    "regret_pct": float(row["regret_pct"]),
                }
            )
    return curve, pd.DataFrame(marker_rows)


def _action_mix() -> pd.DataFrame:
    frame = _read_csv("action_policy.csv")
    frame["share_pct"] = _percent(frame["share"])
    frame["bad_rate_pct"] = _percent(frame["bad_rate"])
    frame["expected_loss_millions"] = (frame["expected_loss"] / 1e6).round(4)
    frame["exposure_total_millions"] = (frame["exposure_total"] / 1e6).round(4)
    frame["scenario_label"] = frame["scenario"].str.title()
    return frame[
        [
            "scenario",
            "scenario_label",
            "action",
            "share",
            "share_pct",
            "bad_rate",
            "bad_rate_pct",
            "margin",
            "loss",
            "exposure",
            "expected_loss_millions",
            "exposure_total_millions",
        ]
    ]


def _operating_characteristics() -> pd.DataFrame:
    frame = _read_csv("operating_characteristics.csv")
    # These two source columns hold Python dicts; Tableau cannot aggregate
    # them. The flattened action shares live in kpi.csv and action_mix.csv.
    frame = frame.drop(columns=["action_mix", "bad_rate_by_action"], errors="ignore")
    for column in (
        "profit_uplift_vs_binary",
        "binary_regret_vs_action_oracle",
        "action_regret",
        "expected_loss_change",
        "exposure_change",
    ):
        frame[f"{column}_pct"] = _percent(frame[column])
    frame["binary_profit_millions"] = (frame["binary_profit"] / 1e6).round(4)
    frame["action_profit_millions"] = (frame["action_profit"] / 1e6).round(4)
    frame["action_oracle_profit_millions"] = (
        frame["action_oracle_profit"] / 1e6
    ).round(4)
    frame["scenario_label"] = frame["scenario"].str.title()
    return frame


def _score_histogram() -> pd.DataFrame:
    frame = _read_csv("score_histogram.csv")
    frame["share_pct"] = _percent(frame["weight"] / frame["weight"].sum())
    frame["bad_rate_pct"] = _percent(frame["bad_rate"])
    frame["action_label"] = frame["action"].str.title()
    return frame


def _calibration() -> pd.DataFrame:
    frame = _read_csv("calibration_bins.csv")
    frame["mean_predicted_pct"] = _percent(frame["mean_predicted"])
    frame["observed_pct"] = _percent(frame["observed"])
    frame["weight_share_pct"] = _percent(frame["weight_share"])
    return frame


def _sensitivity() -> pd.DataFrame:
    frame = _read_csv("sensitivity.csv")
    for column in (
        "f1_captured",
        "ks_captured",
        "binary_minimax_captured",
        "action_policy_captured",
        "binary_minimax_regret",
        "action_policy_regret",
    ):
        frame[f"{column}_pct"] = _percent(frame[column])
    return frame


def _drift() -> pd.DataFrame:
    frame = _read_csv("drift_features.csv")
    frame["missing_delta_pct"] = _percent(frame["missing_delta"])
    return frame.head(50)


def _kpi(policy: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for row in policy["comparison"]:
        rows.append(
            {
                "scenario": row["scenario"],
                "scenario_label": row["scenario"].title(),
                "cost_ratio": row["cost_ratio"],
                "binary_profit_millions": round(row["binary_profit"] / 1e6, 4),
                "action_profit_millions": round(row["action_profit"] / 1e6, 4),
                "oracle_profit_millions": round(
                    row["action_oracle_profit"] / 1e6, 4
                ),
                "profit_uplift_pct": round(
                    row["profit_uplift_vs_binary"] * 100, 4
                ),
                "binary_regret_vs_oracle_pct": round(
                    row["binary_regret_vs_action_oracle"] * 100, 4
                ),
                "action_regret_pct": round(row["action_regret"] * 100, 4),
                "expected_loss_change_pct": round(
                    row["expected_loss_change"] * 100, 4
                ),
                "exposure_change_pct": round(row["exposure_change"] * 100, 4),
                "grow_share_pct": round(row["action_mix"]["grow"] * 100, 4),
                "hold_share_pct": round(row["action_mix"]["hold"] * 100, 4),
                "restrict_share_pct": round(
                    row["action_mix"]["restrict"] * 100, 4
                ),
            }
        )
    return pd.DataFrame(rows)


def _policy_card(policy: dict[str, Any]) -> pd.DataFrame:
    action = policy["action_policy"]
    return pd.DataFrame(
        [
            {
                "binary_minimax_cutoff": policy["binary_minimax"]["threshold"],
                "grow_cutoff": action["grow_cutoff"],
                "hold_upper_cutoff": action["restrict_cutoff"],
                "action_worst_case_regret_pct": round(
                    action["worst_case_regret"] * 100, 4
                ),
                "binary_worst_case_regret_pct": round(
                    policy["binary_minimax"]["max_regret"] * 100, 4
                ),
                "max_grow_share": action["constraints"]["max_grow_share"],
                "min_hold_share": action["constraints"]["min_hold_share"],
                "max_expected_loss_increase": action["constraints"][
                    "max_expected_loss_increase"
                ],
                "grow_margin_multiplier": action["actions"]["grow"][
                    "margin_multiplier"
                ],
                "grow_exposure_multiplier": action["actions"]["grow"][
                    "exposure_multiplier"
                ],
                "hold_margin_multiplier": action["actions"]["hold"][
                    "margin_multiplier"
                ],
                "hold_exposure_multiplier": action["actions"]["hold"][
                    "exposure_multiplier"
                ],
            }
        ]
    )


def _bootstrap() -> pd.DataFrame:
    payload = load_json(REPORTS_DIR / "bootstrap.json")
    rows = []
    for scenario, result in payload.get("results", {}).items():
        for comparison in ("action_vs_binary", "binary_vs_f1"):
            values = result[comparison]
            rows.append(
                {
                    "scenario": scenario,
                    "scenario_label": scenario.title(),
                    "comparison": comparison,
                    "observed_difference_millions": round(
                        values["observed_difference"] / 1e6, 4
                    ),
                    "ci_low_millions": round(values["ci_low"] / 1e6, 4),
                    "ci_high_millions": round(values["ci_high"] / 1e6, 4),
                    "p_value": values["p_value"],
                    "probability_positive_pct": round(
                        values["probability_positive"] * 100, 4
                    ),
                }
            )
    return pd.DataFrame(rows)


def _power() -> pd.DataFrame:
    payload = load_json(REPORTS_DIR / "power_analysis.json")
    frame = pd.DataFrame(payload["results"])
    frame["power_pct"] = _percent(frame["power"])
    frame["required_n_per_arm_80pct"] = payload.get("required_n_per_arm_80pct")
    return frame


def export_tableau_data() -> dict[str, Any]:
    ensure_directories()
    DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    policy = load_json(REPORTS_DIR / "policy.json")
    if not policy:
        raise FileNotFoundError("reports/policy.json not found. Run make decide.")

    curve, markers = _threshold_curve(policy)
    outputs = {
        "threshold_curve.csv": curve,
        "policy_markers.csv": markers,
        "action_mix.csv": _action_mix(),
        "operating_characteristics.csv": _operating_characteristics(),
        "score_histogram.csv": _score_histogram(),
        "calibration.csv": _calibration(),
        "sensitivity.csv": _sensitivity(),
        "drift.csv": _drift(),
        "kpi.csv": _kpi(policy),
        "policy_card.csv": _policy_card(policy),
        "bootstrap.csv": _bootstrap(),
        "power.csv": _power(),
    }
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "canonical reports/metrics.json and reports/policy.json",
        "files": {},
        "environment": environment_metadata(),
    }
    for filename, frame in outputs.items():
        path = DASHBOARD_DATA_DIR / filename
        frame.to_csv(path, index=False)
        manifest["files"][filename] = {
            "rows": int(len(frame)),
            "columns": list(frame.columns),
        }

    write_json(DASHBOARD_DATA_DIR / "manifest.json", manifest)
    print(f"Exported {len(outputs)} Tableau-ready files to {DASHBOARD_DATA_DIR}")
    for filename, frame in outputs.items():
        print(f"  {filename:32s} {len(frame):,} rows")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    export_tableau_data()


if __name__ == "__main__":
    main()
