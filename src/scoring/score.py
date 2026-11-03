"""Score customer-level features with the persisted champion model.

Example
-------
``python -m src.scoring.score --input data/demo/scoring_input.csv``

The input must contain the 200 model features (plus ``customer_ID``). It is the
customer-level feature table, not raw transaction statements. Use the training
feature builder to create it from raw data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap

from src.config import ARTIFACTS_DIR, REPORTS_DIR, ensure_directories
from src.models.preprocessing import apply_categoricals
from src.reporting import environment_metadata, write_json
from src.scoring.artifacts import ArtifactBundle


def read_input(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("Input must be .parquet or .csv")


def _add_reason_codes(
    bundle: ArtifactBundle, frame: pd.DataFrame, scored: pd.DataFrame, max_rows: int
) -> pd.DataFrame:
    n_explain = min(len(frame), max_rows)
    if n_explain == 0:
        return scored
    x_frame = apply_categoricals(
        frame.iloc[:n_explain][bundle.features],
        bundle.categorical,
        bundle.levels,
    )
    explainer = shap.TreeExplainer(bundle.model)
    values = explainer.shap_values(x_frame)
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]

    for position in range(n_explain):
        row_values = values[position]
        order = np.argsort(row_values)[::-1]
        order = [index for index in order if row_values[index] > 0][:3]
        for rank, feature_index in enumerate(order, start=1):
            scored.loc[scored.index[position], f"reason_{rank}_feature"] = (
                bundle.features[feature_index]
            )
            scored.loc[scored.index[position], f"reason_{rank}_shap"] = float(
                row_values[feature_index]
            )
    return scored


def score_file(
    input_path: Path,
    output_path: Path,
    scenario: str = "moderate",
    limit: int | None = None,
    explain: bool = False,
    explain_rows: int = 100,
) -> dict[str, Any]:
    ensure_directories()
    bundle = ArtifactBundle.load(scenario_name=scenario)
    frame = read_input(input_path)
    if limit is not None:
        frame = frame.head(limit).copy()
    scored = bundle.score_frame(frame)
    if explain:
        scored = _add_reason_codes(bundle, frame, scored, explain_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        scored.to_parquet(output_path, index=False)
    else:
        scored.to_csv(output_path, index=False)

    action_counts = {
        str(key): int(value)
        for key, value in scored["action"].value_counts().to_dict().items()
    }
    summary = {
        "n_rows": int(len(scored)),
        "action_counts": action_counts,
        "action_shares": {
            key: value / len(scored) for key, value in action_counts.items()
        },
        "mean_probability_of_default": float(
            scored["probability_of_default"].mean()
        ),
        "grow_cutoff": bundle.grow_cutoff,
        "restrict_cutoff": bundle.restrict_cutoff,
        "scenario": scenario,
        "input": str(input_path),
        "output": str(output_path),
        "explained_rows": int(min(len(frame), explain_rows)) if explain else 0,
        "environment": environment_metadata(),
    }
    write_json(REPORTS_DIR / "scoring_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=REPORTS_DIR / "scored_accounts.csv"
    )
    parser.add_argument(
        "--scenario",
        choices=["conservative", "moderate", "aggressive", "stress"],
        default="moderate",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--explain-rows", type=int, default=100)
    args = parser.parse_args()

    summary = score_file(
        input_path=args.input,
        output_path=args.output,
        scenario=args.scenario,
        limit=args.limit,
        explain=args.explain,
        explain_rows=args.explain_rows,
    )
    print(f"Scored {summary['n_rows']:,} accounts -> {args.output}")
    print(f"  actions: {summary['action_counts']}")
    print(f"  mean PD: {summary['mean_probability_of_default']:.4%}")
    print(
        f"  policy: grow <= {summary['grow_cutoff']:.6f}, "
        f"restrict > {summary['restrict_cutoff']:.6f}"
    )


if __name__ == "__main__":
    main()
