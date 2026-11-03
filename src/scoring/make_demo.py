"""Create a small, real sample of customer features for the scoring demo.

The sample is drawn from the processed customer feature table. It contains no
labels, only the model's feature contract and a customer identifier.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, NEGATIVE_WEIGHT, PATHS, SEED, ensure_directories
from src.reporting import load_json


def make_demo(rows: int = 200) -> Path:
    ensure_directories()
    contract = load_json(ARTIFACTS_DIR / "feature_list.json")
    if not contract:
        raise FileNotFoundError(
            "artifacts/feature_list.json not found. Run `make train` first."
        )
    features = contract["all_features"]
    frame = pd.read_parquet(
        PATHS["processed_features"], columns=["customer_ID", *features]
    )
    # The modelling sample oversamples defaults. Sampling with the 20x
    # population weight produces a demo that resembles the served portfolio
    # (~1.7% defaults) instead of the 26% modelling sample.
    labels = pd.read_parquet(
        PATHS["parquet_train_labels"], columns=["customer_ID", "target"]
    )
    frame = frame.merge(labels, on="customer_ID", how="left", validate="one_to_one")
    weights = np.where(frame["target"].to_numpy() == 0, float(NEGATIVE_WEIGHT), 1.0)
    sample = frame.sample(
        n=min(rows, len(frame)), weights=weights, random_state=SEED
    ).reset_index(drop=True)
    observed_rate = float(sample["target"].mean())
    sample = sample.drop(columns=["target"])

    output = PATHS["demo"] / "scoring_input.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output, index=False)
    print(
        f"Wrote {len(sample):,} rows x {len(sample.columns)} columns to {output} "
        f"(sampled default rate {observed_rate:.2%})"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=200)
    args = parser.parse_args()
    make_demo(rows=args.rows)


if __name__ == "__main__":
    main()
