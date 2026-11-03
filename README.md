# Breakpoint

**A calibrated credit-risk model and a constrained Grow / Hold / Restrict policy.**

Most credit-risk projects stop at AUC. Breakpoint asks the next question: *given the economics of a good account and a default, what should we do with this customer — and how much does that decision make or lose?*

The project trains a 200-feature LightGBM model on 458,913 American Express customers, calibrates the out-of-fold probabilities, and optimizes a three-action policy under explicit portfolio guardrails. It is not an approve/decline exercise. The policy grows the safest accounts, holds the middle, and exits the riskiest — and beats a binary minimax threshold in every economic scenario tested.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tableau](https://img.shields.io/badge/Tableau-ready%20aggregates-E97627.svg)](dashboards/data/)

**Reviewer paths:** [static demo](docs/index.html) · [results](docs/RESULTS.md) · [model card](docs/MODEL_CARD.md) · [decision memo](docs/DECISION_MEMO.md) · [Tableau guide](docs/TABLEAU_DASHBOARD_GUIDE.md) · [interview defense](docs/INTERVIEW_DEFENSE.md)

---

## Headline results

<!-- BEGIN GENERATED: readme_headline -->
| Metric | Value | Evidence |
| --- | ---: | ---: |
| Calibrated AUC | 0.9584 | Out-of-fold, isotonic |
| Amex competition metric | 0.7814 | Out-of-fold |
| Brier score | 0.012055 | Population-weighted |
| ECE (quantile) | 0.0075% | Population-weighted |
| Grow / Hold / Restrict | 60.0% / 31.1% / 9.0% | Population-weighted action mix |
| Worst-case regret | 1.83% | Binary minimax: 4.26% |
| Profit uplift vs binary | +5.1% to +9.7% | Conservative through stress |
| Expected loss / exposure | -17.8% / +30.7% | Stress scenario vs binary |
| Bootstrap advantage | $102.0M [$100.6M, $103.5M] | Stress, 1,000 paired replicates |
| Champion/challenger | 17,610 / arm | 80% power, stress distribution |
| Monitoring | score PSI 0.0003; 199 green / 1 yellow | Temporal 80/20 holdout |
<!-- END GENERATED: readme_headline -->

The central comparison is **binary minimax** versus the **three-action robust policy**. Binary minimax picks one cutoff that minimizes worst-case regret across four scenarios. The three-action policy chooses Grow, Hold, or Restrict per account, subject to a 60% Grow cap, a 10% Hold floor, and a 10% expected-loss increase limit. Binary leaves **6.5%–10.5%** of the action-oracle profit on the table; the three-action policy leaves **0.15%–1.83%**.

## The policy

<!-- BEGIN GENERATED: readme_policy -->
| Action | Share | Bad rate | Margin | Exposure |
| --- | ---: | ---: | ---: | ---: |
| Grow | 60.0% | 0.02% | +10% | +50% |
| Hold | 31.1% | 0.90% | baseline | baseline |
| Restrict | 9.0% | 15.92% | -100% | -100% |

Cutoffs: Grow ≤ `0.000827`, Hold ≤ `0.040517`, Restrict > `0.040517`. Constraints: Grow ≤ 60%, Hold ≥ 10%, expected-loss increase ≤ 10%.
<!-- END GENERATED: readme_policy -->

The policy is deliberately simple to audit:

- **Grow** if calibrated PD ≤ `0.000827` — increase the line 50%, margin scales 10%.
- **Hold** if calibrated PD ≤ `0.040517` — keep the current line.
- **Restrict** above `0.040517` — exit the line: zero margin, zero exposure.

Those cutoffs are not guessed. They are the minimax-regret solution over a log-spaced and score-quantile grid, subject to the guardrails above. The optimizer binds the 60% Grow cap exactly. A partial-restrict alternative (50% line reduction) is also evaluated; its worst-case regret is **2.34%**, worse than the full-exit policy's **1.83%**.

## Scenario comparison

<!-- BEGIN GENERATED: readme_scenarios -->
| Scenario | Binary | 3-action | Oracle | Uplift | Binary regret | Action regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Conservative | $1,176.6M | $1,256.7M | $1,258.5M | +6.8% | 6.5% | 0.15% |
| Moderate | $3,116.6M | $3,286.1M | $3,333.0M | +5.4% | 6.5% | 1.41% |
| Aggressive | $6,333.4M | $6,654.7M | $6,774.6M | +5.1% | 6.5% | 1.77% |
| Stress | $1,051.5M | $1,153.5M | $1,175.1M | +9.7% | 10.5% | 1.83% |
<!-- END GENERATED: readme_scenarios -->

All dollar values are **expected profit under labelled economic assumptions**, not actual Amex P&L. The assumptions are in [`config/economic_scenarios.yaml`](config/economic_scenarios.yaml). The point is not that one dollar figure is true; it is that the same policy is robust across four very different margin/loss regimes.

## Why this is not another approve/decline model

A binary threshold answers one question: *approve or decline?* That collapses three economically different actions into one:

1. A safe account should often get **more** line, not just approval.
2. A marginal account should be **held**, not approved at the same exposure or declined.
3. A risky account should be **exited**, not merely declined at origination.

The policy makes the action explicit, prices the exposure change, and reports the action mix and bad rate for each band. That is a decision a risk committee can review.

## Architecture

```
breakpoint/
├── config/
│   ├── settings.yaml              # seed, 20× negative weight, paths, CV
│   └── economic_scenarios.yaml    # margin / LGD / exposure assumptions
├── data/
│   ├── raw/                       # Kaggle CSVs (not committed)
│   ├── parquet/                   # DuckDB-converted parquet (not committed)
│   ├── processed/                 # customer-level features + OOF (not committed)
│   └── demo/scoring_input.csv     # 500-row real-schema scoring sample (committed)
├── src/
│   ├── config.py                  # central YAML loader
│   ├── data/                      # CSV → parquet, validation
│   ├── features/                  # SQL feature build + post-processing
│   ├── models/                    # LightGBM, LR baseline, cross-fitted calibration
│   ├── policy/                    # constrained 3-action optimizer
│   ├── economics/                 # profit curves, regret, sensitivity
│   ├── evaluation/                # DeLong, paired bootstrap, power analysis
│   ├── monitoring/                # customer-level PSI, score/action drift
│   ├── explainability/            # SHAP on the saved model, reason codes
│   ├── scoring/                   # strict artifact bundle + scoring CLI
│   ├── decision/                  # policy reports + Tableau CSV export
│   └── demo/                      # static reviewer page generator
├── artifacts/                     # committed model + calibrator + contract
├── reports/                       # committed metrics, policy, figures, CSVs
├── dashboards/data/               # 12 Tableau-ready aggregate CSVs + manifest
├── figures/                       # generated plots
├── docs/                          # results, model card, memo, guides, static demo
├── tests/                         # 48 tests: economics, calibration, drift, scoring, contracts
├── scripts/                       # docs render/verify
├── Makefile
└── .github/workflows/ci.yml
```

## Data and weighting

The Amex/Kaggle training data is a 5% negative-class sample: the observed default rate is **25.89%**, while the population-weighted rate is **1.717%**. Every economic calculation, calibration metric, score drift, and action mix applies a **20× weight to non-defaults** to undo that sampling. The weight is defined once in [`config/settings.yaml`](config/settings.yaml) and tested.

The model uses **200 features**: 198 numeric and 2 categorical (`D_63`, `D_64`). `S_2` is the statement date and is used only for the temporal monitoring split, not as a model feature.

The public test labels are unavailable. Everything in this repository is **out-of-fold evidence on the labelled training population**, not a live out-of-time validation result. The 1% test sample is not used for performance claims.

## Reproduce

### Fast path (no 6.2 GB data download)

The persisted model, calibrator, feature contract, policy, and reports are committed.

```bash
git clone https://github.com/mahinigam/breakpoint.git
cd breakpoint
make setup
make score-demo      # score 500 real-schema accounts with the committed model
make demo            # rebuild docs/index.html from committed reports
make test            # 48 tests
make verify-docs     # fail if docs drift from reports/metrics.json or policy.json
```

`make score-demo` runs the same artifact bundle used by the CLI:

```bash
python -m src.scoring.score \
  --input data/demo/scoring_input.csv \
  --output reports/scored_demo.csv \
  --explain --explain-rows 20
```

### Full path (requires the Kaggle data)

Place the Amex CSV files in `data/raw/`, then:

```bash
make pipeline        # data → features → train → calibrate → policy → uncertainty → monitoring → SHAP → Tableau → demo
make tableau         # refresh the 12 aggregate CSVs in dashboards/data/
make verify          # tests + docs check + demo rebuild
```

The full training run uses 5-fold stratified OOF predictions, a final model, cross-fitted isotonic calibration, and a 1,000-replicate paired bootstrap. The full pipeline is data-heavy and may take tens of minutes after the parquet conversion; the committed artifacts make the reviewer path independent of that run.

## Calibration

AUC is a ranking metric. Expected profit needs absolute probabilities. Raw LightGBM AUC is **0.9585**; isotonic calibration keeps AUC at **0.9584** and reduces Brier from **0.012211** to **0.012055**. Quantile ECE falls from **0.2399%** to **0.0075%**.

Calibration is cross-fitted: each fold's calibrator is fit on the other folds and evaluated out of fold. The final isotonic map is then fit on all OOF rows for scoring. This avoids the common mistake of fitting and evaluating a calibrator on the same rows.

## Uncertainty and power

A paired, class-stratified bootstrap resamples customers 1,000 times, preserving the 20× population weight and the pairing between policies. The action-vs-binary profit advantage is positive in every scenario and every replicate. In the stress scenario it is **+$102.0M** with a 95% interval of **[$100.6M, $103.5M]**.

The power analysis simulates a two-arm champion/challenger test from the population-weighted score/label distribution. A test has **80% power at roughly 17,610 accounts per arm**; 25,000 per arm reaches **93.4%**, and 50,000 reaches **99.6%**. This is an offline design, not a live A/B test.

## Monitoring

Monitoring is customer-level, not statement-level. The report splits the labelled population by statement date: earliest 80% reference vs latest 20% current. Feature PSI is unweighted (observed account distributions); score PSI and action mix are population-weighted.

- Score PSI: **0.0003**.
- Weighted default rate: **1.7259% → 1.6810%**.
- Action mix: **60.07 / 31.02 / 8.92 → 59.73 / 31.19 / 9.08**.
- Feature status: **199 green, 1 yellow**, no red.
- Retrain review: **not triggered**.

The trigger is explicit: >10% of features with PSI > 0.25, or ≥3 delinquency/balance features above 0.25, or score PSI > 0.25, or an action share moving >5 percentage points. These are review triggers, not automatic retraining.

## Explainability

SHAP is computed with `TreeExplainer` on the **saved** LightGBM model, not a proxy. The sample is 5,000 customers. The top features are `P_2`, `payment_to_balance_ratio`, `P_2_rolling3_mean`, `B_4`, and `D_39`; delinquency, payment, and balance families account for roughly 80% of total mean absolute SHAP.

`reports/reason_codes.csv` contains 25 restricted accounts spanning the full Restrict band, each with three plain-English reason codes. Because the Amex features are anonymized, reason codes are family-level (e.g., "D_39 (Delinquency) increased risk [SHAP +0.42; value 2.0]"), not regulator-ready adverse-action notices.

## Tableau

`dashboards/data/` contains 12 aggregate CSVs generated from the canonical reports: threshold curves, policy markers, action mix, operating characteristics, score histogram, calibration, sensitivity, drift, KPIs, policy card, bootstrap, and power. `dashboards/data/manifest.json` records every file's row count and columns.

The build guide is [`docs/TABLEAU_DASHBOARD_GUIDE.md`](docs/TABLEAU_DASHBOARD_GUIDE.md). It covers the joins, calculated fields, dashboard layout, and publishing steps. **Publishing to Tableau Public requires your Tableau account**; the repository prepares the data and workbook recipe, but does not claim a published URL.

## Limitations

- **Offline evidence.** The test labels are unavailable. All performance and profit numbers are out-of-fold on the labelled training population.
- **Assumed economics.** Margin, LGD, and exposure are labelled scenarios, not Amex's actual P&L. The policy is robust across them, but the dollar levels are illustrative.
- **Anonymized features.** Fairness, adverse-action, and regulatory reviews require named features and protected-class testing. This project does not claim those.
- **No production system.** There is no feature store, real-time latency test, model-risk sign-off, or live monitoring integration.
- **Monitoring is a temporal holdout, not true out-of-time.** The current window is the latest 20% of the same labelled month; it tests the monitoring machinery and observed feature stability, not next-year drift.
- **SHAP is not causal.** It attributes the model's prediction, not the effect of changing a feature.

## Documentation

| Document | What it answers |
|---|---|
| [`docs/RESULTS.md`](docs/RESULTS.md) | Every metric, table, and caveat, generated from `reports/metrics.json` and `reports/policy.json` |
| [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) | Intended use, training data, metrics, fairness, limitations |
| [`docs/DECISION_MEMO.md`](docs/DECISION_MEMO.md) | One-page recommendation for a risk committee |
| [`docs/TABLEAU_DASHBOARD_GUIDE.md`](docs/TABLEAU_DASHBOARD_GUIDE.md) | Build and publish the Tableau dashboard |
| [`docs/INTERVIEW_DEFENSE.md`](docs/INTERVIEW_DEFENSE.md) | Technical defense of each design choice |
| [`docs/index.html`](docs/index.html) | Static reviewer demo, no JavaScript or external fonts |

## License

MIT. See [`LICENSE`](LICENSE).

The Amex/Kaggle data is not included. Download it from the competition page and comply with its terms.
