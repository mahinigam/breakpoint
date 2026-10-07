# Breakpoint: Profit-Optimal Credit Decisioning

> **A credit-line action policy that stays near-optimal even when the economics are unknown.**

[![CI](https://github.com/YOUR_USERNAME/breakpoint/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/breakpoint/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Why This Project Exists

Most credit risk projects optimize AUC and stop. But AUC doesn't answer the question the business actually asks:

> *"At what probability threshold should we grow, hold, or restrict a credit line — and how much money does that decision make us?"*

The answer depends on economics — margin per good account, loss given default, exposure — that are **never exactly known**. This project builds a decision policy that is **robust to that uncertainty**, using minimax regret from classical decision theory.

---

## What This Project Demonstrates

| Capability | What It Shows |
|---|---|
| **SQL Feature Engineering** | Point-in-time safe features (LAGs, rolling windows, ratios) in DuckDB, version-controlled |
| **Model Comparison** | Logistic regression baseline vs LightGBM with DeLong test for statistical significance |
| **Probability Calibration** | Platt scaling, isotonic regression, and proof that miscalibration destroys profit at equal AUC |
| **Profit-Optimal Decisioning** | Expected profit curves under 4 economic scenarios, 6 cutoff methods compared |
| **Minimax-Regret Policy** | A single cutoff that minimizes worst-case regret across all plausible cost ratios |
| **Statistical Rigor** | Paired-bootstrap 95% CIs on profit differences; simulation-based power analysis for champion-challenger test design |
| **Production Monitoring** | PSI per feature with automated drift flags and plain-English retrain triggers |
| **Explainability** | SHAP grouped by feature family; top-4 plain-English reason codes per credit-line decision |
| **Executive Communication** | One-page decision memo, Tableau dashboard, LaTeX resume bullets |
| **Engineering Hygiene** | pytest (including leakage and determinism tests), Makefile, GitHub Actions CI, model card |

---

## Architecture

```
breakpoint/
├── data/
│   ├── raw/                   # Original CSVs (gitignored)
│   ├── parquet/               # Converted Parquet files
│   └── processed/             # Customer-level feature table
├── config/
│   ├── settings.yaml          # Seeds, paths, subsample weight (20×)
│   └── economic_scenarios.yaml # Margin / LGD / exposure scenarios
├── sql/
│   ├── 01_csv_to_parquet.sql  # DuckDB streaming conversion
│   ├── 02_feature_engineering.sql  # Point-in-time windowed features
│   └── 03_collapse_to_customer.sql # One row per customer
├── src/
│   ├── config.py              # Central configuration loader
│   ├── data/                  # Data loading and validation
│   ├── features/              # Post-processing, dtype casting
│   ├── models/                # LR baseline, LightGBM, calibration
│   ├── economics/             # Profit curves, regret map, cutoff methods
│   ├── evaluation/            # Metrics, DeLong, bootstrap CIs, power analysis
│   ├── monitoring/            # PSI, drift flags, retrain triggers
│   └── explainability/        # SHAP analysis, reason codes
├── tests/                     # pytest suite (leakage, determinism, economics, PSI)
├── dashboards/                # Tableau workbook + data export scripts
├── docs/                      # RESULTS.md, INTERVIEW_DEFENSE.md, DECISION_MEMO.md,
│                              #   MODEL_CARD.md, resume_bullets.tex
├── figures/                   # All generated plots
├── Makefile                   # make data → features → train → evaluate → all
├── .github/workflows/ci.yml  # CI on 1% stratified sample
└── README.md
```

---

## Key Results

> All numbers are populated after pipeline execution and traceable to their source scripts.
> See [`docs/RESULTS.md`](docs/RESULTS.md) for the complete, verified results table.

| Metric | Logistic Regression | LightGBM | Source |
|---|---|---|---|
| AUC | 0.9555 | 0.9584 | `src/evaluation/metrics.py` |
| KS Statistic | 0.7774 | 0.7862 | `src/evaluation/metrics.py` |
| Gini Coefficient | 0.9111 | 0.9167 | `src/evaluation/metrics.py` |
| Competition Metric (M) | 0.7720 | 0.7811 | `src/evaluation/metrics.py` |
| Brier Score (calibrated) | — | 0.0120 (Isotonic) | `src/models/calibration.py` |
| DeLong p-value (AUC diff) | — | ~0.9999 (invalid) | `src/evaluation/delong.py` |

| Decision Metric | Value | Source |
|---|---|---|
| Profit improvement over fixed-0.5 cutoff (Stress) | +$714M | `src/evaluation/bootstrap.py` |
| Profit improvement over F1-optimal cutoff (Stress) | +$312M (95% CI: [+$308M, +$316M]) | `src/evaluation/bootstrap.py` |
| Minimax-regret cutoff | 0.0500 (Max Regret: 4.7%) | `src/economics/regret_map.py` |
| Accounts per arm (champion-challenger) | 10,000 | `src/evaluation/power_analysis.py` |
| Test duration at 100K/month volume | < 1 week | `src/evaluation/power_analysis.py` |
| Features flagged by PSI (>0.25) | 0 | `src/monitoring/psi.py` |

---

## The Core Thesis in Four Panels

### 1. Profit Curves Under Uncertainty
For any cost ratio `r = expected_loss / margin`, the profit-optimal cutoff is `τ*(r) = 1/(1+r)`. But `r` is never exactly known. We plot profit vs cutoff under four named economic scenarios (conservative, moderate, aggressive, stress) and show that F1-optimal, KS-optimal, and top-4% capture cutoffs leave money on the table.

### 2. Regret Map
A heatmap of `Regret(τ, r) = OracleProfit(r) − ActualProfit(τ, r)` across cutoffs and cost ratios. The minimax-regret cutoff minimizes the worst-case row of this matrix — it's the policy that's "least bad" when you don't know `r`.

### 3. Calibration → Profit Link
Two models with identical AUC can yield different profit because one is miscalibrated. We demonstrate this concretely and quantify the profit lost from using uncalibrated scores.

### 4. Champion-Challenger Test Design
A simulation-based power analysis (bootstrap, not Normal approximation) determines how many accounts per arm and how many weeks are needed to validate the minimax policy in production. **This is an offline design on historical data, not a live A/B test.**

---

## Data

**Source:** [Kaggle American Express Default Prediction](https://www.kaggle.com/competitions/amex-default-prediction)

| Attribute | Value |
|---|---|
| Training customers | 458,913 |
| Statements per customer | Up to 13 (avg ~12) |
| Raw features | 188 (anonymized, prefixed: D, S, P, B, R) |
| Observed default rate | 25.9% (after 5% negative subsampling) |
| Estimated true default rate | ~1.6% (20× weight on negatives) |
| Test labels | Not public — used only for unlabeled drift analysis |

> [!IMPORTANT]
> **Raw data is not included in this repository.** To reproduce, download from the Kaggle competition page (requires accepting competition rules) and place files in `data/raw/`.

### Subsampling Correction

The competition subsampled negatives at 5%, making the observed default rate 25.9%. In reality, the true default rate is ~1.6%. Every economic calculation in this project applies a **20× weight to negative (non-default) samples** to correct for this. This weight is defined once in `config/settings.yaml` and unit-tested.

---

## Decision Framing

These are **existing customers** with 13 months of behavioral data. The decision is not origination (approve/decline) — it is a credit-line action:

| Action | Criteria | Business Meaning |
|---|---|---|
| **Grow** | Low predicted PD, high spend patterns | Increase credit line |
| **Hold** | Moderate predicted PD | Maintain current line |
| **Restrict** | High predicted PD | Reduce line or flag for review |

The cutoff between Hold and Restrict is what this project optimizes.

---

## Economic Scenarios

All dollar values are **labeled assumptions**, not claims. The project's value is showing robustness *across* scenarios.

| Scenario | Margin | LGD | Exposure | r = loss/margin | Interpretation |
|---|---|---|---|---|---|
| Conservative | $200 | 80% | $5,000 | 20.0 | High-loss, low-margin product |
| Moderate | $500 | 60% | $5,000 | 6.0 | Typical revolving credit |
| Aggressive | $1,000 | 40% | $5,000 | 2.0 | Premium, high-margin product |
| Stress | $200 | 90% | $10,000 | 45.0 | Recession scenario |

---

## Quickstart

### Prerequisites

- Python 3.10+
- macOS (Apple Silicon supported)
- Kaggle account with competition rules accepted
- ~10 GB free disk space (after Parquet conversion)

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/breakpoint.git
cd breakpoint

# Install dependencies
make setup

# Download data from Kaggle (requires kaggle API token)
# Place train_data.csv, train_labels.csv, test_data.csv, sample_submission.csv in data/raw/

# Convert CSV → Parquet (streaming, out-of-core via DuckDB)
make data

# Run the full pipeline
make all

# Run tests
make test
```

### Makefile Targets

| Target | Description |
|---|---|
| `make setup` | Install Python dependencies |
| `make data` | CSV → Parquet conversion and validation |
| `make features` | SQL feature engineering → customer-level table |
| `make train` | Train LR baseline + LightGBM + calibration |
| `make evaluate` | Metrics, economics, bootstrap CIs, SHAP, PSI |
| `make all` | Full pipeline (setup → data → features → train → evaluate) |
| `make test` | Run pytest suite (full data) |
| `make test-ci` | Run pytest suite (1% sample, for CI) |
| `make clean` | Remove generated files (preserves data) |

---

## Monitoring & Drift Detection

PSI (Population Stability Index) is computed per feature between the training set and a 1% stratified sample of the test set.

**Retrain trigger (plain English):**
> *"If more than 10% of features exceed PSI > 0.25, or if any 3 features in the delinquency (D) or balance (B) families exceed PSI > 0.25, initiate model retraining review."*

**Limitation:** PSI is computed on a 1% sample of test data due to disk constraints. Confidence in drift detection is lower than on the full test set.

---

## Explainability

- **SHAP values** computed via `TreeExplainer` (exact, not approximate)
- **Grouped by feature family** (D, S, P, B, R) because individual features are anonymized
- **Top-4 reason codes** per decision in plain English
- **Limitation:** With anonymized features, reason codes are inferred from prefix families and directional behavior, not from human-interpretable feature names

---

## Deliverables

| Document | Description | Location |
|---|---|---|
| `RESULTS.md` | Every number, traceable to source script | [`docs/RESULTS.md`](docs/RESULTS.md) |
| `INTERVIEW_DEFENSE.md` | Interview Q&A for every phase, including limitations | [`docs/INTERVIEW_DEFENSE.md`](docs/INTERVIEW_DEFENSE.md) |
| `DECISION_MEMO.md` | One-page memo for a non-technical executive | [`docs/DECISION_MEMO.md`](docs/DECISION_MEMO.md) |
| `MODEL_CARD.md` | Google-format model card | [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) |
| `resume_bullets.tex` | Four LaTeX resume bullets using verified numbers | [`docs/resume_bullets.tex`](docs/resume_bullets.tex) |
| Tableau Dashboard | Executive dashboard (published to Tableau Public) | *URL TBD* |

---

## Known Limitations

1. **Anonymized features:** Cannot perform WoE binning, domain-specific feature naming, or fully interpretable reason codes. The LR baseline is a linear benchmark, not a production scorecard.
2. **No real dollar values:** All economic scenarios are labeled assumptions. The project demonstrates methodology robustness, not actual P&L impact.
3. **Subsampled negatives:** The 20× correction is applied everywhere, but the original sampling methodology is controlled by Kaggle/Amex, not by us.
4. **No test labels:** Model performance is evaluated via cross-validation on training data only. Test data is used exclusively for drift analysis.
5. **PSI on 1% sample:** Drift detection has reduced statistical power compared to full-data PSI.
6. **Offline test design:** The champion-challenger analysis is designed on historical data. A live deployment would require additional infrastructure, compliance review, and real-time monitoring.

---

## What I Would Do Differently With Real Data

- Use actual exposure amounts, margin rates, and recovery rates instead of scenarios
- Add macroeconomic features (unemployment rate, interest rates, GDP growth)
- Build a proper WoE scorecard with interpretable features for regulatory compliance
- Implement real-time scoring with model versioning (MLflow or similar)
- Run the champion-challenger test live with proper randomization and guardrails
- Add segment-level analysis (product type, vintage, geography)

---

## Reproducibility

- All random operations seeded with `SEED=42` (from `config/settings.yaml`)
- `test_determinism.py` verifies: same seed → same model → same predictions → same metrics
- Feature engineering is pure SQL (no stochastic operations)
- `float32` precision used for features; spot-checked against original CSV values

---

## License

This project is for educational and portfolio purposes only. The underlying data is provided by American Express via Kaggle and is subject to [Kaggle competition rules](https://www.kaggle.com/competitions/amex-default-prediction/rules). Raw data is not redistributed.
