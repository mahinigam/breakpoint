# Interview Defense

A technical defense of the choices in Breakpoint. The answers are written for a risk-model reviewer, not a general audience. Every number is reproducible from `reports/metrics.json` and `reports/policy.json`.

## 1. Problem framing

**Why three actions instead of approve/decline?**

Because the economic action is not binary. A safe existing account can profitably take more line; a marginal account should stay where it is; a risky account should exit. Collapsing those into approve/decline forces one exposure and one margin for every approved account, which leaves profit on the table. The three-action policy makes the action and the exposure change explicit.

**Why not just optimize AUC?**

AUC is rank-only. The policy needs calibrated probabilities because the cutoffs are probability thresholds. A model with the same AUC but worse calibration can choose the wrong action and lose money. Calibration is therefore part of the product, not a cosmetic afterthought.

**What is the actual decision unit?**

A customer-level account review, using 13 months of statement behaviour. It is not an origination model and it is not a new-customer acquisition model.

## 2. Data

**How large is the dataset?**

458,913 labelled training customers and 200 model features: 198 numeric plus two categorical (`D_63`, `D_64`). `S_2` is the statement date; it is used only for the temporal monitoring split.

**Why is the reported default rate 1.717% when the sample is 25.89% defaults?**

The Kaggle competition kept only 5% of non-defaults. The 20× weight on non-defaults restores the population default rate. The unweighted 25.89% is a sampling artifact, not the portfolio default rate. The README, model card, and monitoring report label which rate is weighted.

**How do you prevent future leakage?**

SQL windows use only `PRECEDING` and `CURRENT ROW`; `tests/test_leakage.py` fails if `FOLLOWING` appears. The feature table is checked for one row per customer. The split is customer-level, not statement-level, so a customer cannot appear in both train and validation.

**Why DuckDB?**

The raw CSVs are large and the feature build is a windowed aggregation. DuckDB streams the conversion out of core and writes parquet without loading the full dataset into pandas. The processed table is then small enough for model training.

## 3. Model

**Why LightGBM?**

It handles missing values and categorical features natively, trains quickly on 458k rows, and is a strong tabular baseline. The logistic regression baseline is kept to quantify the complexity premium rather than to claim LightGBM is universally better.

**What are the actual hyperparameters?**

282 final trees, learning rate 0.05, 31 leaves, max depth 6, min child samples 20, 0.8 row and column subsampling, 75 early-stopping rounds, 5-fold stratified CV. There was no Optuna search and no GroupKFold in the final run.

**How did you test the AUC difference?**

Two ways. The unweighted DeLong test is reported as a supplementary check; the primary test is a class-stratified weighted bootstrap of the AUC difference, which respects the 20× negative-class design weight. The weighted bootstrap difference is **+0.0029** with a 95% interval of **[+0.0027, +0.0031]**. The old project narrative that DeLong is "invalid" was too strong; the correct statement is that the unweighted test is supplementary and the weighted bootstrap is primary.

**Did calibration change AUC?**

Barely. Isotonic regression is monotonic, so the calibrated AUC is **0.9584** versus **0.9585** raw. The gain is in Brier and ECE, not ranking.

## 4. Calibration

**Why isotonic rather than Platt?**

Platt assumes a sigmoid mapping. LightGBM's tree scores are not sigmoid-shaped. Isotonic is non-parametric and won on Brier and ECE in this dataset: Brier **0.012055** vs **0.012924** for Platt, quantile ECE **0.0075%** vs **1.3852%**.

**How is calibration cross-fitted?**

For each of the five folds, the calibrator is fit on the other folds and applied to the held-out fold. The final calibrator is then fit on all OOF rows for scoring. This avoids evaluating a calibrator on the rows used to fit it.

**Why does ECE use quantile bins?**

Quantile bins put roughly equal population weight in each bin, which is more stable for a heavily skewed PD distribution. Uniform-bin ECE is also reported in `reports/metrics.json` as a robustness check.

## 5. Policy optimization

**What exactly is optimized?**

Worst-case fractional regret across the four economic scenarios, over a joint grid of Grow and Restrict cutoffs. The grid combines log-spaced points for the low-PD region with score quantiles for the bulk. The constraints are a 60% Grow cap, a 10% Hold floor, and a 10% expected-loss increase limit.

**Why not a single cutoff?**

A single cutoff cannot express Grow/Hold/Restrict. The binary minimax baseline is still optimized properly — it is the best single cutoff under the same scenarios — so the comparison is not a straw man.

**What are the cutoffs and the action mix?**

Grow ≤ **0.000827**, Hold ≤ **0.040517**, Restrict above **0.040517**. The population-weighted mix is **60.0% Grow / 31.1% Hold / 9.0% Restrict**. Bad rates are **0.02% / 0.90% / 15.92%**.

**What does worst-case regret mean here?**

For each scenario, the oracle is the best constrained three-action policy for that scenario. Regret is `(oracle profit − policy profit) / oracle profit`. The selected policy minimizes the maximum regret across scenarios. Its worst case is **1.83%** in stress; binary minimax's worst case is **4.26%**.

**Why is the Grow cap binding?**

At a 60% cap, the optimizer chooses a Grow cutoff that puts almost exactly 60% of the population in Grow. Raising the cap to 80% lowers worst-case regret to **1.66%**, but increases exposure and distributional risk. The cap is a business choice, not a modelling artifact.

## 6. Uncertainty

**Why bootstrap rather than a t-test?**

Profit is a heavy-tailed, weighted functional of the score distribution. A paired bootstrap resamples customers with replacement, preserves the pairing and the class weight, and makes no normal approximation. The 1,000-replicate intervals are tight because the estimate uses 458,913 labelled accounts.

**What does the bootstrap show?**

The action-vs-binary advantage is positive in every replicate in every scenario. Stress: **+$102.0M**, 95% CI **[$100.6M, $103.5M]**. Conservative: **+$80.1M** **[$79.2M, $80.9M]**. Moderate: **+$169.5M** **[$167.8M, $171.0M]**. Aggressive: **+$321.4M** **[$318.2M, $324.5M]**.

**Why run a power analysis?**

Because an offline profit difference is not a test design. The simulation draws from the population-weighted score/label distribution and compares two arms. It answers: how many accounts would a champion/challenger test need? The answer is roughly **17,610 per arm** for 80% power; 25,000 gives **93.4%** and 50,000 gives **99.6%**.

**What are the limits of the confidence intervals?**

They describe sampling uncertainty in this labelled dataset. They do not cover model risk, economic-regime uncertainty, or the possibility that the future population differs from the training population.

## 7. Monitoring

**What is PSI?**

Population Stability Index: `sum((actual% − expected%) × ln(actual% / expected%))` over bins defined on the reference distribution. It measures distribution shift, not predictive performance.

**What is the monitoring split?**

The earliest 80% of accounts by statement date versus the latest 20%. It is a temporal holdout from the same labelled month, not a true out-of-time validation set. The report says so.

**Why are feature PSI and score PSI weighted differently?**

Feature PSI is unweighted because it describes the observed account-level feature distribution. Score PSI and the action mix are population-weighted with the 20× negative-class weight so they match the portfolio the policy serves. Mixing the two without labelling would be misleading.

**What is the retrain trigger?**

More than 10% of features with PSI > 0.25, or at least three delinquency/balance features above 0.25, or score PSI > 0.25, or any action share moving more than 5 percentage points. It is a review trigger, not an automatic retrain.

**What did the current report find?**

Score PSI **0.0003**, **199 green / 1 yellow** features, weighted default rate **1.7259% → 1.6810%**, action mix stable within **0.34 percentage points**, no retrain trigger. That is evidence of stability in this window, not a guarantee.

## 8. Explainability

**How are reason codes generated?**

`shap.TreeExplainer` on the saved LightGBM model. For each restricted account in the report, the top positive SHAP contributions become reason codes; the 25 examples span the full Restrict band, not only the 100%-PD tail. The SHAP values are in raw model output (log-odds/margin), not probability units.

**Why not train a proxy model on a sample?**

The old version did that. A proxy explains the proxy, not the deployed model. The current version loads the persisted model, feature contract, category levels, and calibrator.

**What are the top features?**

`P_2`, `payment_to_balance_ratio`, `P_2_rolling3_mean`, `B_4`, `D_39`, `S_3`, `D_45`, `B_3`. Delinquency, payment, and balance families account for roughly 80% of total mean absolute SHAP.

**Are these regulator-ready adverse-action reasons?**

No. The features are anonymized, and the reason codes are family-level. A production adverse-action process needs named features, business descriptions, and legal review.

## 9. Tableau and reporting

**What does the Tableau deliverable contain?**

12 aggregate CSVs plus a packaged workbook. The CSVs are threshold curve, policy markers, action mix, operating characteristics, score histogram, calibration, sensitivity, drift, KPI, policy card, bootstrap, and power. `manifest.json` records row counts and columns. All are generated from canonical reports. The committed `dashboards/breakpoint_policy.twbx` contains five worksheets and the **Breakpoint — Policy Brief** dashboard, with the CSVs embedded under `Data/`.

**Why not connect Tableau directly to the 6.2 GB parquet?**

A reviewer needs a fast, aggregate, privacy-safe workbook. The 257 KB threshold curve and the small KPI tables load instantly and contain no customer-level data. The full data stays in the reproducible pipeline.

**Is the dashboard published?**

The repository ships a publish-ready packaged workbook and a build guide. Publishing to Tableau Public requires the owner's signed-in Tableau account, so the repo does not claim a public Tableau URL. The live static reviewer demo at `https://mahinigam.github.io/breakpoint/` is the no-account fallback.

## 10. Engineering and limitations

**What does the test suite cover?**

52 tests: profit algebra and action economics, threshold grids, minimax behavior, guardrails, corrected DeLong, weighted bootstrap AUC, calibration bins/ECE, PSI and drift triggers, scoring contract, report invariants, determinism, and leakage.

**What is committed vs generated?**

The model artifacts, reports, figures, Tableau aggregates, packaged workbook, and static demo are committed so a reviewer can run the fast path. The raw data, parquet files, processed features, and OOF predictions are not committed.

**What is the biggest limitation?**

The economics are assumed, and the test labels are unavailable. The policy is robust across four labelled scenarios, but the dollar levels are illustrative. A real deployment needs Amex's actual margins, losses, and exposure, plus fair-lending and model-risk review.

**What would you do differently in a bank?**

Use named features and a regulatory-grade reason-code process; add a feature store and real-time scoring contract; run a true out-of-time validation; add protected-class fairness testing; and put the policy through model-risk validation and a shadow-mode champion/challenger test before any line change.
