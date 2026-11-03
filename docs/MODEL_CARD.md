# Model Card: Breakpoint Credit-Risk Model

## Model details

| Item | Value |
|---|---|
| Model | LightGBM gradient-boosted trees |
| Task | Probability of default within the Amex 18-month label window |
| Training rows | 458,913 customers |
| Features | 200: 198 numeric, 2 categorical (`D_63`, `D_64`) |
| Validation | 5-fold stratified out-of-fold predictions |
| Final trees | 282 |
| Learning rate | 0.05 |
| Leaves / depth | 31 leaves / max depth 6 |
| Min child samples | 20 |
| Subsampling | 0.8 rows, 0.8 columns |
| Early stopping | 75 rounds |
| Class weighting | 20× on non-defaults |
| Calibration | Cross-fitted isotonic regression |
| Artifacts | `artifacts/lgbm_model.joblib`, `calibrator.joblib`, `feature_list.json`, `categorical_levels.json` |

The model is a decision-support model for **existing customers** with observed statement behaviour. It is not an origination scorecard, and it is not trained on application or bureau data.

## Intended use

**Primary use.** Rank existing accounts by probability of default and feed a constrained three-action policy: Grow, Hold, or Restrict. The policy is designed for a credit-line review, where the economic question is exposure and margin, not a binary approve/decline.

**Out of scope.**

- Origination or new-customer acquisition.
- Adverse-action notices without a named-feature reason-code review.
- Any jurisdiction or product where the Amex feature definitions, label window, or economic assumptions do not apply.
- Fully automated line changes without human review, model-risk sign-off, and fair-lending testing.
- Causal claims about changing a feature or a customer's behaviour.

## Training data

The American Express/Kaggle default prediction dataset contains 458,913 training customers and up to 13 monthly statements per customer. The public test labels are not available.

The competition retained only 5% of non-defaults. The observed training default rate is **25.89%**; the population-weighted rate is **1.717%**. Every calibration metric, profit calculation, score-drift statistic, and action mix applies a **20× weight to non-defaults** to correct the sampling. Feature PSI is reported unweighted because it describes the observed account-level feature distribution; that choice is labelled in the monitoring report.

`S_2` is the statement date. It is used for the temporal monitoring split but is **not** a model feature. The model features are the persisted 200-column contract in `artifacts/feature_list.json`.

## Evaluation design

- **Out-of-fold predictions.** Each customer is scored by a model that did not see that customer in training.
- **Calibration.** Each fold's isotonic map is fit on the other folds and applied out of fold. The final map is fit on all OOF rows for scoring.
- **No test-label claims.** The public test set has no labels. All reported performance is on the labelled training population.
- **No leakage test.** `tests/test_leakage.py` fails if any SQL window uses `FOLLOWING`; `test_leakage.py` also checks one row per customer in the processed feature table.
- **Determinism.** Seeds are fixed; `tests/test_determinism.py` checks LightGBM reproducibility.

## Metrics

<!-- BEGIN GENERATED: model_card_metrics -->
### Predictive performance

| Model | AUC | Amex | KS | Gini | Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic regression (OOF) | 0.9556 | 0.7716 | 0.7776 | 0.9112 | 0.012243 |
| LightGBM (OOF, raw) | 0.9585 | 0.7814 | 0.7863 | 0.9170 | 0.012211 |
| LightGBM (OOF, isotonic) | 0.9584 | 0.7814 | 0.7861 | 0.9168 | 0.012055 |

### Calibration

| Method | AUC | Brier | Log loss | ECE (quantile) | ECE (uniform) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Isotonic | 0.9584 | 0.012055 | 0.04678 | 0.0075% | 0.0171% |
| Platt | 0.9583 | 0.012924 | 0.05814 | 1.3852% | 0.2583% |
| Raw | 0.9585 | 0.012211 | 0.04712 | 0.2399% | 0.2399% |

Final model: **282 trees**, **200 features** (198 numeric, 2 categorical), **5-fold** OOF, **20×** negative-class weight. Population-weighted default rate: **1.717%** (unweighted sample: 25.89%).

Decision policy: Grow ≤ `0.000827`, Restrict > `0.040517`; worst-case regret **1.83%** vs **4.26%** for binary minimax.
<!-- END GENERATED: model_card_metrics -->

The raw LightGBM and calibrated LightGBM have nearly identical AUC because isotonic regression is monotonic. The calibrated model is the one used for policy and scoring because the cutoffs are probability thresholds, not rank thresholds.

## Decision policy

The policy is fitted on calibrated OOF probabilities and evaluated across four economic scenarios:

| Scenario | Margin | Loss given default | Exposure | Cost ratio |
|---|---:|---:|---:|---:|
| Conservative | $200 | 80% | $5,000 | 20.0 |
| Moderate | $500 | 60% | $5,000 | 6.0 |
| Aggressive | $1,000 | 40% | $5,000 | 2.0 |
| Stress | $200 | 90% | $10,000 | 45.0 |

Grow raises margin 10% and exposure 50%; Hold is baseline; Restrict exits the line with zero margin and zero exposure. The optimizer minimizes worst-case fractional regret subject to `max_grow_share = 0.60`, `min_hold_share = 0.10`, and `max_expected_loss_increase = 0.10`.

The selected policy uses **Grow ≤ 0.000827** and **Restrict > 0.040517**, producing a population-weighted mix of **60.0% Grow / 31.1% Hold / 9.0% Restrict**. Worst-case regret is **1.83%**, versus **4.26%** for binary minimax. Action-vs-binary profit is positive in every scenario and every one of 1,000 paired bootstrap replicates.

The policy is a **portfolio** decision. An individual account at the Grow cutoff has a low absolute PD; the action also changes exposure, so the expected-loss and exposure totals are reported alongside the action mix.

## Monitoring

The monitoring report compares the earliest 80% of accounts by statement date with the latest 20%. It reports:

- Feature PSI for all 200 model features, with missingness deltas.
- Population-weighted score PSI and action-mix drift.
- Weighted default rate in each window.
- A structured retrain-review decision with explicit reasons.

The current report shows score PSI **0.0003**, **199 green / 1 yellow** features, and no retrain trigger. This is a temporal holdout from the labelled month, not a true out-of-time validation set.

## Fairness and ethical considerations

The Amex features are anonymized. It is not possible to verify whether a feature proxies a protected class, and this project does **not** claim a fair-lending or disparate-impact review. A production deployment would require:

- Named features and a legal/regulatory mapping.
- Protected-class data and disparate-impact testing.
- Reason codes that satisfy adverse-action requirements.
- Human review for Restrict decisions.
- Model-risk documentation, validation, and ongoing monitoring.

The economic policy is not a fairness policy. Optimizing profit can conflict with access and distributional goals; those trade-offs must be made explicitly, not hidden inside a cutoff.

## Limitations

1. **Offline only.** No live P&L, no production latency test, no feature store.
2. **Assumed economics.** The dollar levels are labelled scenarios, not Amex's actual margins or losses.
3. **No test labels.** The public test set cannot support performance claims.
4. **Temporal monitoring is limited.** The latest 20% of one labelled month is not next-year drift.
5. **Anonymized explanations.** Reason codes are family-level, not feature-level business explanations.
6. **No causal interpretation.** SHAP attributes model output, not intervention effects.
7. **Binary minimax is a strong baseline.** The three-action policy is compared with the best binary cutoff, not a straw man.

## Maintenance

- Re-run `make pipeline` when new labelled data is available.
- Review `reports/drift_summary.json` and the retrain trigger each monitoring cycle.
- Re-fit the policy when margin, loss, or exposure assumptions change materially.
- Re-run `make verify` before publishing any change.
- Treat a policy cutoff change as a model change requiring review, not a dashboard tweak.
