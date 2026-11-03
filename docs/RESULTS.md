# Results

Generated from `reports/metrics.json`, `reports/policy.json`,
`reports/bootstrap.json`, `reports/power_analysis.json`,
`reports/drift_summary.json`, and `reports/explainability.json`.

> **Scope.** All performance and profit numbers are out-of-fold evidence on the
> labelled Amex training population. The public test labels are unavailable.
> Dollar values are expected profit under labelled assumptions, not actual P&L.

## 1. Executive summary

- **458,913** customers, **200** features (198 numeric, 2 categorical), 5-fold
  stratified out-of-fold predictions.
- Calibrated AUC **0.9584**, Amex metric
  **0.7814**, KS **0.7861**,
  Brier **0.012055**.
- Three-action policy: Grow ≤ `0.000827`, Hold ≤
  `0.040517`, Restrict above.
- Population-weighted action mix: **60.0% Grow /
  31.1% Hold /
  9.0% Restrict**.
- Worst-case regret **1.83%**, versus
  **4.26%** for binary minimax.
- Profit uplift versus binary: **+5.1%**
  to **+9.7%**.
- Expected loss **-17.8%**,
  exposure **+30.7%** (stress).
- Bootstrap action-vs-binary advantage in stress:
  **$102.0M**
  **[$100.6M,
  $103.5M]**.
- Champion/challenger: **17,610 accounts per arm**
  for 80% power.

## 2. Validation design

| Item | Value |
|---|---|
| Training rows | 458,913 |
| Features | 200 (198 numeric, 2 categorical) |
| Folds | 5 stratified |
| Final trees | 282 |
| Early stopping | 75 rounds |
| Negative-class weight | 20× |
| Population-weighted default rate | 1.717% |
| Unweighted sample default rate | 25.89% |

The weighting correction is essential. The model is trained and evaluated with
the 20× negative-class weight; the unweighted 25.89% is the subsampled training
rate, not the portfolio default rate.

## 3. Model performance

| Model | AUC | Amex | KS | Gini | Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic regression (OOF) | 0.9556 | 0.7716 | 0.7776 | 0.9112 | 0.012243 |
| LightGBM (OOF, raw) | 0.9585 | 0.7814 | 0.7863 | 0.9170 | 0.012211 |
| LightGBM (OOF, isotonic) | 0.9584 | 0.7814 | 0.7861 | 0.9168 | 0.012055 |

The unweighted DeLong test is supplementary. The primary AUC comparison is a
class-stratified weighted bootstrap: observed difference
**0.0029**,
95% CI **[0.0027,
0.0031]**, with
300 replicates.

## 4. Calibration

| Method | AUC | Brier | Log loss | ECE (quantile) | ECE (uniform) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Isotonic | 0.9584 | 0.012055 | 0.04678 | 0.0075% | 0.0171% |
| Platt | 0.9583 | 0.012924 | 0.05814 | 1.3852% | 0.2583% |
| Raw | 0.9585 | 0.012211 | 0.04712 | 0.2399% | 0.2399% |

Isotonic calibration is the selected method. It is monotonic, so AUC is
essentially unchanged; it wins on Brier and ECE. Calibration is cross-fitted:
each fold's map is fit on the other folds and applied out of fold.

## 5. Threshold heuristics and policy cutoffs

| Policy | Cutoff | Role |
| --- | ---: | ---: |
| F1-optimal | 0.242866 | Classification heuristic |
| KS-optimal | 0.017041 | Separability heuristic |
| Top 4% reject | 0.103072 | Volume heuristic |
| Fixed 0.5 | 0.500000 | Default convention |
| Binary minimax | 0.053124 | Single-cutoff robust policy |
| Action Grow | 0.000827 | Three-action policy |
| Action Restrict | 0.040517 | Three-action policy |

The F1, KS, and top-4% cutoffs are statistical heuristics. They are not
profit-optimal, and their regret varies sharply by scenario. The binary minimax
cutoff is the strongest single-cutoff baseline. The action policy uses two
cutoffs because it chooses among three actions.

## 6. Three-action policy

| Action | Margin multiplier | Exposure multiplier | Share | Bad rate |
| --- | ---: | ---: | ---: | ---: |
| Grow | 110% | 150% | 60.0% | 0.02% |
| Hold | 100% | 100% | 31.1% | 0.90% |
| Restrict | 0% | 0% | 9.0% | 15.92% |

Guardrails: Grow ≤ **60%**,
Hold ≥ **10%**, expected-loss
increase ≤ **10%**.
The selected policy binds the Grow cap exactly.

## 7. Scenario comparison

| Scenario | Binary | 3-action | Oracle | Uplift | Binary regret | Action regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Conservative | $1,176.6M | $1,256.7M | $1,258.5M | +6.8% | 6.5% | 0.15% |
| Moderate | $3,116.6M | $3,286.1M | $3,333.0M | +5.4% | 6.5% | 1.41% |
| Aggressive | $6,333.4M | $6,654.7M | $6,774.6M | +5.1% | 6.5% | 1.77% |
| Stress | $1,051.5M | $1,153.5M | $1,175.1M | +9.7% | 10.5% | 1.83% |

The binary policy is the best single cutoff across the four scenarios; the
three-action policy is the best constrained action rule. The action oracle is
the best constrained three-action rule **per scenario**, so it is an upper bound,
not a deployable policy.

## 8. Regret, guardrails, and partial restrict

| Grow cap | Grow cutoff | Restrict cutoff | Worst-case regret |
| --- | ---: | ---: | ---: |
| 50% | 0.000412 | 0.040517 | 1.87% |
| 60% | 0.000827 | 0.040517 | 1.83% |
| 70% | 0.002463 | 0.040353 | 1.70% |
| 80% | 0.006632 | 0.039206 | 1.66% |

A 50% Grow cap costs **1.87%**
worst-case regret; 80% costs **1.66%**.
The cap is a business choice, not a modelling artifact.

A partial-restrict alternative (50% line reduction instead of full exit) has
worst-case regret **2.34%**, worse than the
full-exit policy's **1.83%**.

Continuous cost-ratio sensitivity: the action policy's worst-case regret is
**0.38%** at cost ratio
**1.0**; the binary
minimax sweep's worst case is **5.29%**
at **50.0**.

## 9. Paired bootstrap

| Scenario | Action − binary | 95% CI | p-value |
| --- | ---: | ---: | ---: |
| Conservative | $80.1M | [$79.2M, $80.9M] | <0.001 |
| Moderate | $169.5M | [$167.8M, $171.0M] | <0.001 |
| Aggressive | $321.4M | [$318.2M, $324.5M] | <0.001 |
| Stress | $102.0M | [$100.6M, $103.5M] | <0.001 |

The bootstrap resamples customers with replacement, preserving the 20×
population weight and the pairing between policies. The action-vs-binary
advantage is positive in every replicate in every scenario. Confidence
intervals describe sampling uncertainty in this labelled dataset, not
model-risk or future-regime uncertainty.

## 10. Power analysis

| Accounts / arm | Power | Mean Δ ($/account) | SD Δ ($/account) |
| --- | ---: | ---: | ---: |
| 1,000 | 16.0% | 15.51 | 23.42 |
| 2,500 | 25.8% | 13.91 | 14.93 |
| 5,000 | 44.0% | 15.40 | 10.45 |
| 10,000 | 66.2% | 15.49 | 7.56 |
| 25,000 | 93.4% | 14.98 | 4.80 |
| 50,000 | 99.6% | 14.39 | 3.49 |

Required sample size for 80% power: **17,610
accounts per arm**. This is a two-arm simulation from the population-weighted
score/label distribution, not a live test.

## 11. Monitoring

| Monitoring item | Value |
| --- | ---: |
| Reference rows | 368,710 |
| Current rows | 90,203 |
| Split | Earliest 80% of accounts by statement date vs latest 20% (cutoff 2018-03-25) |
| Weighted default rate | 1.7259% → 1.6810% |
| Unweighted default rate | 25.99% → 25.48% |
| Score PSI | 0.0003 |
| Status counts | {'green': 199, 'yellow': 1} |
| Retrain review | not triggered |

Feature PSI is unweighted; score PSI and action mix are population-weighted. The
current split is a temporal holdout from the same labelled month, not true
out-of-time validation. The retrain trigger is a review trigger, not an
automatic retrain.

## 12. Explainability

Top features by mean absolute SHAP:

| Feature | Family | Mean |SHAP| | Share |
| --- | ---: | ---: | ---: |
| P_2 | Payment | 1.096 | 19.7% |
| payment_to_balance_ratio | Engineered/Other | 0.371 | 6.7% |
| P_2_rolling3_mean | Payment | 0.232 | 4.2% |
| B_4 | Balance | 0.171 | 3.1% |
| D_39 | Delinquency | 0.165 | 3.0% |
| S_3 | Spend | 0.153 | 2.8% |
| D_45 | Delinquency | 0.120 | 2.2% |
| B_3 | Balance | 0.115 | 2.1% |
| D_43 | Delinquency | 0.115 | 2.1% |
| D_50 | Delinquency | 0.102 | 1.8% |

Family contribution:

| Family | Mean |SHAP| | Share |
| --- | ---: | ---: |
| Delinquency (D) | 1.926 | 34.7% |
| Payment (P) | 1.441 | 26.0% |
| Balance (B) | 1.069 | 19.3% |
| Spend (S) | 0.438 | 7.9% |
| Engineered / Other | 0.371 | 6.7% |
| Risk (R) | 0.306 | 5.5% |

SHAP is computed on the saved LightGBM model with `TreeExplainer` and 5,000
customers. Values are in raw model output (log-odds/margin). Reason codes are
family-level because the Amex features are anonymized. The 25 examples in
`reports/reason_codes.csv` span the full Restrict band, not only the highest-PD
tail, so the table is representative of the accounts the policy actually exits.

## 13. Reproducibility

```bash
make pipeline   # full retrain from data/raw
make tableau    # regenerate Tableau aggregates
make demo       # rebuild docs/index.html
make verify     # tests + docs check + demo rebuild
```

The committed artifacts and reports make the fast path work without the 6.2 GB
parquet. The full pipeline needs the Kaggle CSVs in `data/raw/`.

## 14. Caveats

1. **No test labels.** The public test set cannot support performance claims.
2. **Assumed economics.** Margin, LGD, and exposure are labelled scenarios.
3. **No out-of-time validation.** Monitoring is a temporal holdout, not next-year drift.
4. **Anonymized features.** No fair-lending or adverse-action claim.
5. **No production system.** No feature store, latency test, or model-risk sign-off.
6. **SHAP is not causal.** It attributes model output, not intervention effects.
7. **Binary minimax is a strong baseline.** The comparison is not a straw man.

See [`MODEL_CARD.md`](MODEL_CARD.md), [`DECISION_MEMO.md`](DECISION_MEMO.md),
and [`TABLEAU_DASHBOARD_GUIDE.md`](TABLEAU_DASHBOARD_GUIDE.md).
