# Decision Memo: A Three-Action Credit-Line Policy

**To:** Chief Risk Officer / Credit Policy Committee (illustrative)<br>
**From:** Risk Analytics<br>
**Date:** 3 November 2026<br>
**Subject:** Replace the binary minimax cutoff with a constrained Grow / Hold / Restrict policy<br>
**Status:** Portfolio project; not an actual American Express communication

---

## Recommendation

Adopt a **three-action policy** for existing-account credit-line reviews:

- **Grow** accounts with calibrated PD ≤ **0.000827**: increase the line 50%; margin scales 10%.
- **Hold** accounts with PD ≤ **0.040517**: keep the current line.
- **Restrict** accounts above **0.040517**: exit the line with zero margin and zero exposure.

The policy is optimized on calibrated out-of-fold probabilities and constrained to a 60% Grow cap, a 10% Hold floor, and no more than a 10% increase in expected loss. Its worst-case regret is **1.83%**, compared with **4.26%** for the best binary minimax cutoff.

## The problem with a binary cutoff

A binary approve/decline threshold treats three economically different actions as one. It cannot grow a safe account, hold a marginal account at the current line, and exit a risky account at different exposure levels. It also hides the portfolio trade-off between profit and expected loss.

The binary minimax cutoff is **0.053124**. It is a strong baseline: it minimizes worst-case regret across the four scenarios. But it still leaves **6.5%–10.5%** of the action-oracle profit on the table because it cannot vary the action by account.

## Evidence

<!-- BEGIN GENERATED: decision_evidence -->
| Scenario | Binary profit | 3-action profit | Uplift | Binary regret | Action regret | EL change | Exposure change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Conservative | $1,176.6M | $1,256.7M | +6.8% | 6.5% | 0.15% | -17.8% | +30.7% |
| Moderate | $3,116.6M | $3,286.1M | +5.4% | 6.5% | 1.41% | -17.8% | +30.7% |
| Aggressive | $6,333.4M | $6,654.7M | +5.1% | 6.5% | 1.77% | -17.8% | +30.7% |
| Stress | $1,051.5M | $1,153.5M | +9.7% | 10.5% | 1.83% | -17.8% | +30.7% |

Paired bootstrap, stress: **$102.0M** 95% CI **[$100.6M, $103.5M]**, p < 0.001, 1,000 replicates.
<!-- END GENERATED: decision_evidence -->

The three-action policy improves expected profit in every scenario, from **+5.1%** in the aggressive case to **+9.7%** in stress. It reduces expected loss by **17.8%** while increasing exposure by **30.7%**; the guardrail is satisfied in every scenario.

The paired bootstrap resamples customers 1,000 times with the 20× population weight. The action-minus-binary advantage is positive in every replicate. In the stress scenario the advantage is **+$102.0M** with a 95% interval of **[$100.6M, $103.5M]**.

## Guardrails are a business choice

The 60% Grow cap is not a statistical artifact. It bounds how much of the portfolio can receive a line increase. The sensitivity is:

| Grow cap | Grow cutoff | Restrict cutoff | Worst-case regret |
|---:|---:|---:|---:|
| 50% | 0.000412 | 0.040517 | 1.87% |
| 60% | 0.000827 | 0.040517 | 1.83% |
| 70% | 0.002463 | 0.040353 | 1.70% |
| 80% | 0.006632 | 0.039206 | 1.66% |

A higher cap buys lower regret but increases exposure and distributional risk. The recommendation is the 60% cap because it is defensible without knowing the institution's risk appetite. The committee should choose the cap explicitly.

## What this memo does not claim

- **Not actual Amex P&L.** Margin, loss given default, and exposure are labelled scenarios, not observed financials.
- **Not an out-of-time result.** The test labels are unavailable. All evidence is out-of-fold on the labelled training population.
- **Not a fair-lending review.** Features are anonymized; protected-class testing and adverse-action review are outstanding.
- **Not a production deployment.** There is no feature store, latency test, model-risk sign-off, or live monitoring integration.
- **Not causal.** SHAP and the policy describe model behaviour and expected economics, not the causal effect of changing a customer's line.

## Implementation plan

1. **Shadow mode (4 weeks).** Score the full eligible population; compare the three-action policy with the current binary policy. Do not change customer lines.
2. **Champion/challenger design.** A two-arm test has 80% power at roughly **17,610 accounts per arm**; 25,000 per arm reaches **93.4%**, and 50,000 reaches **99.6%**. Randomize at the account level; pre-register the primary metric (expected profit) and guardrails (expected loss, exposure, complaint rate).
3. **Policy review.** Confirm the margin, loss, and exposure assumptions with Finance and Credit Policy. Re-run the optimizer with the approved values.
4. **Fairness and adverse-action review.** Map anonymized features to business definitions; run disparate-impact and reason-code reviews before any Restrict action.
5. **Monitoring.** Track feature PSI, score PSI, action mix, weighted default rate, and the explicit retrain trigger. The current temporal holdout shows no trigger, but that is not a guarantee.
6. **Tableau reporting.** Use the packaged aggregate workbook at `dashboards/breakpoint_policy.twbx`; publish it only from a signed-in Tableau account. Do not expose customer-level data.

## Decision requested

Approve **shadow-mode evaluation** of the three-action policy at the 60% Grow cap, with the 17,610-per-arm champion/challenger design and the guardrails above. Do not deploy line changes until the shadow results, economics sign-off, and fair-lending review are complete.

---

*Prepared from `reports/metrics.json`, `reports/policy.json`, `reports/bootstrap.json`, and `reports/power_analysis.json`. Every generated table in this memo is reproduced by `scripts/render_docs.py`; `scripts/verify_docs.py` fails if the memo drifts from the reports.*
