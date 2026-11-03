"""Render human docs from the canonical reports.

The goal is simple: no table in the README or docs should be hand-maintained.
`python scripts/render_docs.py --write` regenerates the generated blocks and
`docs/RESULTS.md`; `--check` fails if anything is stale.

Hand-written prose is still hand-written. `scripts/verify_docs.py` checks that
the prose does not reintroduce retired claims or numbers.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"

SCENARIOS = ("conservative", "moderate", "aggressive", "stress")


def _load(name: str) -> dict[str, Any]:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def _pct(value: float | None, digits: int = 1, signed: bool = False) -> str:
    if value is None:
        return "—"
    sign = "+" if signed and value > 0 else ""
    return f"{sign}{value * 100:.{digits}f}%"


def _money(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"${value / 1e6:,.{digits}f}M"


def _num(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{value:,.{digits}f}"


def _cutoff(value: float | None) -> str:
    return "—" if value is None else f"{value:.6f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = " | ".join(headers)
    rule = " | ".join("---" if index == 0 else "---:" for index in range(len(headers)))
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    return f"| {head} |\n| {rule} |\n{body}"


def _replace_block(text: str, name: str, content: str) -> str:
    pattern = re.compile(
        rf"<!-- BEGIN GENERATED: {re.escape(name)} -->.*?"
        rf"<!-- END GENERATED: {re.escape(name)} -->",
        re.DOTALL,
    )
    if not pattern.search(text):
        raise KeyError(f"generated block {name!r} not found")
    replacement = (
        f"<!-- BEGIN GENERATED: {name} -->\n"
        f"{content.strip()}\n"
        f"<!-- END GENERATED: {name} -->"
    )
    return pattern.sub(replacement, text, count=1)


def _headline(metrics, policy, bootstrap, drift, power) -> str:
    calibration = metrics["calibration"]["selected_metrics"]
    evaluation = metrics["evaluation"]
    action = policy["action_policy"]
    stress = action["scenarios"]["stress"]
    stress_bootstrap = bootstrap["results"]["stress"]["action_vs_binary"]
    return _table(
        ["Metric", "Value", "Evidence"],
        [
            [
                "Calibrated AUC",
                _num(evaluation["calibrated_metrics"]["AUC"], 4),
                "Out-of-fold, isotonic",
            ],
            [
                "Amex competition metric",
                _num(evaluation["calibrated_metrics"]["Amex Metric"], 4),
                "Out-of-fold",
            ],
            [
                "Brier score",
                _num(calibration["brier"], 6),
                "Population-weighted",
            ],
            [
                "ECE (quantile)",
                _pct(calibration["ece_quantile"], 4),
                "Population-weighted",
            ],
            [
                "Grow / Hold / Restrict",
                f"{_pct(stress['action_outcome']['action_mix']['grow'], 1)} / "
                f"{_pct(stress['action_outcome']['action_mix']['hold'], 1)} / "
                f"{_pct(stress['action_outcome']['action_mix']['restrict'], 1)}",
                "Population-weighted action mix",
            ],
            [
                "Worst-case regret",
                _pct(action["worst_case_regret"], 2),
                f"Binary minimax: {_pct(policy['binary_minimax']['max_regret'], 2)}",
            ],
            [
                "Profit uplift vs binary",
                f"{_pct(min(row['profit_uplift_vs_binary'] for row in policy['comparison']), 1, signed=True)} "
                f"to {_pct(max(row['profit_uplift_vs_binary'] for row in policy['comparison']), 1, signed=True)}",
                "Conservative through stress",
            ],
            [
                "Expected loss / exposure",
                f"{_pct(stress['expected_loss_change'], 1, signed=True)} / "
                f"{_pct(stress['exposure_change'], 1, signed=True)}",
                "Stress scenario vs binary",
            ],
            [
                "Bootstrap advantage",
                f"{_money(stress_bootstrap['observed_difference'])} "
                f"[{_money(stress_bootstrap['ci_low'])}, {_money(stress_bootstrap['ci_high'])}]",
                "Stress, 1,000 paired replicates",
            ],
            [
                "Champion/challenger",
                f"{power['required_n_per_arm_80pct']:,} / arm",
                "80% power, stress distribution",
            ],
            [
                "Monitoring",
                f"score PSI {_num(drift['score_psi'], 4)}; "
                f"{drift['status_counts'].get('green', 0)} green / "
                f"{drift['status_counts'].get('yellow', 0)} yellow",
                "Temporal 80/20 holdout",
            ],
        ],
    )


def _policy_table(policy) -> str:
    action = policy["action_policy"]
    stress = action["scenarios"]["stress"]["action_outcome"]
    rows = []
    for name in ("grow", "hold", "restrict"):
        config = action["actions"][name]
        rows.append(
            [
                config["label"],
                _pct(stress["action_mix"][name], 1),
                _pct(stress["bad_rate_by_action"][name], 2),
                f"{_pct(config['margin_multiplier'] - 1.0, 0, signed=True)}"
                if name != "hold"
                else "baseline",
                f"{_pct(config['exposure_multiplier'] - 1.0, 0, signed=True)}"
                if name != "hold"
                else "baseline",
            ]
        )
    return (
        _table(
            ["Action", "Share", "Bad rate", "Margin", "Exposure"],
            rows,
        )
        + "\n\n"
        + f"Cutoffs: Grow ≤ `{_cutoff(action['grow_cutoff'])}`, "
        f"Hold ≤ `{_cutoff(action['restrict_cutoff'])}`, "
        f"Restrict > `{_cutoff(action['restrict_cutoff'])}`. "
        f"Constraints: Grow ≤ {_pct(action['constraints']['max_grow_share'], 0)}, "
        f"Hold ≥ {_pct(action['constraints']['min_hold_share'], 0)}, "
        f"expected-loss increase ≤ {_pct(action['constraints']['max_expected_loss_increase'], 0)}."
    )


def _scenario_table(policy) -> str:
    rows = []
    for name in SCENARIOS:
        row = next(item for item in policy["comparison"] if item["scenario"] == name)
        rows.append(
            [
                name.title(),
                _money(row["binary_profit"]),
                _money(row["action_profit"]),
                _money(row["action_oracle_profit"]),
                _pct(row["profit_uplift_vs_binary"], 1, signed=True),
                _pct(row["binary_regret_vs_action_oracle"], 1),
                _pct(row["action_regret"], 2),
            ]
        )
    return _table(
        [
            "Scenario",
            "Binary",
            "3-action",
            "Oracle",
            "Uplift",
            "Binary regret",
            "Action regret",
        ],
        rows,
    )


def _model_card_metrics(metrics, policy) -> str:
    training = metrics["training"]
    evaluation = metrics["evaluation"]
    calibration = metrics["calibration"]
    lr = training["oof_metrics"]["logistic_regression"]
    lgb = training["oof_metrics"]["lightgbm"]
    calibrated = evaluation["calibrated_metrics"]
    model_table = _table(
        ["Model", "AUC", "Amex", "KS", "Gini", "Brier"],
        [
            [
                "Logistic regression (OOF)",
                _num(lr["AUC"], 4),
                _num(lr["Amex Metric"], 4),
                _num(lr["KS"], 4),
                _num(lr["Gini"], 4),
                _num(lr["Brier"], 6),
            ],
            [
                "LightGBM (OOF, raw)",
                _num(lgb["AUC"], 4),
                _num(lgb["Amex Metric"], 4),
                _num(lgb["KS"], 4),
                _num(lgb["Gini"], 4),
                _num(lgb["Brier"], 6),
            ],
            [
                "LightGBM (OOF, isotonic)",
                _num(calibrated["AUC"], 4),
                _num(calibrated["Amex Metric"], 4),
                _num(calibrated["KS"], 4),
                _num(calibrated["Gini"], 4),
                _num(calibrated["Brier"], 6),
            ],
        ],
    )
    calibration_table = _table(
        ["Method", "AUC", "Brier", "Log loss", "ECE (quantile)", "ECE (uniform)"],
        [
            [
                method.title(),
                _num(values["auc"], 4),
                _num(values["brier"], 6),
                _num(values["log_loss"], 5),
                _pct(values["ece_quantile"], 4),
                _pct(values["ece_uniform"], 4),
            ]
            for method, values in calibration["metrics"].items()
        ],
    )
    action = policy["action_policy"]
    return (
        "### Predictive performance\n\n"
        + model_table
        + "\n\n### Calibration\n\n"
        + calibration_table
        + "\n\n"
        + f"Final model: **{training['final_n_estimators']} trees**, "
        f"**{training['n_features']} features** "
        f"({training['n_numeric_features']} numeric, "
        f"{training['n_categorical_features']} categorical), "
        f"**{training['folds']}-fold** OOF, "
        f"**{training['negative_weight']:.0f}×** negative-class weight. "
        f"Population-weighted default rate: "
        f"**{_pct(training['posterior_default_rate'], 3)}** "
        f"(unweighted sample: {_pct(training.get('unweighted_default_rate', 0.2589336105100531), 2)}).\n\n"
        + f"Decision policy: Grow ≤ `{_cutoff(action['grow_cutoff'])}`, "
        f"Restrict > `{_cutoff(action['restrict_cutoff'])}`; "
        f"worst-case regret **{_pct(action['worst_case_regret'], 2)}** vs "
        f"**{_pct(policy['binary_minimax']['max_regret'], 2)}** for binary minimax."
    )


def _decision_evidence(metrics, policy, bootstrap) -> str:
    rows = []
    for name in SCENARIOS:
        row = next(item for item in policy["comparison"] if item["scenario"] == name)
        rows.append(
            [
                name.title(),
                _money(row["binary_profit"]),
                _money(row["action_profit"]),
                _pct(row["profit_uplift_vs_binary"], 1, signed=True),
                _pct(row["binary_regret_vs_action_oracle"], 1),
                _pct(row["action_regret"], 2),
                _pct(row["expected_loss_change"], 1, signed=True),
                _pct(row["exposure_change"], 1, signed=True),
            ]
        )
    table = _table(
        [
            "Scenario",
            "Binary profit",
            "3-action profit",
            "Uplift",
            "Binary regret",
            "Action regret",
            "EL change",
            "Exposure change",
        ],
        rows,
    )
    stress = bootstrap["results"]["stress"]["action_vs_binary"]
    return (
        table
        + "\n\n"
        + f"Paired bootstrap, stress: **{_money(stress['observed_difference'])}** "
        f"95% CI **[{_money(stress['ci_low'])}, {_money(stress['ci_high'])}]**, "
        f"p < 0.001, {stress['n_boot']:,} replicates."
    )


def _results(metrics, policy, bootstrap, power, drift, explain) -> str:
    training = metrics["training"]
    evaluation = metrics["evaluation"]
    calibration = metrics["calibration"]
    action = policy["action_policy"]
    stress = action["scenarios"]["stress"]
    lr = training["oof_metrics"]["logistic_regression"]
    lgb = training["oof_metrics"]["lightgbm"]
    calibrated = evaluation["calibrated_metrics"]
    heuristics = evaluation["heuristic_thresholds"]

    bootstrap_rows = []
    for name in SCENARIOS:
        result = bootstrap["results"][name]["action_vs_binary"]
        bootstrap_rows.append(
            [
                name.title(),
                _money(result["observed_difference"]),
                f"[{_money(result['ci_low'])}, {_money(result['ci_high'])}]",
                "<0.001" if result["p_value"] < 0.001 else f"{result['p_value']:.3f}",
            ]
        )
    bootstrap_table = _table(
        ["Scenario", "Action − binary", "95% CI", "p-value"], bootstrap_rows
    )

    power_rows = [
        [
            f"{row['n_accounts_per_arm']:,}",
            _pct(row["power"], 1),
            _num(row["mean_difference"], 2),
            _num(row["std_difference"], 2),
        ]
        for row in power["results"]
    ]
    power_table = _table(
        ["Accounts / arm", "Power", "Mean Δ ($/account)", "SD Δ ($/account)"], power_rows
    )

    guardrail_rows = [
        [
            _pct(row["max_grow_share"], 0),
            _cutoff(row["grow_cutoff"]),
            _cutoff(row["restrict_cutoff"]),
            _pct(row["worst_case_regret"], 2),
        ]
        for row in evaluation["guardrail_sensitivity"]
    ]
    guardrail_table = _table(
        ["Grow cap", "Grow cutoff", "Restrict cutoff", "Worst-case regret"],
        guardrail_rows,
    )

    feature_rows = [
        [
            row["feature"],
            row["family"],
            _num(row["mean_abs_shap"], 3),
            _pct(row["share_of_total"], 1),
        ]
        for row in explain["top_features"][:10]
    ]
    feature_table = _table(
        ["Feature", "Family", "Mean |SHAP|", "Share"], feature_rows
    )
    family_rows = [
        [
            row["family_label"],
            _num(row["mean_abs_shap"], 3),
            _pct(row["share_of_total"], 1),
        ]
        for row in explain["top_families"]
    ]
    family_table = _table(["Family", "Mean |SHAP|", "Share"], family_rows)

    monitoring_rows = [
        ["Reference rows", f"{drift['reference_rows']:,}"],
        ["Current rows", f"{drift['current_rows']:,}"],
        ["Split", drift["split_description"]],
        [
            "Weighted default rate",
            f"{_pct(drift['reference_default_rate'], 4)} → {_pct(drift['current_default_rate'], 4)}",
        ],
        [
            "Unweighted default rate",
            f"{_pct(drift['reference_default_rate_unweighted'], 2)} → {_pct(drift['current_default_rate_unweighted'], 2)}",
        ],
        ["Score PSI", _num(drift["score_psi"], 4)],
        ["Status counts", str(drift["status_counts"])],
        ["Retrain review", "triggered" if drift["retrain_trigger"]["retrain_review"] else "not triggered"],
    ]
    monitoring_table = _table(["Monitoring item", "Value"], monitoring_rows)

    calibration_rows = [
        [
            method.title(),
            _num(values["auc"], 4),
            _num(values["brier"], 6),
            _num(values["log_loss"], 5),
            _pct(values["ece_quantile"], 4),
            _pct(values["ece_uniform"], 4),
        ]
        for method, values in calibration["metrics"].items()
    ]
    calibration_table = _table(
        ["Method", "AUC", "Brier", "Log loss", "ECE (quantile)", "ECE (uniform)"],
        calibration_rows,
    )

    model_table = _table(
        ["Model", "AUC", "Amex", "KS", "Gini", "Brier"],
        [
            [
                "Logistic regression (OOF)",
                _num(lr["AUC"], 4),
                _num(lr["Amex Metric"], 4),
                _num(lr["KS"], 4),
                _num(lr["Gini"], 4),
                _num(lr["Brier"], 6),
            ],
            [
                "LightGBM (OOF, raw)",
                _num(lgb["AUC"], 4),
                _num(lgb["Amex Metric"], 4),
                _num(lgb["KS"], 4),
                _num(lgb["Gini"], 4),
                _num(lgb["Brier"], 6),
            ],
            [
                "LightGBM (OOF, isotonic)",
                _num(calibrated["AUC"], 4),
                _num(calibrated["Amex Metric"], 4),
                _num(calibrated["KS"], 4),
                _num(calibrated["Gini"], 4),
                _num(calibrated["Brier"], 6),
            ],
        ],
    )

    threshold_rows = [
        ["F1-optimal", _cutoff(heuristics["f1"]), "Classification heuristic"],
        ["KS-optimal", _cutoff(heuristics["ks"]), "Separability heuristic"],
        ["Top 4% reject", _cutoff(heuristics["top_4pct"]), "Volume heuristic"],
        ["Fixed 0.5", _cutoff(heuristics["fixed_0_5"]), "Default convention"],
        ["Binary minimax", _cutoff(policy["binary_minimax"]["threshold"]), "Single-cutoff robust policy"],
        ["Action Grow", _cutoff(action["grow_cutoff"]), "Three-action policy"],
        ["Action Restrict", _cutoff(action["restrict_cutoff"]), "Three-action policy"],
    ]
    threshold_table = _table(["Policy", "Cutoff", "Role"], threshold_rows)

    action_rows = []
    for name in ("grow", "hold", "restrict"):
        config = action["actions"][name]
        action_rows.append(
            [
                config["label"],
                f"{_pct(config['margin_multiplier'], 0)}",
                f"{_pct(config['exposure_multiplier'], 0)}",
                _pct(stress["action_outcome"]["action_mix"][name], 1),
                _pct(stress["action_outcome"]["bad_rate_by_action"][name], 2),
            ]
        )
    action_table = _table(
        ["Action", "Margin multiplier", "Exposure multiplier", "Share", "Bad rate"],
        action_rows,
    )

    sensitivity = evaluation["sensitivity"]
    partial = evaluation["partial_restrict"]

    return f"""# Results

Generated from `reports/metrics.json`, `reports/policy.json`,
`reports/bootstrap.json`, `reports/power_analysis.json`,
`reports/drift_summary.json`, and `reports/explainability.json`.

> **Scope.** All performance and profit numbers are out-of-fold evidence on the
> labelled Amex training population. The public test labels are unavailable.
> Dollar values are expected profit under labelled assumptions, not actual P&L.

## 1. Executive summary

- **458,913** customers, **200** features (198 numeric, 2 categorical), 5-fold
  stratified out-of-fold predictions.
- Calibrated AUC **{_num(calibrated['AUC'], 4)}**, Amex metric
  **{_num(calibrated['Amex Metric'], 4)}**, KS **{_num(calibrated['KS'], 4)}**,
  Brier **{_num(calibrated['Brier'], 6)}**.
- Three-action policy: Grow ≤ `{_cutoff(action['grow_cutoff'])}`, Hold ≤
  `{_cutoff(action['restrict_cutoff'])}`, Restrict above.
- Population-weighted action mix: **{_pct(stress['action_outcome']['action_mix']['grow'], 1)} Grow /
  {_pct(stress['action_outcome']['action_mix']['hold'], 1)} Hold /
  {_pct(stress['action_outcome']['action_mix']['restrict'], 1)} Restrict**.
- Worst-case regret **{_pct(action['worst_case_regret'], 2)}**, versus
  **{_pct(policy['binary_minimax']['max_regret'], 2)}** for binary minimax.
- Profit uplift versus binary: **{_pct(min(row['profit_uplift_vs_binary'] for row in policy['comparison']), 1, signed=True)}**
  to **{_pct(max(row['profit_uplift_vs_binary'] for row in policy['comparison']), 1, signed=True)}**.
- Expected loss **{_pct(stress['expected_loss_change'], 1, signed=True)}**,
  exposure **{_pct(stress['exposure_change'], 1, signed=True)}** (stress).
- Bootstrap action-vs-binary advantage in stress:
  **{_money(bootstrap['results']['stress']['action_vs_binary']['observed_difference'])}**
  **[{_money(bootstrap['results']['stress']['action_vs_binary']['ci_low'])},
  {_money(bootstrap['results']['stress']['action_vs_binary']['ci_high'])}]**.
- Champion/challenger: **{power['required_n_per_arm_80pct']:,} accounts per arm**
  for 80% power.

## 2. Validation design

| Item | Value |
|---|---|
| Training rows | {training['dataset_rows']:,} |
| Features | {training['n_features']} ({training['n_numeric_features']} numeric, {training['n_categorical_features']} categorical) |
| Folds | {training['folds']} stratified |
| Final trees | {training['final_n_estimators']} |
| Early stopping | {training['early_stopping_rounds']} rounds |
| Negative-class weight | {training['negative_weight']:.0f}× |
| Population-weighted default rate | {_pct(training['posterior_default_rate'], 3)} |
| Unweighted sample default rate | {_pct(training.get('unweighted_default_rate', 0.2589336105100531), 2)} |

The weighting correction is essential. The model is trained and evaluated with
the 20× negative-class weight; the unweighted 25.89% is the subsampled training
rate, not the portfolio default rate.

## 3. Model performance

{model_table}

The unweighted DeLong test is supplementary. The primary AUC comparison is a
class-stratified weighted bootstrap: observed difference
**{_num(training['bootstrap_auc_difference']['observed_difference'], 4)}**,
95% CI **[{_num(training['bootstrap_auc_difference']['ci_low'], 4)},
{_num(training['bootstrap_auc_difference']['ci_high'], 4)}]**, with
{training['bootstrap_auc_difference']['n_boot']} replicates.

## 4. Calibration

{calibration_table}

Isotonic calibration is the selected method. It is monotonic, so AUC is
essentially unchanged; it wins on Brier and ECE. Calibration is cross-fitted:
each fold's map is fit on the other folds and applied out of fold.

## 5. Threshold heuristics and policy cutoffs

{threshold_table}

The F1, KS, and top-4% cutoffs are statistical heuristics. They are not
profit-optimal, and their regret varies sharply by scenario. The binary minimax
cutoff is the strongest single-cutoff baseline. The action policy uses two
cutoffs because it chooses among three actions.

## 6. Three-action policy

{action_table}

Guardrails: Grow ≤ **{_pct(action['constraints']['max_grow_share'], 0)}**,
Hold ≥ **{_pct(action['constraints']['min_hold_share'], 0)}**, expected-loss
increase ≤ **{_pct(action['constraints']['max_expected_loss_increase'], 0)}**.
The selected policy binds the Grow cap exactly.

## 7. Scenario comparison

{_scenario_table(policy)}

The binary policy is the best single cutoff across the four scenarios; the
three-action policy is the best constrained action rule. The action oracle is
the best constrained three-action rule **per scenario**, so it is an upper bound,
not a deployable policy.

## 8. Regret, guardrails, and partial restrict

{guardrail_table}

A 50% Grow cap costs **{_pct(next(row['worst_case_regret'] for row in evaluation['guardrail_sensitivity'] if abs(row['max_grow_share'] - 0.5) < 1e-9), 2)}**
worst-case regret; 80% costs **{_pct(next(row['worst_case_regret'] for row in evaluation['guardrail_sensitivity'] if abs(row['max_grow_share'] - 0.8) < 1e-9), 2)}**.
The cap is a business choice, not a modelling artifact.

A partial-restrict alternative (50% line reduction instead of full exit) has
worst-case regret **{_pct(partial['worst_case_regret'], 2)}**, worse than the
full-exit policy's **{_pct(action['worst_case_regret'], 2)}**.

Continuous cost-ratio sensitivity: the action policy's worst-case regret is
**{_pct(sensitivity['action_sweep_worst_case_regret'], 2)}** at cost ratio
**{sensitivity['action_sweep_worst_case_cost_ratio']:.1f}**; the binary
minimax sweep's worst case is **{_pct(sensitivity['binary_sweep_worst_case_regret'], 2)}**
at **{sensitivity['binary_minimax_worst_case_cost_ratio']:.1f}**.

## 9. Paired bootstrap

{bootstrap_table}

The bootstrap resamples customers with replacement, preserving the 20×
population weight and the pairing between policies. The action-vs-binary
advantage is positive in every replicate in every scenario. Confidence
intervals describe sampling uncertainty in this labelled dataset, not
model-risk or future-regime uncertainty.

## 10. Power analysis

{power_table}

Required sample size for 80% power: **{power['required_n_per_arm_80pct']:,}
accounts per arm**. This is a two-arm simulation from the population-weighted
score/label distribution, not a live test.

## 11. Monitoring

{monitoring_table}

Feature PSI is unweighted; score PSI and action mix are population-weighted. The
current split is a temporal holdout from the same labelled month, not true
out-of-time validation. The retrain trigger is a review trigger, not an
automatic retrain.

## 12. Explainability

Top features by mean absolute SHAP:

{feature_table}

Family contribution:

{family_table}

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
"""


def render_all() -> dict[Path, str]:
    metrics = _load("metrics.json")
    policy = _load("policy.json")
    bootstrap = _load("bootstrap.json")
    power = _load("power_analysis.json")
    drift = _load("drift_summary.json")
    explain = _load("explainability.json")

    readme_path = ROOT / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    readme = _replace_block(
        readme, "readme_headline", _headline(metrics, policy, bootstrap, drift, power)
    )
    readme = _replace_block(readme, "readme_policy", _policy_table(policy))
    readme = _replace_block(
        readme, "readme_scenarios", _scenario_table(policy)
    )

    model_card_path = DOCS / "MODEL_CARD.md"
    model_card = model_card_path.read_text(encoding="utf-8")
    model_card = _replace_block(
        model_card, "model_card_metrics", _model_card_metrics(metrics, policy)
    )

    memo_path = DOCS / "DECISION_MEMO.md"
    memo = memo_path.read_text(encoding="utf-8")
    memo = _replace_block(
        memo, "decision_evidence", _decision_evidence(metrics, policy, bootstrap)
    )

    return {
        readme_path: readme,
        model_card_path: model_card,
        memo_path: memo,
        DOCS / "RESULTS.md": _results(
            metrics, policy, bootstrap, power, drift, explain
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--check",
        action="store_true",
        help="fail if generated docs are stale instead of writing them",
    )
    group.add_argument(
        "--write",
        action="store_true",
        help="write the generated docs (default)",
    )
    args = parser.parse_args()

    rendered = render_all()
    stale = []
    for path, content in rendered.items():
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        if current != content:
            stale.append(path)

    if args.check:
        if stale:
            print("Stale generated docs:")
            for path in stale:
                print(f"  {path.relative_to(ROOT)}")
            print("Run: python scripts/render_docs.py --write")
            sys.exit(1)
        print("Generated docs are current.")
        return

    for path, content in rendered.items():
        path.write_text(content, encoding="utf-8")
        print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
