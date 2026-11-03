"""Decision analysis: profit curves, minimax regret, and the three-action policy.

Everything in this module uses **calibrated** probabilities. The outputs are
the single source of truth for the README, memo, model card, Tableau extract,
and the scoring CLI's default policy.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    ACTION_ECONOMICS,
    CALIBRATION_METHOD,
    COST_RATIO_SWEEP,
    DASHBOARD_DATA_DIR,
    ECONOMIC_SCENARIOS,
    FIGURES_DIR,
    NEGATIVE_WEIGHT,
    PATHS,
    POLICY_CONSTRAINTS,
    REPORTS_DIR,
    THRESHOLD_GRID_CONFIG,
    ensure_directories,
    scenario_cost_ratio,
    scenario_loss,
)
from src.economics.cutoff_methods import (
    get_f1_optimal_threshold,
    get_fixed_threshold,
    get_ks_optimal_threshold,
    get_top_pct_threshold,
)
from src.economics.profit_curve import ScoreDistribution, plot_profit_curves, threshold_grid
from src.economics.regret_map import plot_regret_map
from src.economics.sensitivity import plot_sensitivity_sweep, run_sensitivity_sweep
from src.evaluation.metrics import evaluate_model
from src.policy.actions import (
    binary_minimax_threshold,
    binary_outcome,
    optimize_robust_action_policy,
)
from src.reporting import environment_metadata, update_json, write_json


def _load_calibrated() -> pd.DataFrame:
    path = PATHS["calibrated_predictions"]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.models.calibration` first."
        )
    frame = pd.read_parquet(path)
    required = {"target", "lgb_prob", "lgb_prob_calibrated"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Calibrated OOF is missing columns: {sorted(missing)}")
    return frame


def _clip_threshold(value: float, grid: np.ndarray) -> float:
    if not np.isfinite(value):
        return float(grid[-1])
    return float(np.clip(value, grid[0], grid[-1]))


def _score_histogram(
    distribution: ScoreDistribution,
    grow_cutoff: float,
    restrict_cutoff: float,
    n_bins: int = 60,
) -> pd.DataFrame:
    """Population-weighted score histogram with the policy's action band."""
    scores = distribution.scores
    positive = scores[scores > 0]
    lower = max(float(positive.min()), 1e-6) if len(positive) else 1e-6
    edges = np.concatenate(
        [[0.0], np.logspace(np.log10(lower), 0.0, n_bins)]
    )
    edges = np.unique(edges)
    bin_ids = np.digitize(scores, edges[1:-1], right=True)

    weights = distribution.row_weights()
    goods = distribution.row_good()
    bads = distribution.row_bad()
    rows: list[dict[str, Any]] = []
    for bin_id in range(len(edges) - 1):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        midpoint = float(np.sqrt(max(edges[bin_id], 1e-10) * max(edges[bin_id + 1], 1e-9)))
        if midpoint <= grow_cutoff:
            action = "grow"
        elif midpoint <= restrict_cutoff:
            action = "hold"
        else:
            action = "restrict"
        weight = float(weights[mask].sum())
        good = float(goods[mask].sum())
        bad = float(bads[mask].sum())
        rows.append(
            {
                "bin": bin_id,
                "lower": float(edges[bin_id]),
                "upper": float(edges[bin_id + 1]),
                "midpoint": midpoint,
                "count": int(mask.sum()),
                "weight": weight,
                "good_weight": good,
                "bad_weight": bad,
                "bad_rate": bad / weight if weight else 0.0,
                "action": action,
            }
        )
    return pd.DataFrame(rows)


def _plot_action_map(
    histogram: pd.DataFrame,
    grow_cutoff: float,
    restrict_cutoff: float,
    action_mix: dict[str, float],
    bad_rate_by_action: dict[str, float],
    output_dir: Path,
) -> None:
    colors = {"grow": "#2ca02c", "hold": "#1f77b4", "restrict": "#d62728"}
    fig, (ax_hist, ax_mix) = plt.subplots(
        1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [2.0, 1.0]}
    )
    for action, group in histogram.groupby("action"):
        ax_hist.bar(
            group["lower"],
            group["weight"] / histogram["weight"].sum(),
            width=group["upper"] - group["lower"],
            align="edge",
            color=colors.get(action, "#777777"),
            label=action.title(),
            alpha=0.85,
        )
    ax_hist.set_xscale("log")
    ax_hist.axvline(grow_cutoff, color="#111111", linestyle="--", linewidth=1.5)
    ax_hist.axvline(restrict_cutoff, color="#111111", linestyle="--", linewidth=1.5)
    ax_hist.set_xlabel("Calibrated probability of default (log scale)")
    ax_hist.set_ylabel("Share of population")
    ax_hist.set_title("Policy bands across the score distribution")
    ax_hist.legend(frameon=False)
    ax_hist.grid(True, linestyle="--", alpha=0.25)

    shares = pd.Series(action_mix).reindex(["grow", "hold", "restrict"]).fillna(0.0)
    bad_rates = (
        pd.Series(bad_rate_by_action)
        .reindex(["grow", "hold", "restrict"])
        .fillna(0.0)
    )
    ax_mix.bar(
        [name.title() for name in shares.index],
        shares.to_numpy() * 100,
        color=[colors[name] for name in shares.index],
    )
    for index, (share, bad_rate) in enumerate(zip(shares, bad_rates)):
        ax_mix.text(
            index,
            share * 100 + 1.5,
            f"{share:.1%}\nbad {bad_rate:.2%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax_mix.set_ylim(0, 100)
    ax_mix.set_ylabel("Share of population (%)")
    ax_mix.set_title("Action mix")
    ax_mix.grid(True, axis="y", linestyle="--", alpha=0.25)

    fig.suptitle(
        f"Grow below {grow_cutoff:.4f} PD; hold below {restrict_cutoff:.4f}; "
        "restrict above",
        fontsize=11,
    )
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "action_policy_map.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_binary_vs_action(
    rows: list[dict[str, Any]], output_dir: Path
) -> None:
    labels = [row["scenario"].title() for row in rows]
    binary = np.array([row["binary_profit"] for row in rows]) / 1e6
    action = np.array([row["action_profit"] for row in rows]) / 1e6
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width / 2, binary, width, label="Binary minimax", color="#8c8c8c")
    ax.bar(x + width / 2, action, width, label="Three-action robust", color="#1f77b4")
    for index, (b_value, a_value) in enumerate(zip(binary, action)):
        ax.text(
            index,
            max(b_value, a_value) + 15,
            f"+{a_value / b_value - 1:.1%}",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Expected profit ($ millions)")
    ax.set_ylim(0, max(float(action.max()), float(binary.max())) * 1.14)
    ax.set_title("Three-action policy improves profit in every scenario")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "binary_vs_action_profit.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _guardrail_sensitivity(
    distribution: ScoreDistribution,
    grid: np.ndarray,
    baseline_loss: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cap in (0.50, 0.60, 0.70, 0.80):
        constraints = dict(POLICY_CONSTRAINTS)
        constraints["max_grow_share"] = cap
        policy = optimize_robust_action_policy(
            distribution,
            ECONOMIC_SCENARIOS,
            ACTION_ECONOMICS,
            constraints,
            grid,
            baseline_expected_loss=baseline_loss,
        )
        rows.append(
            {
                "max_grow_share": cap,
                "grow_cutoff": policy.grow_cutoff,
                "restrict_cutoff": policy.restrict_cutoff,
                "worst_case_regret": policy.worst_case_regret,
                "grow_share": float(
                    np.mean([o.action_mix["grow"] for o in policy.outcomes.values()])
                ),
                "hold_share": float(
                    np.mean([o.action_mix["hold"] for o in policy.outcomes.values()])
                ),
                "restrict_share": float(
                    np.mean([o.action_mix["restrict"] for o in policy.outcomes.values()])
                ),
                "mean_profit": float(
                    np.mean([o.profit for o in policy.outcomes.values()])
                ),
            }
        )
    return rows


def _partial_restrict_sensitivity(
    distribution: ScoreDistribution,
    grid: np.ndarray,
    baseline_loss: dict[str, float],
) -> dict[str, Any]:
    partial_actions = copy.deepcopy(ACTION_ECONOMICS)
    partial_actions["restrict"]["margin_multiplier"] = 0.50
    partial_actions["restrict"]["exposure_multiplier"] = 0.50
    # This sensitivity isolates the restrict economics. Keep the growth cap but
    # relax the expected-loss cap so the alternative remains feasible.
    constraints = dict(POLICY_CONSTRAINTS)
    constraints["max_expected_loss_increase"] = 10.0
    policy = optimize_robust_action_policy(
        distribution,
        ECONOMIC_SCENARIOS,
        partial_actions,
        constraints,
        grid,
        baseline_expected_loss=baseline_loss,
    )
    return {
        "description": "Restrict modelled as a 50% line reduction instead of a full exit",
        "grow_cutoff": policy.grow_cutoff,
        "restrict_cutoff": policy.restrict_cutoff,
        "worst_case_regret": policy.worst_case_regret,
        "scenarios": {
            name: outcome.to_dict() for name, outcome in policy.outcomes.items()
        },
    }


def run_decision_analysis() -> dict[str, Any]:
    ensure_directories()
    frame = _load_calibrated()
    y_true = frame["target"].to_numpy(dtype=int)
    calibrated = frame["lgb_prob_calibrated"].to_numpy(dtype=float)

    distribution = ScoreDistribution.from_labels(
        y_true, calibrated, negative_weight=NEGATIVE_WEIGHT
    )
    grid = threshold_grid(
        calibrated,
        log_min=float(THRESHOLD_GRID_CONFIG["log_min"]),
        log_max=float(THRESHOLD_GRID_CONFIG["log_max"]),
        log_points=int(THRESHOLD_GRID_CONFIG["log_points"]),
        quantile_points=int(THRESHOLD_GRID_CONFIG["quantile_points"]),
        include=THRESHOLD_GRID_CONFIG.get("include", []),
    )

    raw_metrics = {
        key: float(value)
        for key, value in evaluate_model(
            y_true, frame["lgb_prob"].to_numpy(dtype=float), NEGATIVE_WEIGHT
        ).items()
    }
    calibrated_metrics = {
        key: float(value)
        for key, value in evaluate_model(
            y_true, calibrated, NEGATIVE_WEIGHT
        ).items()
    }

    heuristics = {
        "f1": _clip_threshold(
            get_f1_optimal_threshold(y_true, calibrated, NEGATIVE_WEIGHT), grid
        ),
        "ks": _clip_threshold(
            get_ks_optimal_threshold(y_true, calibrated, NEGATIVE_WEIGHT), grid
        ),
        "top_4pct": _clip_threshold(
            get_top_pct_threshold(
                y_true, calibrated, NEGATIVE_WEIGHT, pct=0.04
            ),
            grid,
        ),
        "fixed_0_5": get_fixed_threshold(
            y_true, calibrated, NEGATIVE_WEIGHT, threshold=0.5
        ),
    }

    binary_threshold, binary_regrets, binary_profits = binary_minimax_threshold(
        distribution, ECONOMIC_SCENARIOS, grid
    )
    binary_baseline = {
        name: binary_outcome(
            distribution,
            binary_threshold,
            float(scenario["margin"]),
            scenario_loss(scenario),
            float(scenario["exposure"]),
        )
        for name, scenario in ECONOMIC_SCENARIOS.items()
    }
    baseline_loss = {
        name: float(scenario_loss(scenario) * distribution.weighted_pd_at(binary_threshold))
        for name, scenario in ECONOMIC_SCENARIOS.items()
    }

    action_policy = optimize_robust_action_policy(
        distribution,
        ECONOMIC_SCENARIOS,
        ACTION_ECONOMICS,
        POLICY_CONSTRAINTS,
        grid,
        baseline_expected_loss=baseline_loss,
    )

    # Profit curves and the regret map use the binary policy's economics.
    scenario_profits: dict[str, dict[str, Any]] = {}
    profit_rows: list[dict[str, Any]] = []
    regret_rows: list[np.ndarray] = []
    scenario_names = list(ECONOMIC_SCENARIOS)
    for name, scenario in ECONOMIC_SCENARIOS.items():
        margin = float(scenario["margin"])
        loss = scenario_loss(scenario)
        curve = distribution.profit_curve(grid, margin, loss)
        empirical_index = int(np.argmax(curve))
        empirical_opt = float(grid[empirical_index])
        maximum = float(curve[empirical_index])
        regret = (maximum - curve) / maximum if maximum > 0 else np.zeros_like(curve)
        regret_rows.append(regret)
        scenario_profits[name] = {
            "thresholds": grid,
            "profits": curve,
            "max_profit": maximum,
            "margin": margin,
            "loss": loss,
            "empirical_opt": empirical_opt,
        }
        for threshold, profit, regret_value in zip(grid, curve, regret):
            profit_rows.append(
                {
                    "scenario": name,
                    "cost_ratio": scenario_cost_ratio(scenario),
                    "threshold": float(threshold),
                    "profit": float(profit),
                    "oracle_profit": maximum,
                    "regret": float(regret_value),
                }
            )
    plot_profit_curves(scenario_profits, FIGURES_DIR)

    regret_matrix = np.stack(regret_rows)
    plot_regret_map(
        regret_matrix,
        grid,
        scenario_names,
        FIGURES_DIR,
        minimax_threshold=binary_threshold,
    )

    sensitivity_frame, sensitivity_summary = run_sensitivity_sweep(
        distribution,
        grid,
        heuristics["f1"],
        heuristics["ks"],
        binary_threshold,
        action_policy.grow_cutoff,
        action_policy.restrict_cutoff,
        ACTION_ECONOMICS,
        COST_RATIO_SWEEP,
    )
    sensitivity_frame.to_csv(REPORTS_DIR / "sensitivity.csv", index=False)
    plot_sensitivity_sweep(sensitivity_frame, FIGURES_DIR, sensitivity_summary)

    # Per-scenario comparison: binary baseline vs three-action policy.
    comparison_rows: list[dict[str, Any]] = []
    action_payload: dict[str, Any] = {
        "grow_cutoff": action_policy.grow_cutoff,
        "restrict_cutoff": action_policy.restrict_cutoff,
        "worst_case_regret": action_policy.worst_case_regret,
        "score_semantics": "grow if PD <= grow_cutoff, hold if <= restrict_cutoff, restrict above",
        "actions": ACTION_ECONOMICS,
        "constraints": POLICY_CONSTRAINTS,
        "scenarios": {},
    }
    for name, scenario in ECONOMIC_SCENARIOS.items():
        matrix = action_policy.scenarios[name]
        outcome = action_policy.outcomes[name]
        binary = binary_baseline[name]
        oracle = matrix.oracle_profit
        binary_profit = float(binary.profit)
        comparison_rows.append(
            {
                "scenario": name,
                "cost_ratio": scenario_cost_ratio(scenario),
                "binary_profit": binary_profit,
                "action_profit": float(outcome.profit),
                "action_oracle_profit": float(oracle),
                "binary_regret_vs_action_oracle": float(
                    (oracle - binary_profit) / oracle if oracle > 0 else 0.0
                ),
                "action_regret": float(outcome.regret),
                "profit_uplift_vs_binary": float(
                    outcome.profit / binary_profit - 1.0
                    if binary_profit > 0
                    else 0.0
                ),
                "expected_loss_binary": float(binary.expected_loss),
                "expected_loss_action": float(outcome.expected_loss),
                "expected_loss_change": float(
                    outcome.expected_loss / binary.expected_loss - 1.0
                    if binary.expected_loss > 0
                    else 0.0
                ),
                "exposure_binary": float(binary.exposure),
                "exposure_action": float(outcome.exposure),
                "exposure_change": float(
                    outcome.exposure / binary.exposure - 1.0
                    if binary.exposure > 0
                    else 0.0
                ),
                "action_mix": outcome.action_mix,
                "bad_rate_by_action": outcome.bad_rate_by_action,
                "binary_bad_rate": binary.bad_rate_by_action.get("approve", 0.0),
            }
        )
        action_payload["scenarios"][name] = {
            "cost_ratio": scenario_cost_ratio(scenario),
            "binary_profit": binary_profit,
            "binary_regret_vs_action_oracle": float(
                (oracle - binary_profit) / oracle if oracle > 0 else 0.0
            ),
            "action_oracle_profit": float(oracle),
            "action_outcome": outcome.to_dict(),
            "expected_loss_binary": float(binary.expected_loss),
            "exposure_binary": float(binary.exposure),
            "expected_loss_change": float(
                outcome.expected_loss / binary.expected_loss - 1.0
                if binary.expected_loss > 0
                else 0.0
            ),
            "exposure_change": float(
                outcome.exposure / binary.exposure - 1.0
                if binary.exposure > 0
                else 0.0
            ),
        }

    comparison_frame = pd.DataFrame(comparison_rows)
    comparison_frame.to_csv(REPORTS_DIR / "operating_characteristics.csv", index=False)
    _plot_binary_vs_action(comparison_rows, FIGURES_DIR)

    action_rows = []
    for name, scenario in ECONOMIC_SCENARIOS.items():
        outcome = action_policy.outcomes[name]
        actions = action_policy.scenarios[name].actions
        for action_name in ("grow", "hold", "restrict"):
            economics = actions[action_name]
            action_rows.append(
                {
                    "scenario": name,
                    "action": action_name,
                    "share": outcome.action_mix[action_name],
                    "bad_rate": outcome.bad_rate_by_action[action_name],
                    "margin": economics.margin,
                    "loss": economics.loss,
                    "exposure": economics.exposure,
                    "expected_loss": outcome.expected_loss,
                    "exposure_total": outcome.exposure,
                }
            )
    pd.DataFrame(action_rows).to_csv(
        REPORTS_DIR / "action_policy.csv", index=False
    )

    histogram = _score_histogram(
        distribution,
        action_policy.grow_cutoff,
        action_policy.restrict_cutoff,
    )
    histogram.to_csv(REPORTS_DIR / "score_histogram.csv", index=False)
    reference_outcome = action_policy.outcomes[scenario_names[0]]
    _plot_action_map(
        histogram,
        action_policy.grow_cutoff,
        action_policy.restrict_cutoff,
        reference_outcome.action_mix,
        reference_outcome.bad_rate_by_action,
        FIGURES_DIR,
    )

    guardrails = _guardrail_sensitivity(distribution, grid, baseline_loss)
    guardrail_frame = pd.DataFrame(guardrails)
    guardrail_frame.to_csv(REPORTS_DIR / "guardrail_sensitivity.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        guardrail_frame["max_grow_share"] * 100,
        guardrail_frame["worst_case_regret"] * 100,
        marker="o",
        color="#1f77b4",
        linewidth=2,
    )
    ax.set_xlabel("Maximum share of accounts allowed in Grow (%)")
    ax.set_ylabel("Worst-case regret (%)")
    ax.set_title("Guardrail sensitivity: growth cap vs worst-case regret")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "guardrail_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    partial_restrict = _partial_restrict_sensitivity(
        distribution, grid, baseline_loss
    )
    write_json(REPORTS_DIR / "partial_restrict.json", partial_restrict)

    pd.DataFrame(profit_rows).to_csv(REPORTS_DIR / "profit_curves.csv", index=False)
    regret_frame = pd.DataFrame(
        regret_matrix.T, columns=scenario_names
    )
    regret_frame.insert(0, "threshold", grid)
    regret_frame.to_csv(REPORTS_DIR / "regret_matrix.csv", index=False)

    # Tableau-ready KPI table.
    kpi_rows = []
    for row in comparison_rows:
        kpi_rows.append(
            {
                "scenario": row["scenario"],
                "cost_ratio": row["cost_ratio"],
                "binary_profit": row["binary_profit"],
                "action_profit": row["action_profit"],
                "action_oracle_profit": row["action_oracle_profit"],
                "profit_uplift_vs_binary": row["profit_uplift_vs_binary"],
                "binary_regret_vs_action_oracle": row[
                    "binary_regret_vs_action_oracle"
                ],
                "action_regret": row["action_regret"],
                "expected_loss_change": row["expected_loss_change"],
                "exposure_change": row["exposure_change"],
            }
        )
    pd.DataFrame(kpi_rows).to_csv(DASHBOARD_DATA_DIR / "kpi.csv", index=False)

    policy_payload = {
        "generated_at": environment_metadata()["generated_at"],
        "calibrated": True,
        "calibration_method": CALIBRATION_METHOD,
        "score_semantics": "lower PD is safer; approve/grow when PD <= cutoff",
        "sample_rows": int(len(frame)),
        "scenarios": {
            name: {
                **scenario,
                "cost_ratio": scenario_cost_ratio(scenario),
                "loss": scenario_loss(scenario),
            }
            for name, scenario in ECONOMIC_SCENARIOS.items()
        },
        "heuristics": heuristics,
        "binary_minimax": {
            "threshold": binary_threshold,
            "max_regret": float(max(binary_regrets.values())),
            "regrets": binary_regrets,
            "profits": binary_profits,
            "baseline": {
                name: outcome.to_dict() for name, outcome in binary_baseline.items()
            },
        },
        "action_policy": action_payload,
        "comparison": comparison_rows,
        "sensitivity": sensitivity_summary,
        "guardrail_sensitivity": guardrails,
        "partial_restrict": partial_restrict,
        "environment": environment_metadata(),
    }
    write_json(REPORTS_DIR / "policy.json", policy_payload)

    evaluation_payload = {
        "n_rows": int(len(frame)),
        "raw_metrics": raw_metrics,
        "calibrated_metrics": calibrated_metrics,
        "heuristic_thresholds": heuristics,
        "binary_minimax": {
            "threshold": binary_threshold,
            "max_regret": float(max(binary_regrets.values())),
            "regrets": binary_regrets,
            "profits": binary_profits,
        },
        "action_policy": {
            "grow_cutoff": action_policy.grow_cutoff,
            "restrict_cutoff": action_policy.restrict_cutoff,
            "worst_case_regret": action_policy.worst_case_regret,
            "scenarios": {
                name: {
                    "binary_profit": row["binary_profit"],
                    "action_profit": row["action_profit"],
                    "action_oracle_profit": row["action_oracle_profit"],
                    "binary_regret_vs_action_oracle": row[
                        "binary_regret_vs_action_oracle"
                    ],
                    "action_regret": row["action_regret"],
                    "profit_uplift_vs_binary": row["profit_uplift_vs_binary"],
                    "expected_loss_change": row["expected_loss_change"],
                    "exposure_change": row["exposure_change"],
                    "action_mix": row["action_mix"],
                    "bad_rate_by_action": row["bad_rate_by_action"],
                }
                for row in comparison_rows
            },
            "constraints": POLICY_CONSTRAINTS,
            "actions": ACTION_ECONOMICS,
        },
        "sensitivity": sensitivity_summary,
        "guardrail_sensitivity": guardrails,
        "partial_restrict": partial_restrict,
        "environment": environment_metadata(),
    }
    update_json(REPORTS_DIR / "metrics.json", "evaluation", evaluation_payload)

    print(f"Binary minimax cutoff: {binary_threshold:.6f}")
    print(
        f"Action policy: grow <= {action_policy.grow_cutoff:.6f}, "
        f"restrict > {action_policy.restrict_cutoff:.6f}"
    )
    for row in comparison_rows:
        print(
            f"  {row['scenario']:12s} binary ${row['binary_profit']/1e6:,.1f}M "
            f"-> action ${row['action_profit']/1e6:,.1f}M "
            f"({row['profit_uplift_vs_binary']:+.1%}); "
            f"regret {row['action_regret']:.2%}"
        )
    print(f"Saved reports to {REPORTS_DIR}")
    return policy_payload


def main() -> None:
    run_decision_analysis()


if __name__ == "__main__":
    main()
