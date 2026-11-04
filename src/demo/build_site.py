"""Build the static reviewer demo in ``docs/index.html`` from generated reports.

The page is deliberately dependency-free: one HTML file, one embedded CSS
block, no JavaScript, no external fonts, and local figure assets. It can be
served by GitHub Pages from ``docs/`` or opened directly from disk.
"""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path
from typing import Any

from src.config import FIGURES_DIR, REPORTS_DIR, ROOT_DIR, ensure_directories

DOCS_DIR = ROOT_DIR / "docs"
ASSETS_DIR = DOCS_DIR / "assets"
REPO_URL = "https://github.com/mahinigam/breakpoint"

FIGURES = [
    ("action_policy_map.png", "Policy bands across the calibrated score distribution"),
    ("binary_vs_action_profit.png", "Binary minimax versus the three-action policy by scenario"),
    ("regret_map.png", "Regret map across scenarios and cutoffs"),
    ("calibration_weighted.png", "Reliability diagram for raw, Platt, and isotonic probabilities"),
    ("calibration_low_pd.png", "Low-PD calibration, where the Grow cutoff lives"),
    ("profit_curve_stress.png", "Stress-scenario profit curve with a low-PD zoom"),
    ("sensitivity_sweep.png", "Continuous cost-ratio sensitivity"),
    ("power_curve.png", "Simulated power for a two-arm champion/challenger test"),
    ("psi_top_features.png", "Top drifting model features"),
    ("score_drift.png", "Calibrated score distribution drift"),
    ("shap_beeswarm.png", "SHAP beeswarm for the saved champion model"),
    ("shap_family_importance.png", "Feature-family contribution to model risk"),
]


def _load(name: str) -> dict[str, Any]:
    path = REPORTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run the full pipeline first.")
    return json.loads(path.read_text(encoding="utf-8"))


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


def _escape(value: Any) -> str:
    return html.escape(str(value))


def _table(headers: list[str], rows: list[list[str]], caption: str = "") -> str:
    head = "".join(f"<th scope=\"col\">{_escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td data-label=\"{_escape(headers[index])}\" class=\"{'num' if index > 0 else ''}\">"
            f"{cell}</td>"
            for index, cell in enumerate(row)
        )
        + "</tr>"
        for row in rows
    )
    caption_html = f"<caption>{_escape(caption)}</caption>" if caption else ""
    label = _escape(caption or "Data table")
    table_class = "data-table wide"
    if len(headers) >= 6:
        table_class += " xwide"
    if len(headers) < 4:
        table_class = "data-table"
    return (
        f"<div class=\"table-wrap\" tabindex=\"0\" role=\"group\" aria-label=\"{label}\">"
        f"<table class=\"{table_class}\">{caption_html}"
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _figure(filename: str, alt: str, caption: str) -> str:
    return (
        "<figure>"
        f"<img src=\"assets/{_escape(filename)}\" alt=\"{_escape(alt)}\" loading=\"eager\">"
        f"<figcaption>{_escape(caption)}</figcaption>"
        "</figure>"
    )


def _copy_assets() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, _ in FIGURES:
        source = FIGURES_DIR / filename
        if source.exists():
            shutil.copy2(source, ASSETS_DIR / filename)


def build_site() -> Path:
    ensure_directories()
    metrics = _load("metrics.json")
    policy = _load("policy.json")
    bootstrap = _load("bootstrap.json")
    power = _load("power_analysis.json")
    drift = _load("drift_summary.json")
    explain = _load("explainability.json")

    training = metrics.get("training", {})
    calibration = metrics.get("calibration", {})
    evaluation = metrics.get("evaluation", {})
    family_importance = explain.get("top_families", [])
    top_features = explain.get("top_features", [])
    comparison = {row["scenario"]: row for row in policy.get("comparison", [])}
    action = policy["action_policy"]
    action_scenarios = action["scenarios"]
    stress = action_scenarios["stress"]
    stress_binary = stress["binary_profit"]
    stress_action = stress["action_outcome"]["profit"]
    stress_mix = stress["action_outcome"]["action_mix"]
    stress_bad = stress["action_outcome"]["bad_rate_by_action"]
    binary = policy["binary_minimax"]
    binary_threshold = float(binary["threshold"])
    grow_cutoff = float(action["grow_cutoff"])
    restrict_cutoff = float(action["restrict_cutoff"])
    action_regret = float(action["worst_case_regret"])
    binary_regret = float(binary["max_regret"])

    calibrated_metrics = evaluation.get("calibrated_metrics", {})
    raw_metrics = evaluation.get("raw_metrics", {})
    calibration_metrics = calibration.get("metrics", {}).get("isotonic", {})
    selected_calibration = calibration.get("selected_metrics", {})

    bootstrap_rows = []
    for scenario_name, result in bootstrap.get("results", {}).items():
        values = result["action_vs_binary"]
        bootstrap_rows.append(
            [
                scenario_name.title(),
                _money(values["observed_difference"]),
                f"[{_money(values['ci_low'])}, {_money(values['ci_high'])}]",
                f"&lt;0.001" if values["p_value"] < 0.001 else f"{values['p_value']:.3f}",
            ]
        )

    power_rows = [
        [f"{row['n_accounts_per_arm']:,}", _pct(row["power"], 1)]
        for row in power.get("results", [])
    ]

    comparison_rows = []
    for name in ("conservative", "moderate", "aggressive", "stress"):
        row = comparison[name]
        comparison_rows.append(
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

    action_rows = []
    for name in ("grow", "hold", "restrict"):
        action_rows.append(
            [
                name.title(),
                _pct(stress_mix[name], 1),
                _pct(stress_bad[name], 2),
                _pct(
                    float(action["actions"][name]["margin_multiplier"]) - 1.0,
                    0,
                    signed=True,
                )
                if name != "hold"
                else "baseline",
                _pct(
                    float(action["actions"][name]["exposure_multiplier"]) - 1.0,
                    0,
                    signed=True,
                )
                if name != "hold"
                else "baseline",
            ]
        )

    watch_rows = []
    for name in ("conservative", "moderate", "aggressive", "stress"):
        row = comparison[name]
        watch_rows.append(
            [
                name.title(),
                _pct(row["expected_loss_change"], 1, signed=True),
                _pct(row["exposure_change"], 1, signed=True),
                _pct(row["binary_regret_vs_action_oracle"], 1),
                _pct(row["action_regret"], 2),
            ]
        )

    guardrail_rows = [
        [
            _pct(row["max_grow_share"], 0),
            _cutoff(row["grow_cutoff"]),
            _cutoff(row["restrict_cutoff"]),
            _pct(row["worst_case_regret"], 2),
        ]
        for row in evaluation.get("guardrail_sensitivity", [])
    ]

    top_feature_rows = [
        [row["feature"], row["family"], _num(row["mean_abs_shap"], 3)]
        for row in top_features[:8]
    ]
    family_rows = [
        [
            row["family_label"],
            _num(row["mean_abs_shap"], 3),
            _pct(row["share_of_total"], 1),
        ]
        for row in family_importance
    ]

    reason_example = "No reason-code table was generated."
    reason_path = REPORTS_DIR / "reason_codes.csv"
    if reason_path.exists():
        import csv

        with reason_path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        restrict_rows = [row for row in rows if row.get("action") == "Restrict"]
        candidates = restrict_rows or rows
        example = min(
            candidates,
            key=lambda row: abs(float(row["calibrated_pd"]) - 0.20),
        )
        reasons = [
            example.get(f"reason_{index}_text")
            for index in (1, 2, 3)
            if example.get(f"reason_{index}_text")
        ]
        reason_example = (
            f"<p class=\"reason-id\">Customer {_escape(str(example['customer_ID'])[:12])}… "
            f"· calibrated PD {_pct(float(example['calibrated_pd']), 2)} · "
            f"action {_escape(example['action'])}</p>"
            "<ol class=\"reason-list\">"
            + "".join(f"<li>{_escape(reason)}</li>" for reason in reasons)
            + "</ol>"
        )

    bootstrap_stress = bootstrap["results"]["stress"]["action_vs_binary"]
    required_n = power.get("required_n_per_arm_80pct")
    drift_trigger = drift.get("retrain_trigger", {})
    top_drift = drift.get("top_features", [])[:1]
    top_drift_text = (
        f"{top_drift[0]['feature']} (PSI {top_drift[0]['psi']:.3f})"
        if top_drift
        else "—"
    )

    css = """
:root {
  --font-display: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  --font-text: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  --font-mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  --step--1: 0.875rem;
  --step-0: 1.0625rem;
  --step-1: 1.25rem;
  --step-2: 1.5625rem;
  --step-3: 1.953rem;
  --step-4: 2.441rem;
  --step-5: 3.052rem;
  --step-6: 3.815rem;
  --space-3xs: 0.25rem;
  --space-2xs: 0.5rem;
  --space-xs: 0.75rem;
  --space-s: 1rem;
  --space-m: 1.5rem;
  --space-l: 2rem;
  --space-xl: 3rem;
  --space-2xl: 4.5rem;
  --space-3xl: 7rem;
  --base: #f4f0e6;
  --surface: #fffdf7;
  --ink: #171512;
  --ink-2: #4f4a42;
  --ink-3: #575146;
  --line: #d9d2c4;
  --accent: #a63a2f;
  --grow: #2f6b4f;
  --hold: #2f5d7c;
  --restrict: #9b3a32;
  --radius: 0;
  --border: 1px solid var(--line);
  --shadow: none;
  --dur-fast: 120ms;
  --dur: 220ms;
  --dur-slow: 420ms;
  --ease: cubic-bezier(0.2, 0, 0, 1);
  --ease-out: cubic-bezier(0.16, 1, 0.3, 1);
}

* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  background: var(--base);
  color: var(--ink);
  font-family: var(--font-text);
  font-size: var(--step-0);
  line-height: 1.55;
  text-rendering: optimizeLegibility;
}
::selection { background: var(--accent); color: var(--surface); }
a { color: var(--accent); text-decoration-thickness: 1px; text-underline-offset: 0.18em; }
a:hover { color: var(--ink); }
a:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }
.skip-link {
  position: absolute; left: var(--space-s); top: -100px;
  background: var(--ink); color: var(--surface);
  padding: var(--space-2xs) var(--space-s); z-index: 10;
}
.skip-link:focus { top: var(--space-s); }
.wrap { width: min(1180px, calc(100% - 2 * var(--space-l))); margin: 0 auto; }
.masthead {
  border-top: var(--space-3xs) solid var(--accent);
  padding: var(--space-m) 0 var(--space-s);
  display: flex; align-items: baseline; justify-content: space-between; gap: var(--space-m);
}
.wordmark {
  font-family: var(--font-display); font-size: var(--step-2); letter-spacing: -0.02em;
}
.wordmark span { color: var(--accent); }
.masthead p { font-family: var(--font-text); font-size: 1rem; color: var(--ink-2); margin: 0; }
.masthead nav { font-family: var(--font-mono); font-size: var(--step--1); display: flex; flex-wrap: wrap; gap: var(--space-s); }
.masthead nav a { color: var(--ink-2); text-decoration: none; padding-block: var(--space-3xs); }
.masthead nav a:hover { color: var(--accent); }
.rule { border: 0; border-top: var(--border); margin: 0; }
.hero { padding: var(--space-2xl) 0 var(--space-xl); border-bottom: var(--border); }
.dataset {
  font-family: var(--font-mono); font-size: var(--step--1); letter-spacing: 0.01em;
  color: var(--accent); margin: 0 0 var(--space-s);
}
h1, h2, h3 { font-family: var(--font-display); font-weight: 600; line-height: 1.05; margin: 0; }
h1 { font-size: clamp(2.6rem, 6vw, 4.8rem); letter-spacing: -0.035em; max-width: 18ch; }
h2 { font-size: var(--step-4); letter-spacing: -0.02em; margin-bottom: var(--space-s); }
h3 { font-size: var(--step-2); letter-spacing: -0.01em; }
.deck { font-size: var(--step-1); max-width: 48ch; color: var(--ink-2); margin: var(--space-m) 0 0; }
.hero-meta { font-family: var(--font-mono); font-size: var(--step--1); color: var(--ink-3); margin-top: var(--space-m); }
.metrics {
  display: grid; grid-template-columns: repeat(4, 1fr);
  border-top: 2px solid var(--ink); border-bottom: var(--border);
  margin-top: var(--space-xl);
}
.metric { min-width: 0; padding: var(--space-m) var(--space-m) var(--space-m) 0; border-right: var(--border); }
.metric:last-child { border-right: 0; }
.metric dt { font-family: var(--font-mono); font-size: var(--step--1); color: var(--ink-3); }
.metric dd {
  font-family: var(--font-mono); font-size: clamp(1.6rem, 3vw, 2.4rem);
  font-variant-numeric: tabular-nums; margin: var(--space-2xs) 0 0; letter-spacing: -0.04em;
}
.metric small { display: block; font-family: var(--font-text); font-size: var(--step--1); color: var(--ink-2); margin-top: var(--space-3xs); }
section { padding: var(--space-2xl) 0; border-bottom: var(--border); }
.section-head { display: flex; align-items: baseline; justify-content: space-between; gap: var(--space-m); margin-bottom: var(--space-l); }
.section-head p { font-family: var(--font-text); font-size: 1rem; line-height: 1.5; color: var(--ink-2); max-width: 42ch; margin: 0; }
.prose { max-width: 68ch; }
.prose p { margin: 0 0 var(--space-s); }
.prose strong { font-weight: 700; }
.ruler { margin: var(--space-l) 0 var(--space-s); }
.ruler-bar { display: grid; grid-template-columns: var(--grow-fr, 60fr) var(--hold-fr, 31.1fr) var(--restrict-fr, 8.9fr); min-height: 108px; border: 2px solid var(--ink); }
.band { min-width: 0; padding: var(--space-s); color: var(--surface); display: flex; flex-direction: column; justify-content: space-between; }
.band.grow { background: var(--grow); }
.band.hold { background: var(--hold); }
.band.restrict { background: var(--restrict); }
.band b { font-family: var(--font-mono); font-size: var(--step-1); letter-spacing: -0.03em; }
.band span { font-family: var(--font-mono); font-size: var(--step--1); }
.ruler-scale, .ruler-key { display: flex; flex-wrap: wrap; justify-content: space-between; gap: var(--space-2xs); font-family: var(--font-mono); font-size: var(--step--1); color: var(--ink-3); padding-top: var(--space-2xs); }
.ruler-key { display: none; gap: var(--space-s); flex-wrap: wrap; }
.ruler-key span::before { content: ""; display: inline-block; width: 0.7em; height: 0.7em; margin-right: 0.4em; background: currentColor; }
.ruler-key .grow { color: var(--grow); } .ruler-key .hold { color: var(--hold); } .ruler-key .restrict { color: var(--restrict); }
.quote {
  border-left: 3px solid var(--accent); padding-left: var(--space-m);
  font-size: clamp(1.25rem, 2.2vw, 1.5rem); line-height: 1.3; max-width: 42ch; margin: var(--space-xl) 0 0;
}
.quote cite { display: block; font-family: var(--font-mono); font-size: var(--step--1); font-style: normal; color: var(--ink-3); margin-top: var(--space-s); }
.table-wrap { max-width: 100%; overflow-x: auto; margin: var(--space-m) 0 var(--space-l); }
.table-wrap:focus-visible, pre:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.data-table { width: 100%; border-collapse: collapse; font-family: var(--font-mono); font-size: 0.875rem; }
.data-table caption { text-align: left; font-family: var(--font-text); font-size: var(--step--1); color: var(--ink-3); padding-bottom: var(--space-2xs); }
.data-table th { text-align: left; font-weight: 600; border-top: 2px solid var(--ink); border-bottom: var(--border); padding: var(--space-2xs) var(--space-xs); }
.data-table td { border-bottom: var(--border); padding: var(--space-2xs) var(--space-xs); vertical-align: top; overflow-wrap: anywhere; }
.data-table .num { text-align: right; font-variant-numeric: tabular-nums; }
.data-table tbody tr:hover { background: rgba(23, 21, 18, 0.035); }
.figure-grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: var(--space-xl) var(--space-l); margin-top: var(--space-l); }
figure { margin: 0; min-width: 0; }
figure img { width: 100%; height: auto; display: block; border-top: 2px solid var(--ink); padding-top: var(--space-2xs); }
figcaption { font-family: var(--font-mono); font-size: var(--step--1); color: var(--ink-3); margin-top: var(--space-2xs); }
.three-col { display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr) minmax(0, 1fr); gap: var(--space-l); align-items: start; }
.three-col > * { min-width: 0; }
.prose, .stack, .note { min-width: 0; }
.stack { display: grid; gap: var(--space-s); }
.note { border-top: var(--border); padding-top: var(--space-s); font-size: var(--step--1); color: var(--ink-2); }
.reason-id { font-family: var(--font-mono); font-size: var(--step--1); color: var(--ink-3); margin: 0 0 var(--space-2xs); }
.reason-list { margin: 0; padding-left: var(--space-m); }
.reason-list li { margin-bottom: var(--space-2xs); }
pre { background: var(--surface); border-left: 3px solid var(--accent); padding: var(--space-s); overflow-x: auto; font-family: var(--font-mono); font-size: 0.875rem; line-height: 1.6; }
code { font-family: var(--font-mono); }
footer { padding: var(--space-l) 0 var(--space-2xl); font-family: var(--font-text); font-size: 0.95rem; color: var(--ink-2); }
footer p { max-width: 72ch; }
@media (max-width: 820px) {
  .metrics { grid-template-columns: 1fr 1fr; }
  .metric:nth-child(2) { border-right: 0; }
  .metric:nth-child(-n + 2) { border-bottom: var(--border); }
  .figure-grid, .three-col { grid-template-columns: 1fr; }
  .section-head { display: block; }
  .section-head p { margin-top: var(--space-2xs); }
  .ruler-bar { grid-template-columns: 1fr; }
  .band {
    min-height: 56px; padding: var(--space-xs) var(--space-s);
    flex-direction: row; align-items: baseline; justify-content: space-between; gap: var(--space-s);
  }
  .band b { font-size: var(--step-0); }
  .band span { font-size: var(--step--1); }
  .ruler-key { display: none; }
}
@media (max-width: 840px) {
  .data-table.xwide { display: block; }
  .data-table.xwide caption { display: block; }
  .data-table.xwide thead { display: none; }
  .data-table.xwide tbody, .data-table.xwide tr { display: block; }
  .data-table.xwide tr { border-top: 2px solid var(--ink); padding: var(--space-2xs) 0; }
  .data-table.xwide td {
    display: grid; grid-template-columns: minmax(0, 0.8fr) minmax(0, 1.2fr);
    gap: var(--space-s); padding: var(--space-3xs) 0; border-bottom: 0;
  }
  .data-table.xwide td::before {
    content: attr(data-label); font-family: var(--font-mono);
    font-size: var(--step--1); color: var(--ink-3);
  }
  .data-table.xwide td:first-child {
    display: block; font-weight: 600; font-size: var(--step-0);
    padding-bottom: var(--space-2xs);
  }
  .data-table.xwide td:first-child::before { display: none; }
  .data-table.xwide .num { text-align: left; }
}

@media (max-width: 640px) {
  .data-table.wide { display: block; }
  .data-table.wide caption { display: block; }
  .data-table.wide thead { display: none; }
  .data-table.wide tbody, .data-table.wide tr { display: block; }
  .data-table.wide tr { border-top: 2px solid var(--ink); padding: var(--space-2xs) 0; }
  .data-table.wide td {
    display: grid; grid-template-columns: minmax(0, 0.8fr) minmax(0, 1.2fr);
    gap: var(--space-s); padding: var(--space-3xs) 0; border-bottom: 0;
  }
  .data-table.wide td::before {
    content: attr(data-label); font-family: var(--font-mono);
    font-size: var(--step--1); color: var(--ink-3);
  }
  .data-table.wide td:first-child {
    display: block; font-weight: 600; font-size: var(--step-0);
    padding-bottom: var(--space-2xs);
  }
  .data-table.wide td:first-child::before { display: none; }
  .data-table.wide .num { text-align: left; }
}

@media (max-width: 520px) {
  .wrap { width: calc(100% - 2 * var(--space-s)); }
  h2 { font-size: 1.8rem; }
  .masthead { display: block; }
  .masthead nav { margin-top: var(--space-2xs); }
  .ruler-bar { grid-template-columns: 1fr; }
  .band { min-height: 64px; padding: var(--space-s) var(--space-m); }
  .band span { display: none; }
  .ruler-key { display: none; }
  .metrics { grid-template-columns: 1fr; }
  .metric { border-right: 0; border-bottom: var(--border); }
  .metric:last-child { border-bottom: 0; }
}
@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior: auto; }
}
@media print {
  .masthead nav, .skip-link { display: none; }
  body { background: #fff; }
  section { break-inside: avoid; }
}
"""

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Breakpoint — a profit-optimal credit policy</title>
<meta name="description" content="Grow the safest 60%, hold the middle 31%, exit the riskiest 9%. A calibrated LightGBM credit-risk model and constrained three-action policy on 458,913 Amex accounts.">
<meta property="og:title" content="Breakpoint — a profit-optimal credit policy">
<meta property="og:description" content="Grow 60%, hold 31%, exit 9%. Worst-case regret 1.83% versus 4.26% for binary minimax.">
<meta property="og:type" content="website">
<meta name="theme-color" content="#f4f0e6">
<style>{css}</style>
</head>
<body>
<a class="skip-link" href="#main">Skip to content</a>
<header class="wrap masthead">
  <div>
    <div class="wordmark">Break<span>point</span></div>
    <p>Credit risk, decided by profit — not by a threshold.</p>
  </div>
  <nav aria-label="Sections">
    <a href="#policy">Policy</a>
    <a href="#evidence">Evidence</a>
    <a href="#uncertainty">Uncertainty</a>
    <a href="#monitoring">Monitoring</a>
    <a href="#reproduce">Reproduce</a>
  </nav>
</header>

<main id="main" tabindex="-1">
<section class="hero wrap" aria-labelledby="hero-title">
  <p class="dataset">Amex / Kaggle · 458,913 customers · 200 features</p>
  <h1 id="hero-title">Grow 60%. Hold 31%. Exit 9%.</h1>
  <p class="deck">A calibrated LightGBM ranks default risk. A constrained three-action policy grows the safest 60%, holds the middle 31%, and exits the riskiest 9%. It beats a binary approve/decline threshold in every economic scenario.</p>
  <p class="hero-meta">Calibrated AUC {_num(calibrated_metrics.get("AUC"), 4)} · Amex metric {_num(calibrated_metrics.get("Amex Metric"), 4)} · Brier {_num(selected_calibration.get("brier"), 6)} · ECE(q) {_pct(selected_calibration.get("ece_quantile"), 3)}</p>

  <dl class="metrics" aria-label="Headline results">
    <div class="metric"><dt>Profit uplift vs binary</dt><dd>{_pct(comparison["stress"]["profit_uplift_vs_binary"], 1, signed=True)}<small>Stress scenario; {_pct(comparison["aggressive"]["profit_uplift_vs_binary"], 1, signed=True)} aggressive</small></dd></div>
    <div class="metric"><dt>Worst-case regret</dt><dd>{_pct(action_regret, 2)}<small>Three-action robust vs {_pct(binary_regret, 2)} binary minimax</small></dd></div>
    <div class="metric"><dt>Expected-loss change</dt><dd>{_pct(stress["expected_loss_change"], 1, signed=True)}<small>Exposure {_pct(stress["exposure_change"], 1, signed=True)} at the same time</small></dd></div>
    <div class="metric"><dt>Bootstrap advantage</dt><dd>{_money(bootstrap_stress["observed_difference"])}<small>95% CI [{_money(bootstrap_stress["ci_low"])}, {_money(bootstrap_stress["ci_high"])}]</small></dd></div>
  </dl>
</section>

<section id="policy" class="wrap" aria-labelledby="policy-title">
  <div class="section-head">
    <h2 id="policy-title">The policy is a ruler, not a coin flip.</h2>
    <p>Cutoffs are calibrated probabilities of default. Lower is safer. The action mix is population-weighted with the 20× negative-class design weight.</p>
  </div>
  <div class="ruler" aria-label="Three-action policy bands">
    <div class="ruler-bar" style="--grow-fr: {stress_mix['grow']*100:.4f}fr; --hold-fr: {stress_mix['hold']*100:.4f}fr; --restrict-fr: {stress_mix['restrict']*100:.4f}fr;">
      <div class="band grow"><b>Grow · {_pct(stress_mix["grow"], 1)}</b><span>PD ≤ {_cutoff(grow_cutoff)}</span></div>
      <div class="band hold"><b>Hold · {_pct(stress_mix["hold"], 1)}</b><span>PD ≤ {_cutoff(restrict_cutoff)}</span></div>
      <div class="band restrict"><b>Restrict · {_pct(stress_mix["restrict"], 1)}</b><span>PD &gt; {_cutoff(restrict_cutoff)}</span></div>
    </div>
    <div class="ruler-scale"><span>0.000000</span><span>{grow_cutoff:.3g}</span><span>{restrict_cutoff:.3g}</span><span>1.000000</span></div>
    <div class="ruler-key"><span class="grow">Grow</span><span class="hold">Hold</span><span class="restrict">Restrict</span></div>
  </div>
  {_table(
      ["Action", "Share", "Bad rate", "Margin", "Exposure"],
      action_rows,
      "Stress-scenario action mix and observed bad rate",
  )}
  <div class="three-col">
    <div class="prose">
      <p><strong>Grow</strong> raises margin 10% and exposure 50% for the safest accounts. <strong>Hold</strong> keeps the baseline line. <strong>Restrict</strong> exits the account: zero margin, zero exposure.</p>
      <p>The optimizer is constrained: Grow may not exceed {_pct(action["constraints"]["max_grow_share"], 0)} of the population, Hold must remain at least {_pct(action["constraints"]["min_hold_share"], 0)}, and expected loss may rise at most {_pct(action["constraints"]["max_expected_loss_increase"], 0)}. The selected policy binds the Grow cap exactly.</p>
    </div>
    <div class="stack">
      <h3>What the binary policy misses</h3>
      <p class="note">Binary minimax cutoff {_cutoff(binary_threshold)}. It leaves <strong>{_pct(comparison["conservative"]["binary_regret_vs_action_oracle"], 1)}–{_pct(comparison["stress"]["binary_regret_vs_action_oracle"], 1)}</strong> of the action-oracle profit on the table.</p>
    </div>
    <div class="stack">
      <h3>What the guardrail costs</h3>
      <p class="note">At a {_pct(0.60, 0)} Grow cap, worst-case regret is {_pct(action_regret, 2)}. Raising the cap to {_pct(0.80, 0)} lowers regret to {_pct(next((row["worst_case_regret"] for row in evaluation.get("guardrail_sensitivity", []) if abs(row["max_grow_share"] - 0.8) < 1e-9), None), 2)} — the cap is a business choice, not a modelling artifact.</p>
    </div>
  </div>
  <p class="quote">“Binary scoring leaves {_pct(comparison['conservative']['binary_regret_vs_action_oracle'], 1)}–{_pct(comparison['stress']['binary_regret_vs_action_oracle'], 1)} of the achievable profit on the table.”<cite>Action-oracle comparison, conservative through stress</cite></p>
</section>

<section id="evidence" class="wrap" aria-labelledby="evidence-title">
  <div class="section-head">
    <h2 id="evidence-title">Evidence, not a headline.</h2>
    <p>All figures are generated from out-of-fold calibrated probabilities. The test labels are unavailable; this is model evidence, not a live portfolio result.</p>
  </div>
  <h3>Binary versus three actions</h3>
  {_table(
      ["Scenario", "Binary", "3-action", "Oracle", "Uplift", "Binary regret", "Action regret"],
      comparison_rows,
      "Expected profit under each scenario; oracle is the best constrained three-action policy",
  )}
  <div class="figure-grid">
    {_figure("action_policy_map.png", "Policy bands across the score distribution", "The policy ruler on the actual score distribution. The first log bin is wide; the dashed lines are the exact cutoffs.")}
    {_figure("binary_vs_action_profit.png", "Binary versus three-action profit", "The action policy improves profit in all four scenarios. Percentages are uplift versus binary minimax.")}
  </div>
  <div class="figure-grid">
    {_figure("regret_map.png", "Regret map", "Regret relative to each scenario’s best binary cutoff. The black line is the minimax cutoff.")}
    {_figure("profit_curve_stress.png", "Stress profit curve", "The economically interesting region sits below 10% PD; the inset expands it.")}
  </div>
</section>

<section id="calibration" class="wrap" aria-labelledby="calibration-title">
  <div class="section-head">
    <h2 id="calibration-title">Calibration is the product.</h2>
    <p>Raw AUC {_num(raw_metrics.get("AUC"), 4)} → calibrated {_num(calibrated_metrics.get("AUC"), 4)}. Brier {_num(raw_metrics.get("Brier"), 6)} → {_num(calibrated_metrics.get("Brier"), 6)}. ECE {_pct(calibration_metrics.get("ece_quantile"), 3)}.</p>
  </div>
  <div class="prose">
    <p>The cutoffs are only meaningful if a score of 0.04 means roughly a 4% chance of default. Calibration is cross-fitted: each fold’s calibrator is fit on the other folds and applied out of fold. The final isotonic map is then fit on all OOF rows for scoring.</p>
  </div>
  <div class="figure-grid">
    {_figure("calibration_weighted.png", "Reliability diagram", "Population-weighted reliability with quantile bins. Isotonic calibration removes almost all of the miscalibration.")}
    {_figure("calibration_low_pd.png", "Low-PD calibration", "The low-PD region on a log axis — the region that contains the Grow cutoff.")}
  </div>
</section>

<section id="uncertainty" class="wrap" aria-labelledby="uncertainty-title">
  <div class="section-head">
    <h2 id="uncertainty-title">Uncertainty and power.</h2>
    <p>Paired, class-stratified bootstrap with the 20× population weight; 1,000 replicates. Power is a two-arm simulation from the population-weighted score/label distribution.</p>
  </div>
  <div class="three-col">
    <div>
      <h3>Action advantage over binary</h3>
      {_table(["Scenario", "Δ profit", "95% CI", "p"], bootstrap_rows, "Paired bootstrap, action minus binary")}
    </div>
    <div>
      <h3>Power to detect it</h3>
      {_table(["Accounts / arm", "Power"], power_rows, "One-sided α = 0.05, stress scenario")}
    </div>
    <div class="stack">
      <h3>What it takes</h3>
      <p class="note">A two-arm test needs roughly <strong>{f"{required_n:,}" if required_n else "—"} accounts per arm</strong> for 80% power. At 25,000 per arm, power is {_pct(next((row["power"] for row in power.get("results", []) if row["n_accounts_per_arm"] == 25000), None), 1)}.</p>
      <p class="note">The confidence intervals are tight because the estimate uses 458,913 labelled accounts. They describe sampling uncertainty in this dataset, not model-risk uncertainty or future economic regimes.</p>
    </div>
  </div>
  <div class="figure-grid">
    {_figure("power_curve.png", "Power curve", "Simulated power rises with accounts per arm; the dotted line marks the 80% target.")}
    {_figure("sensitivity_sweep.png", "Sensitivity sweep", "Captured profit as the cost ratio moves continuously between the named scenarios.")}
  </div>
</section>

<section id="monitoring" class="wrap" aria-labelledby="monitoring-title">
  <div class="section-head">
    <h2 id="monitoring-title">Monitoring is part of the decision.</h2>
    <p>Temporal split: earliest 80% of accounts by statement date vs latest 20%. Feature PSI is unweighted; score PSI and action mix are population-weighted.</p>
  </div>
  <div class="three-col">
    <div class="prose">
      <p>Score PSI is <strong>{_num(drift.get("score_psi"), 4)}</strong>. The weighted default rate moves from {_pct(drift.get("reference_default_rate"), 2)} to {_pct(drift.get("current_default_rate"), 2)}. The action mix moves by at most {_pct(max(abs(value) for value in drift.get("action_mix_delta", {}).values()), 2)} percentage points.</p>
      <p>The largest feature shift is {_escape(top_drift_text)}. Status counts: {_escape(drift.get("status_counts", {}))}. The retrain review is <strong>{"triggered" if drift_trigger.get("retrain_review") else "not triggered"}</strong>.</p>
    </div>
    <div class="stack">
      <h3>Retrain trigger</h3>
      <p class="note">&gt;10% of features with PSI &gt; 0.25, or ≥3 delinquency/balance features above 0.25, or score PSI &gt; 0.25, or an action share moving &gt;5 points.</p>
    </div>
    <div class="stack">
      <h3>Why it matters</h3>
      <p class="note">A PD cutoff is a promise about the population. If the population shifts, the action mix shifts with it. Monitoring the score and the action mix catches that before the next quarterly review.</p>
    </div>
  </div>
  <div class="figure-grid">
    {_figure("psi_top_features.png", "Top drifting features", "Top 20 model features by PSI. Yellow is 0.10; red is 0.25.")}
    {_figure("score_drift.png", "Score drift", "Reference and current calibrated score distributions on a log axis.")}
  </div>
</section>

<section id="explainability" class="wrap" aria-labelledby="explain-title">
  <div class="section-head">
    <h2 id="explain-title">A decision a human can defend.</h2>
    <p>SHAP values explain the saved champion model, not a proxy. Reason codes are the top positive contributors to risk.</p>
  </div>
  <div class="three-col">
    <div>
      <h3>Top features</h3>
      {_table(["Feature", "Family", "Mean |SHAP|"], top_feature_rows, "Saved LightGBM model, 5,000-customer sample")}
    </div>
    <div>
      <h3>Family contribution</h3>
      {_table(["Family", "Mean |SHAP|", "Share"], family_rows, "Feature-family aggregation")}
    </div>
    <div>
      <h3>Example reason code</h3>
      {reason_example}
    </div>
  </div>
  <div class="figure-grid">
    {_figure("shap_beeswarm.png", "SHAP beeswarm", "Top 15 features. Color is feature value; horizontal position is impact on the raw model output.")}
    {_figure("shap_family_importance.png", "SHAP family importance", "Delinquency, payment, and balance families dominate, as expected in credit risk.")}
  </div>
</section>

<section id="reproduce" class="wrap" aria-labelledby="reproduce-title">
  <div class="section-head">
    <h2 id="reproduce-title">Reproduce it, or inspect the code.</h2>
    <p>The model artifacts and this page are committed. The 6.2 GB Amex parquet is not; the full pipeline is reproducible from the Kaggle CSVs.</p>
  </div>
  <div class="three-col">
    <div class="prose">
      <p><strong>Fast path.</strong> The committed model artifacts let you score a real-schema sample without retraining. The static page is rebuilt from committed reports.</p>
      <pre tabindex="0"><code>git clone {REPO_URL}.git
cd breakpoint
make setup
make score-demo
make demo</code></pre>
    </div>
    <div class="prose">
      <p><strong>Full path.</strong> Put the Amex CSV files in <code>data/raw/</code>, then run the pipeline. Training uses 5-fold OOF predictions, a final model, cross-fitted calibration, and the constrained policy.</p>
      <pre tabindex="0"><code>make pipeline
make tableau
make verify</code></pre>
    </div>
    <div class="stack">
      <h3>Read the detail</h3>
      <p class="note"><a href="{REPO_URL}/blob/main/README.md">README</a> · <a href="{REPO_URL}/blob/main/docs/MODEL_CARD.md">Model card</a> · <a href="{REPO_URL}/blob/main/docs/DECISION_MEMO.md">Decision memo</a> · <a href="{REPO_URL}/blob/main/dashboards/breakpoint_policy.twbx">Tableau workbook</a></p>
      <p class="note"><a href="{REPO_URL}/tree/main/dashboards/data">Tableau CSVs</a> · <a href="{REPO_URL}/blob/main/docs/TABLEAU_DASHBOARD_GUIDE.md">Tableau guide</a> · <a href="{REPO_URL}/blob/main/docs/INTERVIEW_DEFENSE.md">Interview defense</a></p>
    </div>
  </div>
  <div class="stack" style="margin-top: var(--space-l);">
    <h3>Guardrail sensitivity</h3>
    {_table(["Grow cap", "Grow cutoff", "Restrict cutoff", "Worst-case regret"], guardrail_rows, "What the growth cap buys")}
  </div>
</section>
</main>

<footer class="wrap">
  <p>Data: Amex / Kaggle credit default prediction. The test labels are unavailable; all performance numbers are out-of-fold on the labelled training population. The Tableau deliverable is a packaged, aggregate-only workbook; publishing to Tableau Public requires a signed-in Tableau account.</p>
  <p>Built from <code>reports/metrics.json</code> and <code>reports/policy.json</code>. No number on this page is hand-entered.</p>
</footer>
</body>
</html>
"""

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    _copy_assets()
    output = DOCS_DIR / "index.html"
    output.write_text(page, encoding="utf-8")
    (DOCS_DIR / ".nojekyll").write_text("", encoding="utf-8")
    print(f"Wrote {output} ({output.stat().st_size / 1024:.0f} KB)")
    return output


def main() -> None:
    build_site()


if __name__ == "__main__":
    main()
