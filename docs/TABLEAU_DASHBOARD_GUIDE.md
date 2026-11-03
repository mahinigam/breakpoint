# Tableau Dashboard Guide

The Tableau layer is intentionally **aggregate-only**. It contains no customer IDs, no labels, and no raw statements. Every number is exported from the canonical reports by `python -m src.decision.export_dashboard_data`; the exporter is deterministic and records a manifest.

The repository prepares the data and the workbook recipe. **Publishing to Tableau Public requires your Tableau account**, so the repository does not claim a live URL. Once you publish, add the URL to the README and the repo homepage.

## 1. Files

All files live in [`dashboards/data/`](../dashboards/data/). `manifest.json` records the row count and columns for each file.

| File | Rows | Grain | Use |
|---|---:|---|---|
| `threshold_curve.csv` | 2,324 | scenario × probability cutoff | Profit and regret curves |
| `policy_markers.csv` | 24 | scenario × policy marker | Reference lines for F1, KS, top-4%, binary minimax, action cutoffs |
| `action_mix.csv` | 12 | scenario × action | Grow/Hold/Restrict share and bad rate |
| `operating_characteristics.csv` | 4 | scenario | Binary vs action profit, expected loss, exposure |
| `score_histogram.csv` | 57 | score bin | Score distribution and action assignment |
| `calibration.csv` | 36 | method × bin | Reliability diagram |
| `sensitivity.csv` | 50 | cost ratio | Continuous economic sensitivity |
| `drift.csv` | 50 | feature | Top feature PSI values |
| `kpi.csv` | 4 | scenario | Headline KPI cards |
| `policy_card.csv` | 1 | policy | Cutoffs, guardrails, multipliers |
| `bootstrap.csv` | 8 | scenario × comparison | Profit advantage and 95% intervals |
| `power.csv` | 6 | accounts per arm | Simulated power curve |

All percentages in the exported files are already multiplied by 100 and suffixed with `_pct` (for example `profit_uplift_pct = 6.8`). Dollar values are in millions and suffixed with `_millions`.

## 2. Connect

1. Open **Tableau Public Desktop**.
2. **Connect → Text file →** `dashboards/data/threshold_curve.csv`.
3. Add the other CSVs as separate data sources (**Data → New Data Source → Text file**). For a single-source workbook, use relationships on `scenario`.
4. Confirm these field types:
   - `probability_cutoff`, `profit_millions`, `regret_pct`: continuous numeric.
   - `scenario`, `scenario_label`: dimension.
   - `bin`, `method`: dimension.
5. Create an extract (**Data → Extract → Create Extract**) so the workbook is portable and publishable. The files are small; the extract is instant.

Do **not** connect Tableau to the 6.2 GB parquet or the customer-level feature table. Those are pipeline artifacts, not dashboard data.

## 3. Calculated fields

Create these in the `threshold_curve` data source:

```tableau
// Policy band at any cutoff
IF [Probability Cutoff] <= 0.000827 THEN "Grow"
ELSEIF [Probability Cutoff] <= 0.040517 THEN "Hold"
ELSE "Restrict"
END
```

```tableau
// Profit as a share of the action oracle, 0-100
[Profit Millions] / [Oracle Profit Millions] * 100
```

```tableau
// Regret in percentage points (already exported, but explicit)
[Regret Pct]
```

```tableau
// A clean scenario label
[Scenario Label]
```

Create these in `kpi.csv`:

```tableau
// Action profit advantage in millions
[Action Profit Millions] - [Binary Profit Millions]
```

```tableau
// Binary regret as a share of action-oracle profit (already exported)
[Binary Regret Vs Oracle Pct]
```

For `bootstrap.csv`, format `p_value` as a custom number: `0.000;<0.001`, so a zero p-value displays as `<0.001`.

## 4. Dashboard layout

A 1200 × 900 dashboard is enough. Use three zones.

### Top: four KPI cards

Use `kpi.csv`, filtered to `Scenario = stress` (or a parameter to switch scenario):

| Card | Field | Format |
|---|---|---|
| Profit uplift vs binary | `profit_uplift_pct` | `+9.7%` |
| Binary regret vs oracle | `binary_regret_vs_oracle_pct` | `10.5%` |
| Expected-loss change | `expected_loss_change_pct` | `−17.8%` |
| Exposure change | `exposure_change_pct` | `+30.7%` |

Add a subtitle: **“Expected profit under labelled economic assumptions, not actual P&L.”**

### Left: the policy ruler

Use `score_histogram.csv`:

- **Columns:** `midpoint` (continuous, log axis if Tableau allows).
- **Rows:** `weight`.
- **Color:** `action_label` (Grow = green, Hold = blue, Restrict = red).
- **Detail:** `bin`.
- Add vertical reference lines at `0.000827` and `0.040517`.
- Add a text annotation: “Grow ≤ 0.000827 · Hold ≤ 0.040517 · Restrict > 0.040517”.

This is the memorable visual: the three bands sit on the actual score distribution, not on a cartoon ruler.

### Center: profit and regret

**Profit curve** from `threshold_curve.csv`:

- **Columns:** `probability_cutoff` (continuous, log axis; the interesting action region is below 0.1).
- **Rows:** `profit_millions`.
- **Color:** `scenario_label`.
- **Detail:** `scenario`.
- Add a reference line at `0.053124` for binary minimax, and at `0.000827` / `0.040517` for the action cutoffs.
- Use `policy_markers.csv` as a second source if you want labeled markers. A simple reference line per scenario is cleaner.

**Regret curve**:

- Same axes, `regret_pct` on rows.
- Keep the y-axis at 0–12%; the comparison is the shape, not the absolute dollar level.
- Annotate the worst-case point for each scenario.

### Right: evidence

- **Bootstrap interval** from `bootstrap.csv`: a dot plot of `observed_difference_millions` with `ci_low_millions`/`ci_high_millions` as a range. Filter `comparison = action_vs_binary`.
- **Power curve** from `power.csv`: `n_accounts_per_arm` vs `power_pct`, with a reference line at 80% and a marker at 17,610.
- **Action mix** from `action_mix.csv`: a 100% stacked bar by `scenario`, using `share_pct`; tooltip includes `bad_rate_pct`.

### Bottom: calibration and drift

- **Calibration** from `calibration.csv`: `mean_predicted_pct` vs `observed_pct`, color `method`, with a 45-degree reference line. Filter to `method = isotonic` for the headline; show raw and Platt in a secondary sheet.
- **Drift** from `drift.csv`: top 20 features by `psi`, color by `status` (green/yellow/red), with reference lines at 0.10 and 0.25.

## 5. Design rules

- **No pie charts, no 3D, no dual axes with unrelated scales.**
- One color per scenario: Conservative `#4C78A8`, Moderate `#F2A541`, Aggressive `#2F6B4F`, Stress `#9B3A32`.
- Grow/Hold/Restrict use `#2F6B4F` / `#2F5D7C` / `#9B3A32` only in policy visuals.
- Use a light grid, hairline axes, and one annotation per chart.
- Put the assumption label on every profit view: **“Simulated expected profit; not actual P&L.”**
- Format currency as `$#,##0.0"M"` and percentages as `0.0"%"`.
- Keep the dashboard at 1200 × 900; avoid scrollbars in the main story.

## 6. Publish to Tableau Public

1. **File → Save to Tableau Public As…** and sign in.
2. Name the workbook `Breakpoint — Three-Action Credit Policy`.
3. Tableau Public publishes the workbook and its extract. The data is aggregate; no customer-level records are included.
4. Copy the public URL.
5. Add the URL to:
   - `README.md` (reviewer paths),
   - the GitHub repository homepage,
   - `docs/RESULTS.md` if you want a direct link.
6. Take a screenshot of the published dashboard and add it to `docs/assets/` only if you want the static demo to show it. The static demo does **not** embed Tableau; it is the no-account fallback.

## 7. Refresh

When the reports change:

```bash
make tableau          # regenerate the 12 CSVs + manifest
# In Tableau: Data → Refresh Extract, then File → Save to Tableau Public
```

The manifest's `generated_at` field records the export time. If you publish, mention the report date in the dashboard subtitle.

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| Scenario colors inconsistent | Use `scenario_label` as the color dimension, not `scenario`; set the same color mapping in every sheet. |
| Profit curve looks flat | The action region is below 10% PD. Use a log x-axis or filter `probability_cutoff <= 0.1`; do not clip the axis to hide the low-PD region. |
| Regret values look huge | Check that you are using `regret_pct` (already ×100), not the raw fraction. |
| `p_value` shows 0.000 | Format as `0.000;<0.001`; the bootstrap resolution is 1/1,000. |
| CSV dates/numbers import as text | Set the data source locale to English (United States) and refresh. |
| Tableau Public rejects the extract | Confirm the workbook uses an extract, not a live local CSV connection. |
| Dashboard shows customer data | Stop. The Tableau data should be aggregate only. Re-run `make tableau` and inspect `manifest.json`. |
