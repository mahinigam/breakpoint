# Tableau Executive Dashboard Guide

This guide explains how to construct the "Profit-Optimal Credit Decisioning" executive dashboard using the exported data.

## 1. Data Source
- **File:** `data/processed/tableau_dashboard_data.csv`
- **Granularity:** 1 row per Threshold (0.01 to 0.99) per Economic Scenario (Conservative, Moderate, Aggressive, Stress).

## 2. Dashboard Layout & Visualizations

### Viz 1: Expected Profit Curves (Line Chart)
- **Objective:** Show how profit varies with the chosen approval threshold across different economies.
- **X-Axis (Columns):** `Threshold` (Continuous Dimension)
- **Y-Axis (Rows):** `Profit` (Continuous Measure, formatted as Currency)
- **Color:** `Scenario`
- **Design Tip:** Add a vertical reference line at the MiniMax Threshold (`0.0500`) to visually demonstrate where we are locking in our policy.

### Viz 2: Regret / "Money Left on the Table" (Line Chart)
- **Objective:** Show the danger of choosing the wrong threshold.
- **X-Axis (Columns):** `Threshold` (Continuous Dimension)
- **Y-Axis (Rows):** `Regret (%)` (Continuous Measure, formatted as Percentage)
- **Color:** `Scenario`
- **Interpretation:** The lowest point on each curve is the optimal threshold for that specific scenario. The MiniMax threshold is the point where the highest curve (the worst-case scenario) is at its absolute minimum.

### Viz 3: Heuristic Sub-optimality (Bar Chart / Scatter)
- **Objective:** Visually "grill" the standard statistical cutoffs.
- **Filter:** `Heuristic Marker` exclude "None"
- **X-Axis (Columns):** `Scenario`
- **Y-Axis (Rows):** `Regret (%)`
- **Color/Shape:** `Heuristic Marker` (KS Optimal, F1 Optimal, Top 4%, MiniMax Optimal)
- **Interpretation:** This chart will starkly show that F1 Regret spikes to 31.8% in the Stress scenario, KS Regret spikes to 10.9% in the Aggressive scenario, but MiniMax Regret never exceeds 4.7%.

## 3. Interactive Elements for the Interview
When defending this dashboard in an interview, set it up as a story:
1. **The Hook:** Filter to the "Aggressive" scenario and ask, *"If we were in a growth market, which threshold looks best?"* (Answer: F1).
2. **The Twist:** Change the filter to "Stress" scenario. Watch the F1 threshold's profit plummet.
3. **The Solution:** Show the Regret (%) chart with all scenarios enabled, highlighting the MiniMax vertical line that bounds the maximum risk.

## 4. Aesthetics & Branding
- **Colors:** Use a premium, corporate color palette. 
  - Stress: Dark Red (#8B0000)
  - Conservative: Orange (#FF8C00)
  - Moderate: Steel Blue (#4682B4)
  - Aggressive: Forest Green (#228B22)
  - MiniMax Reference Line: Dashed Black with an annotation.
- **Tooltips:** Clean up tooltips to explicitly state: *"At a threshold of [Threshold], in a [Scenario] economy, we leave [Regret %] of maximum profit on the table."*
