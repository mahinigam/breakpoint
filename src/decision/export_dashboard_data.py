import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from src.decision.optimal_thresholds import compute_profit, get_f1_optimal_threshold, get_ks_optimal_threshold, get_top_pct_threshold

# Project config
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
with open(ROOT_DIR / "config" / "settings.yaml", "r") as f:
    NEGATIVE_WEIGHT = yaml.safe_load(f)["negative_weight"]

with open(ROOT_DIR / "config" / "economic_scenarios.yaml", "r") as f:
    scenarios = yaml.safe_load(f)["scenarios"]

def export_tableau_data():
    oof_path = ROOT_DIR / "data" / "processed" / "oof_predictions.parquet"
    if not oof_path.exists():
        print(f"Error: {oof_path} not found.")
        return
        
    df = pd.read_parquet(oof_path)
    y_true = df["target"].values
    y_prob = df["lgb_prob"].values
    
    thresholds = np.linspace(0.01, 0.99, 99)
    
    # Pre-calculate heuristic thresholds for marking in Tableau
    t_f1 = get_f1_optimal_threshold(y_true, y_prob, NEGATIVE_WEIGHT)
    t_ks = get_ks_optimal_threshold(y_true, y_prob, NEGATIVE_WEIGHT)
    t_4pct = get_top_pct_threshold(y_true, y_prob, NEGATIVE_WEIGHT, pct=0.04)
    
    records = []
    
    for name, params in scenarios.items():
        loss = params["lgd"] * params["exposure"]
        margin = params["margin"]
        r = loss / margin
        
        profits = [compute_profit(y_true, y_prob, t, margin, loss, NEGATIVE_WEIGHT) for t in thresholds]
        max_profit = max(profits)
        
        for i, t in enumerate(thresholds):
            profit = profits[i]
            regret_dollar = max_profit - profit
            regret_pct = regret_dollar / max_profit if max_profit > 0 else 0
            
            # Identify if this threshold matches a heuristic (closest)
            heuristic_flag = "None"
            if abs(t - t_f1) < 0.005: heuristic_flag = "F1 Optimal"
            elif abs(t - t_ks) < 0.005: heuristic_flag = "KS Optimal"
            elif abs(t - t_4pct) < 0.005: heuristic_flag = "Top 4% Reject"
            elif abs(t - 0.05) < 0.005: heuristic_flag = "MiniMax Optimal"
            
            records.append({
                "Scenario": name,
                "Cost Ratio (L/M)": r,
                "Threshold": t,
                "Profit": profit,
                "Max Possible Profit": max_profit,
                "Regret ($)": regret_dollar,
                "Regret (%)": regret_pct,
                "Heuristic Marker": heuristic_flag
            })
            
    out_df = pd.DataFrame(records)
    out_path = ROOT_DIR / "data" / "processed" / "tableau_dashboard_data.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Exported Tableau dashboard data to {out_path}")

if __name__ == "__main__":
    export_tableau_data()
