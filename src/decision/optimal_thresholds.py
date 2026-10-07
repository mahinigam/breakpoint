import pandas as pd
import numpy as np
import yaml
from pathlib import Path

from src.economics import (
    compute_profit, plot_profit_curves,
    get_ks_optimal_threshold, get_f1_optimal_threshold, get_top_pct_threshold,
    get_fixed_threshold, get_profit_optimal_threshold,
    plot_regret_map, plot_sensitivity_sweep
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
with open(ROOT_DIR / "config" / "settings.yaml", "r") as f:
    settings = yaml.safe_load(f)
NEGATIVE_WEIGHT = settings["negative_weight"]

with open(ROOT_DIR / "config" / "economic_scenarios.yaml", "r") as f:
    scenarios = yaml.safe_load(f)["scenarios"]

def main():
    print("Loading OOF predictions...")
    oof_path = ROOT_DIR / "data" / "processed" / "oof_predictions.parquet"
    if not oof_path.exists():
        print(f"Error: {oof_path} not found. Run 'make train' first.")
        return
        
    df = pd.read_parquet(oof_path)
    y_true = df["target"].values
    y_prob = df["lgb_prob"].values
    
    print("\n--- Heuristic Thresholds ---")
    t_f1 = get_f1_optimal_threshold(y_true, y_prob, NEGATIVE_WEIGHT)
    t_ks = get_ks_optimal_threshold(y_true, y_prob, NEGATIVE_WEIGHT)
    t_4pct = get_top_pct_threshold(y_true, y_prob, NEGATIVE_WEIGHT, pct=0.04)
    t_fixed = get_fixed_threshold(y_true, y_prob, NEGATIVE_WEIGHT, threshold=0.5)
    
    print(f"F1 Optimal Threshold: {t_f1:.4f}")
    print(f"KS Optimal Threshold: {t_ks:.4f}")
    print(f"Top-4% Reject Threshold: {t_4pct:.4f}")
    print(f"Fixed 0.5 Threshold: {t_fixed:.4f}")
    
    print("\n--- Economic Scenarios ---")
    scenario_profits = {}
    thresholds = np.linspace(0.01, 0.99, 99)
    
    for name, params in scenarios.items():
        loss = params["lgd"] * params["exposure"]
        margin = params["margin"]
        r = loss / margin
        print(f"Scenario '{name}': Margin=${margin}, Loss=${loss:.0f}, Ratio={r:.1f}")
        
        theoretical_opt = get_profit_optimal_threshold(y_true, y_prob, NEGATIVE_WEIGHT, margin, loss)
        
        profits = [compute_profit(y_true, y_prob, t, margin, loss, NEGATIVE_WEIGHT) for t in thresholds]
        opt_idx = np.argmax(profits)
        empirical_opt = thresholds[opt_idx]
        max_profit = profits[opt_idx]
        
        print(f"  -> Theoretical Opt: {theoretical_opt:.4f}")
        print(f"  -> Empirical Opt:   {empirical_opt:.4f} (Max Profit: ${max_profit:,.0f})")
        
        scenario_profits[name] = {
            "profits": np.array(profits),
            "max_profit": max_profit,
            "margin": margin,
            "loss": loss,
            "empirical_opt": empirical_opt,
            "thresholds": thresholds
        }
    
    # 1. Plot profit curves
    plot_profit_curves(scenario_profits, ROOT_DIR / "figures")
    
    print("\n--- Evaluating Sub-optimality of Heuristics ---")
    for name, data in scenario_profits.items():
        opt = data["max_profit"]
        p_f1 = compute_profit(y_true, y_prob, t_f1, data["margin"], data["loss"], NEGATIVE_WEIGHT)
        p_ks = compute_profit(y_true, y_prob, t_ks, data["margin"], data["loss"], NEGATIVE_WEIGHT)
        p_4pct = compute_profit(y_true, y_prob, t_4pct, data["margin"], data["loss"], NEGATIVE_WEIGHT)
        p_fixed = compute_profit(y_true, y_prob, t_fixed, data["margin"], data["loss"], NEGATIVE_WEIGHT)
        
        print(f"Scenario '{name}':")
        print(f"  KS Loss:     ${opt - p_ks:,.0f} ({((opt - p_ks)/opt)*100:.1f}%)")
        print(f"  F1 Loss:     ${opt - p_f1:,.0f} ({((opt - p_f1)/opt)*100:.1f}%)")
        print(f"  4% Loss:     ${opt - p_4pct:,.0f} ({((opt - p_4pct)/opt)*100:.1f}%)")
        print(f"  Fixed 0.5 Loss: ${opt - p_fixed:,.0f} ({((opt - p_fixed)/opt)*100:.1f}%)")
        
    print("\n--- MiniMax Regret Policy ---")
    # 2. Plot regret map and get MiniMax
    regret_matrix = plot_regret_map(thresholds, scenario_profits, ROOT_DIR / "figures")
    
    max_regrets = np.max(regret_matrix, axis=0)
    minimax_idx = np.argmin(max_regrets)
    t_minimax = thresholds[minimax_idx]
    
    print(f"Minimax Regret Threshold: {t_minimax:.4f} (Max fractional regret: {max_regrets[minimax_idx]*100:.1f}%)")

    print("\nCompare MiniMax against others in the worst-case:")
    for heuristic_name, t_h in [("Fixed 0.5", t_fixed), ("KS", t_ks), ("F1", t_f1), ("4%", t_4pct), ("MiniMax", t_minimax)]:
        max_regret = 0.0
        worst_scenario = ""
        for name, data in scenario_profits.items():
            prof = compute_profit(y_true, y_prob, t_h, data["margin"], data["loss"], NEGATIVE_WEIGHT)
            reg = (data["max_profit"] - prof) / data["max_profit"]
            if reg > max_regret:
                max_regret = reg
                worst_scenario = name
        print(f"  {heuristic_name} worst-case regret: {max_regret*100:.1f}% (in {worst_scenario})")
        
    # 3. Plot Sensitivity Sweep
    print("\n--- Running Sensitivity Sweep (r from 1 to 50) ---")
    plot_sensitivity_sweep(y_true, y_prob, NEGATIVE_WEIGHT, ROOT_DIR / "figures")

if __name__ == "__main__":
    main()
