import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.metrics import f1_score, precision_recall_curve, roc_curve

# Project config
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
with open(ROOT_DIR / "config" / "settings.yaml", "r") as f:
    settings = yaml.safe_load(f)

NEGATIVE_WEIGHT = settings["negative_weight"]

with open(ROOT_DIR / "config" / "economic_scenarios.yaml", "r") as f:
    scenarios = yaml.safe_load(f)["scenarios"]

def compute_profit(y_true, y_prob, threshold, margin, loss, neg_weight):
    """
    Computes total profit at a given threshold.
    Approvals are y_prob < threshold.
    """
    approved = (y_prob < threshold)
    
    # In the true population, good accounts (y=0) were subsampled by 1/20, 
    # so we multiply their impact by neg_weight (20).
    profit_good = np.sum((y_true[approved] == 0)) * margin * neg_weight
    profit_bad = np.sum((y_true[approved] == 1)) * (-loss)
    
    return profit_good + profit_bad

def get_ks_optimal_threshold(y_true, y_prob, neg_weight):
    """Returns the threshold that maximizes the KS statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, sample_weight=np.where(y_true==0, neg_weight, 1.0))
    ks_stat = tpr - fpr
    idx = np.argmax(ks_stat)
    return thresholds[idx]

def get_f1_optimal_threshold(y_true, y_prob, neg_weight):
    """Returns threshold that maximizes F1 score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob, sample_weight=np.where(y_true==0, neg_weight, 1.0))
    # precision and recall have 1 more element than thresholds
    f1 = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-15)
    idx = np.argmax(f1)
    return thresholds[idx]

def get_top_pct_threshold(y_true, y_prob, neg_weight, pct=0.04):
    """Returns threshold that rejects exactly pct% of the volume."""
    weights = np.where(y_true == 0, neg_weight, 1.0)
    total_weight = np.sum(weights)
    
    # Sort by probability descending (highest risk first)
    indices = np.argsort(y_prob)[::-1]
    weights_sorted = weights[indices]
    
    cum_weights = np.cumsum(weights_sorted)
    cutoff_weight = pct * total_weight
    cutoff_idx = np.searchsorted(cum_weights, cutoff_weight)
    
    return y_prob[indices[cutoff_idx]]

def main():
    print("Loading OOF predictions...")
    oof_path = ROOT_DIR / "data" / "processed" / "oof_predictions.parquet"
    if not oof_path.exists():
        print(f"Error: {oof_path} not found. Run 'make train' first.")
        return
        
    df = pd.read_parquet(oof_path)
    y_true = df["target"].values
    
    # We will do decisioning on the best model, typically LightGBM
    y_prob = df["lgb_prob"].values
    
    print("\n--- Heuristic Thresholds ---")
    t_f1 = get_f1_optimal_threshold(y_true, y_prob, NEGATIVE_WEIGHT)
    t_ks = get_ks_optimal_threshold(y_true, y_prob, NEGATIVE_WEIGHT)
    t_4pct = get_top_pct_threshold(y_true, y_prob, NEGATIVE_WEIGHT, pct=0.04)
    
    print(f"F1 Optimal Threshold: {t_f1:.4f}")
    print(f"KS Optimal Threshold: {t_ks:.4f}")
    print(f"Top-4% Reject Threshold: {t_4pct:.4f}")
    
    print("\n--- Economic Scenarios ---")
    scenario_profits = {}
    thresholds = np.linspace(0.01, 0.99, 99)
    
    for name, params in scenarios.items():
        loss = params["lgd"] * params["exposure"]
        margin = params["margin"]
        r = loss / margin
        print(f"Scenario '{name}': Margin=${margin}, Loss=${loss:.0f}, Ratio={r:.1f}")
        
        # Calculate theoretical optimal threshold
        # Assuming perfectly calibrated probabilities:
        # Expected profit = (1-p)*M - p*L = 0 -> p = M / (M + L) = 1 / (1 + r)
        theoretical_opt = 1.0 / (1.0 + r)
        
        # Calculate empirical profit curve
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
            "loss": loss
        }
        
    print("\n--- Evaluating Sub-optimality of Heuristics ---")
    for name, data in scenario_profits.items():
        opt = data["max_profit"]
        p_f1 = compute_profit(y_true, y_prob, t_f1, data["margin"], data["loss"], NEGATIVE_WEIGHT)
        p_ks = compute_profit(y_true, y_prob, t_ks, data["margin"], data["loss"], NEGATIVE_WEIGHT)
        p_4pct = compute_profit(y_true, y_prob, t_4pct, data["margin"], data["loss"], NEGATIVE_WEIGHT)
        
        print(f"Scenario '{name}':")
        print(f"  KS Loss:     ${opt - p_ks:,.0f} ({((opt - p_ks)/opt)*100:.1f}%)")
        print(f"  F1 Loss:     ${opt - p_f1:,.0f} ({((opt - p_f1)/opt)*100:.1f}%)")
        print(f"  4% Loss:     ${opt - p_4pct:,.0f} ({((opt - p_4pct)/opt)*100:.1f}%)")
        
    print("\n--- MiniMax Regret Policy ---")
    # For each threshold, max regret across all scenarios
    # Regret = Max Profit for Scenario - Profit(t) for Scenario
    
    # We must normalize the regret, otherwise scenarios with larger nominal dollars dominate.
    # We normalize by the maximum possible profit in that scenario (Fractional Regret).
    
    regrets = []
    for i, t in enumerate(thresholds):
        max_scenario_regret = 0.0
        for name, data in scenario_profits.items():
            regret = data["max_profit"] - data["profits"][i]
            frac_regret = regret / data["max_profit"]
            if frac_regret > max_scenario_regret:
                max_scenario_regret = frac_regret
        regrets.append(max_scenario_regret)
        
    regrets = np.array(regrets)
    minimax_idx = np.argmin(regrets)
    t_minimax = thresholds[minimax_idx]
    
    print(f"Minimax Regret Threshold: {t_minimax:.4f} (Max fractional regret: {regrets[minimax_idx]*100:.1f}%)")

    print("\nCompare MiniMax against others in the worst-case:")
    for heuristic_name, t_h in [("KS", t_ks), ("F1", t_f1), ("4%", t_4pct), ("MiniMax", t_minimax)]:
        max_regret = 0.0
        worst_scenario = ""
        for name, data in scenario_profits.items():
            prof = compute_profit(y_true, y_prob, t_h, data["margin"], data["loss"], NEGATIVE_WEIGHT)
            reg = (data["max_profit"] - prof) / data["max_profit"]
            if reg > max_regret:
                max_regret = reg
                worst_scenario = name
        print(f"  {heuristic_name} worst-case regret: {max_regret*100:.1f}% (in {worst_scenario})")

if __name__ == "__main__":
    main()
