import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import os
from src.decision.optimal_thresholds import compute_profit, get_f1_optimal_threshold, get_ks_optimal_threshold, get_top_pct_threshold

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

with open(ROOT_DIR / "config" / "settings.yaml", "r") as f:
    NEGATIVE_WEIGHT = yaml.safe_load(f)["negative_weight"]

with open(ROOT_DIR / "config" / "economic_scenarios.yaml", "r") as f:
    scenarios = yaml.safe_load(f)["scenarios"]

def bootstrap_profit_diff(y_true, y_prob, weights, thresh_A, thresh_B, margin, loss, n_iterations=1000, seed=42):
    """Computes bootstrap confidence intervals for the profit difference between two thresholds."""
    np.random.seed(seed)
    n = len(y_true)
    diffs = []
    
    # Pre-calculate individual profits to speed up bootstrap
    # Profit per account = w_i * (1[y_pred=good]*margin - 1[y_pred=good & y_true=default]*loss)
    pred_A = (y_prob <= thresh_A) # True if classified as good
    pred_B = (y_prob <= thresh_B)
    
    profit_A_individual = weights * (pred_A * margin - pred_A * y_true * (margin + loss))
    profit_B_individual = weights * (pred_B * margin - pred_B * y_true * (margin + loss))
    
    diff_individual = profit_A_individual - profit_B_individual
    
    for _ in range(n_iterations):
        indices = np.random.randint(0, n, size=n)
        resampled_diff = np.sum(diff_individual[indices])
        diffs.append(resampled_diff)
        
    diffs = np.array(diffs)
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    
    return diffs, ci_lower, ci_upper

def main():
    oof_path = ROOT_DIR / "data" / "processed" / "oof_predictions.parquet"
    df = pd.read_parquet(oof_path)
    
    y_true = df["target"].values
    y_prob = df["lgb_prob"].values
    
    weights = np.ones(len(y_true))
    weights[y_true == 0] = NEGATIVE_WEIGHT
    
    # We will compare MiniMax (0.0500) vs F1-optimal (0.2585) under the Stress scenario
    thresh_minimax = 0.0500
    thresh_f1 = get_f1_optimal_threshold(y_true, y_prob, NEGATIVE_WEIGHT)
    
    stress = scenarios["stress"]
    margin = stress["margin"]
    loss = stress["lgd"] * stress["exposure"]
    
    print(f"Bootstrapping profit difference (MiniMax vs F1) under Stress scenario...")
    print(f"Margin: ${margin}, Loss: ${loss}")
    
    diffs, ci_lower, ci_upper = bootstrap_profit_diff(
        y_true, y_prob, weights, thresh_minimax, thresh_f1, margin, loss, n_iterations=1000
    )
    
    mean_diff = np.mean(diffs)
    print(f"Mean Profit Advantage of MiniMax over F1: ${mean_diff:,.2f}")
    print(f"95% Confidence Interval: [${ci_lower:,.2f}, ${ci_upper:,.2f}]")
    
    # Plot distribution
    plt.figure(figsize=(8, 5))
    plt.hist(diffs / 1e6, bins=50, alpha=0.7, color='steelblue')
    plt.axvline(ci_lower / 1e6, color='red', linestyle='--', label='95% CI')
    plt.axvline(ci_upper / 1e6, color='red', linestyle='--')
    plt.axvline(mean_diff / 1e6, color='black', linestyle='-', label='Mean Difference')
    
    plt.title("Bootstrap Distribution of Profit Advantage (MiniMax vs F1)")
    plt.xlabel("Profit Advantage (Millions $)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(ROOT_DIR / "figures", exist_ok=True)
    plt.savefig(ROOT_DIR / "figures" / "bootstrap_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
