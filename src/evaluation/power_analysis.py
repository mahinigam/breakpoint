import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from src.decision.optimal_thresholds import get_f1_optimal_threshold

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

with open(ROOT_DIR / "config" / "settings.yaml", "r") as f:
    NEGATIVE_WEIGHT = yaml.safe_load(f)["negative_weight"]

with open(ROOT_DIR / "config" / "economic_scenarios.yaml", "r") as f:
    scenarios = yaml.safe_load(f)["scenarios"]

def simulate_power(y_true, y_prob, weights, thresh_A, thresh_B, margin, loss, n_accounts=50000, n_sims=1000, alpha=0.05):
    """Simulates Champion-Challenger offline A/B test power."""
    np.random.seed(42)
    
    n_total = len(y_true)
    
    # Calculate per-account profit
    pred_A = (y_prob <= thresh_A)
    pred_B = (y_prob <= thresh_B)
    
    # We only care about the unweighted profit per account for an actual future test.
    # Wait, the dataset is subsampled. A live test would see the TRUE distribution (1.6% defaults).
    # So we must sample from our dataset with probabilities proportional to `weights` to simulate a live test!
    
    sampling_probs = weights / np.sum(weights)
    
    profit_A_individual = (pred_A * margin - pred_A * y_true * (margin + loss))
    profit_B_individual = (pred_B * margin - pred_B * y_true * (margin + loss))
    
    successes = 0
    
    for _ in range(n_sims):
        # Sample n_accounts for arm A and arm B according to true population weights
        idx_A = np.random.choice(n_total, size=n_accounts, p=sampling_probs, replace=True)
        idx_B = np.random.choice(n_total, size=n_accounts, p=sampling_probs, replace=True)
        
        profit_A = np.sum(profit_A_individual[idx_A])
        profit_B = np.sum(profit_B_individual[idx_B])
        
        mean_A = np.mean(profit_A_individual[idx_A])
        mean_B = np.mean(profit_B_individual[idx_B])
        
        var_A = np.var(profit_A_individual[idx_A])
        var_B = np.var(profit_B_individual[idx_B])
        
        # Z-test for difference in means
        se = np.sqrt(var_A/n_accounts + var_B/n_accounts)
        z = (mean_A - mean_B) / se
        
        # If A is significantly better than B
        if z > 1.645:  # one-sided alpha=0.05
            successes += 1
            
    return successes / n_sims

def main():
    oof_path = ROOT_DIR / "data" / "processed" / "oof_predictions.parquet"
    df = pd.read_parquet(oof_path)
    
    y_true = df["target"].values
    y_prob = df["lgb_prob"].values
    
    weights = np.ones(len(y_true))
    weights[y_true == 0] = NEGATIVE_WEIGHT
    
    thresh_minimax = 0.0500
    thresh_f1 = get_f1_optimal_threshold(y_true, y_prob, NEGATIVE_WEIGHT)
    
    stress = scenarios["stress"]
    margin = stress["margin"]
    loss = stress["lgd"] * stress["exposure"]
    
    # We want to find n_accounts required to reach 80% power.
    print("Running Simulation-based Power Analysis for Champion-Challenger Test...")
    print("Champion: MiniMax (0.0500)")
    print("Challenger: F1-Optimal (0.2585)")
    print("Scenario: Stress")
    
    sample_sizes = [1000, 2500, 5000, 10000, 25000, 50000]
    powers = []
    
    for n in sample_sizes:
        power = simulate_power(y_true, y_prob, weights, thresh_minimax, thresh_f1, margin, loss, n_accounts=n, n_sims=500)
        powers.append(power)
        print(f"Accounts per arm: {n:,} => Simulated Power: {power*100:.1f}%")
        
    print("\nIf volume is 100,000 applications per month, a 10,000 per arm test takes ~1 week.")
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(sample_sizes, [p * 100 for p in powers], marker='o', linestyle='-', color='teal', linewidth=2)
    plt.axhline(y=80, color='r', linestyle='--', label='80% Power Threshold')
    plt.title("Simulation-based Power Curve (Stress Scenario)")
    plt.xlabel("Sample Size (Accounts per Arm)")
    plt.ylabel("Statistical Power (%)")
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT_DIR / "figures" / "power_curve.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
