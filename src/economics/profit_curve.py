import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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

def plot_profit_curves(scenario_profits, output_dir):
    """
    Plots profit curves for all scenarios.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, data in scenario_profits.items():
        plt.figure(figsize=(10, 6))
        thresholds = data["thresholds"]
        profits = data["profits"]
        
        plt.plot(thresholds, profits, label=f"{name} Scenario", color='blue', linewidth=2)
        plt.axvline(x=data["empirical_opt"], color='red', linestyle='--', label=f'Optimal Cutoff ({data["empirical_opt"]:.3f})')
        
        plt.title(f"Profit Curve: {name} Scenario (r = {data['loss']/data['margin']:.1f})")
        plt.xlabel("Probability Cutoff (Threshold)")
        plt.ylabel("Total Expected Profit ($)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        filename = f"profit_curve_{name.lower().replace(' ', '_')}.png"
        plt.savefig(output_dir / filename, dpi=300)
        plt.close()
