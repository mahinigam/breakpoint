import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .profit_curve import compute_profit
from .cutoff_methods import get_ks_optimal_threshold, get_f1_optimal_threshold, get_profit_optimal_threshold

def plot_sensitivity_sweep(y_true, y_prob, neg_weight, output_dir):
    """
    Sweeps the cost ratio r = Loss/Margin from 1 to 50.
    Plots the fractional profit captured by F1, KS, and MiniMax policies.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    r_values = np.linspace(1, 50, 50)
    
    # We need fixed thresholds for F1 and KS, which do not depend on r
    t_f1 = get_f1_optimal_threshold(y_true, y_prob, neg_weight)
    t_ks = get_ks_optimal_threshold(y_true, y_prob, neg_weight)
    
    # Let's say we have a pre-computed minimax from the scenarios
    # For this sweep, we will compute the Minimax dynamically across the r range
    thresholds = np.linspace(0.01, 0.99, 99)
    
    # Precompute profit matrix: r_idx x t_idx
    profit_matrix = np.zeros((len(r_values), len(thresholds)))
    max_profits = np.zeros(len(r_values))
    
    # Base margin = 100 for calculation simplicity, loss = r * margin
    margin = 100
    
    for i, r in enumerate(r_values):
        loss = r * margin
        profits = np.array([compute_profit(y_true, y_prob, t, margin, loss, neg_weight) for t in thresholds])
        profit_matrix[i, :] = profits
        max_profits[i] = np.max(profits)
        
    # Find minimax across this continuous sweep
    regret_matrix = (max_profits[:, None] - profit_matrix) / max_profits[:, None]
    max_regrets = np.max(regret_matrix, axis=0) # Max regret for each threshold across all r
    minimax_idx = np.argmin(max_regrets)
    t_minimax = thresholds[minimax_idx]
    
    f1_captured = []
    ks_captured = []
    minimax_captured = []
    
    for i, r in enumerate(r_values):
        loss = r * margin
        max_p = max_profits[i]
        
        p_f1 = compute_profit(y_true, y_prob, t_f1, margin, loss, neg_weight)
        p_ks = compute_profit(y_true, y_prob, t_ks, margin, loss, neg_weight)
        p_minimax = compute_profit(y_true, y_prob, t_minimax, margin, loss, neg_weight)
        
        f1_captured.append(p_f1 / max_p)
        ks_captured.append(p_ks / max_p)
        minimax_captured.append(p_minimax / max_p)
        
    plt.figure(figsize=(10, 6))
    plt.plot(r_values, [c * 100 for c in f1_captured], label=f'F1 Policy (t={t_f1:.3f})', linestyle='--')
    plt.plot(r_values, [c * 100 for c in ks_captured], label=f'KS Policy (t={t_ks:.3f})', linestyle='-.')
    plt.plot(r_values, [c * 100 for c in minimax_captured], label=f'MiniMax Policy (t={t_minimax:.3f})', color='black', linewidth=2)
    
    plt.title("Sensitivity Sweep: % of Optimal Profit Captured across Cost Ratios")
    plt.xlabel("Cost Ratio r (Loss / Margin)")
    plt.ylabel("% of Maximum Possible Profit")
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_dir / "sensitivity_sweep.png", dpi=300)
    plt.close()
    
    return t_minimax
