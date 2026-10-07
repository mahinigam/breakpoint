import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_regret_map(thresholds, scenario_profits, output_dir):
    """
    Plots a heatmap of fractional regret across cutoffs and economic scenarios.
    Regret(τ, r) = OracleProfit(r) − ActualProfit(τ, r)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    scenarios = list(scenario_profits.keys())
    
    # We construct a matrix: Scenarios (rows) x Thresholds (cols)
    regret_matrix = []
    
    for name in scenarios:
        data = scenario_profits[name]
        max_p = data["max_profit"]
        profits = data["profits"]
        
        # Fractional regret
        regret = (max_p - profits) / max_p
        regret_matrix.append(regret)
        
    regret_matrix = np.array(regret_matrix)
    
    # Downsample thresholds for the heatmap so it's readable
    n_ticks = 10
    idx_to_plot = np.linspace(0, len(thresholds)-1, n_ticks, dtype=int)
    
    plot_matrix = regret_matrix[:, idx_to_plot]
    plot_thresholds = thresholds[idx_to_plot]
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        plot_matrix, 
        annot=True, 
        fmt=".1%", 
        cmap="YlOrRd", 
        xticklabels=[f"{t:.2f}" for t in plot_thresholds],
        yticklabels=scenarios
    )
    plt.title("Regret Map: Fractional Profit Loss by Scenario")
    plt.xlabel("Probability Threshold")
    plt.ylabel("Economic Scenario")
    plt.tight_layout()
    
    plt.savefig(output_dir / "regret_map.png", dpi=300)
    plt.close()
    
    return regret_matrix
