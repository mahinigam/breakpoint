import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import os

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

with open(ROOT_DIR / "config" / "settings.yaml", "r") as f:
    config = yaml.safe_load(f)
    NEGATIVE_WEIGHT = config["negative_weight"]

def compute_ece(y_true, y_prob, weights, n_bins=10):
    """Computes Expected Calibration Error (weighted)."""
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    ece = 0.0
    total_weight = np.sum(weights)
    
    for i in range(n_bins):
        mask = binids == i
        if not np.any(mask):
            continue
            
        bin_weight = np.sum(weights[mask])
        
        # Weighted mean predicted probability
        prob_mean = np.average(y_prob[mask], weights=weights[mask])
        
        # Weighted empirical fraction of positives
        true_fraction = np.average(y_true[mask], weights=weights[mask])
        
        ece += (bin_weight / total_weight) * np.abs(prob_mean - true_fraction)
        
    return ece

def plot_reliability_diagram(y_true, y_prob_dict, weights, title, filename, n_bins=10):
    """Plots a reliability diagram."""
    plt.figure(figsize=(8, 6))
    
    for label, y_prob in y_prob_dict.items():
        bins = np.linspace(0., 1., n_bins + 1)
        binids = np.digitize(y_prob, bins) - 1
        
        prob_means = []
        true_fractions = []
        
        for i in range(n_bins):
            mask = binids == i
            if not np.any(mask):
                continue
            prob_means.append(np.average(y_prob[mask], weights=weights[mask]))
            true_fractions.append(np.average(y_true[mask], weights=weights[mask]))
            
        plt.plot(prob_means, true_fractions, marker='o', label=label)
        
    plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Empirical Fraction of Positives')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(ROOT_DIR / "figures", exist_ok=True)
    plt.savefig(ROOT_DIR / "figures" / filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    oof_path = ROOT_DIR / "data" / "processed" / "oof_predictions.parquet"
    if not oof_path.exists():
        print(f"Error: {oof_path} not found.")
        return
        
    df = pd.read_parquet(oof_path)
    y_true = df["target"].values
    y_prob_raw = df["lgb_prob"].values
    
    weights = np.ones(len(y_true))
    weights[y_true == 0] = NEGATIVE_WEIGHT
    
    # 5-fold CV for calibration to prevent overfitting
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    y_prob_platt = np.zeros_like(y_prob_raw)
    y_prob_iso = np.zeros_like(y_prob_raw)
    
    print("Fitting calibration models...")
    for train_idx, val_idx in skf.split(y_prob_raw, y_true):
        X_train = y_prob_raw[train_idx].reshape(-1, 1)
        y_train = y_true[train_idx]
        w_train = weights[train_idx]
        
        X_val = y_prob_raw[val_idx].reshape(-1, 1)
        
        # Platt Scaling (Logistic Regression)
        lr = LogisticRegression(solver='lbfgs')
        lr.fit(X_train, y_train, sample_weight=w_train)
        y_prob_platt[val_idx] = lr.predict_proba(X_val)[:, 1]
        
        # Isotonic Regression
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        iso.fit(X_train.ravel(), y_train, sample_weight=w_train)
        y_prob_iso[val_idx] = iso.predict(X_val.ravel())
        
    # Evaluate Brier Score
    brier_raw = brier_score_loss(y_true, y_prob_raw, sample_weight=weights)
    brier_platt = brier_score_loss(y_true, y_prob_platt, sample_weight=weights)
    brier_iso = brier_score_loss(y_true, y_prob_iso, sample_weight=weights)
    
    print(f"Brier Score (Raw):      {brier_raw:.5f}")
    print(f"Brier Score (Platt):    {brier_platt:.5f}")
    print(f"Brier Score (Isotonic): {brier_iso:.5f}")
    
    # Evaluate ECE
    ece_raw = compute_ece(y_true, y_prob_raw, weights)
    ece_platt = compute_ece(y_true, y_prob_platt, weights)
    ece_iso = compute_ece(y_true, y_prob_iso, weights)
    
    print(f"ECE (Raw):      {ece_raw:.5f}")
    print(f"ECE (Platt):    {ece_platt:.5f}")
    print(f"ECE (Isotonic): {ece_iso:.5f}")
    
    # Plot Reliability Diagrams
    unweighted = np.ones(len(y_true))
    plot_reliability_diagram(y_true, {"Raw (LGBM)": y_prob_raw}, unweighted, 
                             "Unweighted Reliability Diagram (Shows apparent calibration)", "calibration_unweighted.png")
                             
    plot_reliability_diagram(y_true, {
        "Raw (LGBM)": y_prob_raw, 
        "Platt Scaling": y_prob_platt, 
        "Isotonic": y_prob_iso
    }, weights, "Weighted Reliability Diagram (True Population)", "calibration_weighted.png")
    
    # Save calibrated predictions to use in downstream tasks
    df["lgb_prob_calibrated"] = y_prob_iso # We will prefer Isotonic if it has lower ECE/Brier
    df.to_parquet(ROOT_DIR / "data" / "processed" / "oof_predictions_calibrated.parquet")
    print("Saved calibrated predictions to oof_predictions_calibrated.parquet")

if __name__ == "__main__":
    main()
