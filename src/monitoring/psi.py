import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
import duckdb

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def calculate_psi(expected, actual, bins=10):
    """Calculate Population Stability Index for a single numerical feature."""
    # Create quantile bins based on the expected (training) distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    # Add a tiny amount to the last breakpoint to ensure the max value is included
    breakpoints[-1] += 1e-6
    # Ensure breakpoints are unique
    breakpoints = np.unique(breakpoints)
    
    # If we have less than 2 unique breakpoints (e.g. constant feature), return 0
    if len(breakpoints) < 2:
        return 0.0
        
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    
    # Replace 0% with 0.0001% to avoid division by zero or log(0)
    expected_pct = np.clip(expected_pct, 1e-4, 1.0)
    actual_pct = np.clip(actual_pct, 1e-4, 1.0)
    
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi

def main():
    train_path = str(ROOT_DIR / "data" / "parquet" / "train_data.parquet")
    test_path = str(ROOT_DIR / "data" / "parquet" / "test_sample.parquet")
    
    print("Computing PSI on a subset of raw numerical features (Train vs Test Sample)...")
    
    # Get a list of numerical columns to track
    cols_query = f"SELECT * FROM parquet_scan('{train_path}') LIMIT 1"
    df_schema = duckdb.sql(cols_query).df()
    
    # Select 20 numeric features across different families (D, S, P, B, R)
    numeric_cols = [c for c in df_schema.columns if df_schema[c].dtype in [np.float32, np.float64, np.int64]]
    # Pick a sample of important-looking features
    features_to_monitor = [c for c in numeric_cols if c in ["P_2", "D_39", "B_1", "B_2", "R_1", "S_3", "D_41", "B_3", "D_44", "B_4", "D_45", "B_5", "R_2", "D_46", "D_47", "D_48", "B_6", "B_7", "B_8", "D_51"]]
    
    if not features_to_monitor:
        features_to_monitor = numeric_cols[:20]
        
    psi_values = {}
    
    for feat in features_to_monitor:
        try:
            # Query just the single feature, dropping nulls
            train_vals = duckdb.sql(f"SELECT {feat} FROM parquet_scan('{train_path}') WHERE {feat} IS NOT NULL").df()[feat].values
            test_vals = duckdb.sql(f"SELECT {feat} FROM parquet_scan('{test_path}') WHERE {feat} IS NOT NULL").df()[feat].values
            
            if len(train_vals) > 0 and len(test_vals) > 0:
                psi = calculate_psi(train_vals, test_vals)
                psi_values[feat] = psi
        except Exception as e:
            print(f"Failed PSI for {feat}: {e}")
            
    # Sort and plot
    sorted_psi = dict(sorted(psi_values.items(), key=lambda item: item[1], reverse=True))
    
    # Print drift flags
    flagged = 0
    for f, p in sorted_psi.items():
        if p > 0.25:
            print(f"🔴 RED FLAG: {f} PSI = {p:.3f} (Significant Shift)")
            flagged += 1
        elif p > 0.10:
            print(f"🟡 YELLOW FLAG: {f} PSI = {p:.3f} (Moderate Shift)")
        else:
            print(f"🟢 OK: {f} PSI = {p:.3f}")
            
    print(f"\nTotal Features Flagged (>0.25): {flagged} out of {len(sorted_psi)}")
    
    # Plotting
    names = list(sorted_psi.keys())
    values = list(sorted_psi.values())
    colors = ['red' if v > 0.25 else 'orange' if v > 0.10 else 'green' for v in values]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values, color=colors)
    plt.axhline(0.25, color='red', linestyle='--', label='Red Flag Threshold (0.25)')
    plt.axhline(0.10, color='orange', linestyle='--', label='Yellow Flag Threshold (0.10)')
    
    plt.xticks(rotation=45)
    plt.ylabel("Population Stability Index (PSI)")
    plt.title("Feature Drift: Train vs. 1% Test Sample")
    plt.legend()
    plt.tight_layout()
    
    os.makedirs(ROOT_DIR / "figures", exist_ok=True)
    plt.savefig(ROOT_DIR / "figures" / "psi_barplot.png", dpi=300)
    plt.close()
    
    print("Saved PSI barplot to figures/psi_barplot.png")

if __name__ == "__main__":
    main()
