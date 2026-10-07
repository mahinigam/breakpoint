import pandas as pd
from pathlib import Path

def generate_drift_flags(psi_df, psi_threshold=0.25):
    """
    Applies the retrain trigger logic to a dataframe of PSI values.
    
    Retrain trigger (plain English):
    "If more than 10% of features exceed PSI > 0.25, or if any 3 features 
    in the delinquency (D) or balance (B) families exceed PSI > 0.25, 
    initiate model retraining review."
    """
    print("\n--- Model Drift & Retrain Review ---")
    
    total_features = len(psi_df)
    drifted_features = psi_df[psi_df['PSI'] > psi_threshold]
    drifted_count = len(drifted_features)
    
    pct_drifted = (drifted_count / total_features) * 100
    print(f"Features with PSI > {psi_threshold}: {drifted_count} ({pct_drifted:.1f}%)")
    
    d_b_drifted = drifted_features[drifted_features['Feature'].str.startswith(('D_', 'B_'))]
    d_b_count = len(d_b_drifted)
    print(f"Delinquency/Balance features drifted: {d_b_count}")
    
    retrain_needed = False
    reasons = []
    
    if pct_drifted > 10.0:
        retrain_needed = True
        reasons.append(f"Global drift: >10% of features ({pct_drifted:.1f}%) have PSI > {psi_threshold}")
        
    if d_b_count >= 3:
        retrain_needed = True
        reasons.append(f"Critical family drift: {d_b_count} features in D/B families have PSI > {psi_threshold}")
        
    if retrain_needed:
        print("🚨 ACTION REQUIRED: Model retraining review initiated.")
        for r in reasons:
            print(f"  - {r}")
    else:
        print("✅ Model stable. No retrain required.")
        
    return retrain_needed, reasons

def main():
    # Load PSI results from CSV
    csv_path = Path("dashboards/psi.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return
        
    psi_df = pd.read_csv(csv_path)
    generate_drift_flags(psi_df)

if __name__ == "__main__":
    main()
