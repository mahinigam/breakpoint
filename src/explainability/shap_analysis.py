import pandas as pd
import numpy as np
import shap
import lightgbm as lgb
from pathlib import Path
import matplotlib.pyplot as plt
import os
import gc

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def main():
    print("Loading a 10% sample of data for SHAP Explainability...")
    train_path = ROOT_DIR / "data" / "processed" / "customer_features.parquet"
    if not train_path.exists():
        print(f"Error: {train_path} not found.")
        return
        
    df = pd.read_parquet(train_path)
    
    labels_path = ROOT_DIR / "data" / "parquet" / "train_labels.parquet"
    labels = pd.read_parquet(labels_path)
    df = df.merge(labels, on="customer_ID", how="inner")
    
    # We sample 10% for speed
    df_sample = df.sample(frac=0.1, random_state=42)
    
    y = df_sample['target'].values
    X = df_sample.drop(['customer_ID', 'target'], axis=1)
    
    # Identify categorical columns
    cat_cols = [c for c in X.columns if X[c].dtype.name in ['category', 'object'] or c in ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']]
    # We didn't do OHE, so for SHAP to work simply, let's just use numeric features for the portfolio demo, or tell LightGBM they are numeric if encoded
    # The dataset had D_63 and D_64 as object in CSV. We assume they are dropped or numeric.
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    X = X[numeric_cols]
    
    print("Training a proxy LightGBM model on the sample...")
    model = lgb.LGBMClassifier(
        n_estimators=100, 
        learning_rate=0.05, 
        num_leaves=31, 
        random_state=42, 
        n_jobs=-1,
        verbose=-1
    )
    model.fit(X, y)
    
    print("Computing SHAP values...")
    # TreeExplainer is fast for LightGBM
    explainer = shap.TreeExplainer(model)
    # We compute SHAP on a smaller subset (1,000) for fast plotting
    X_shap = X.sample(n=1000, random_state=42)
    shap_values = explainer.shap_values(X_shap)
    
    # In binary classification, shap_values is a list of [negative_class, positive_class] or just an array depending on version
    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1]
    else:
        shap_values_pos = shap_values
        
    os.makedirs(ROOT_DIR / "figures", exist_ok=True)
    
    # 1. Standard Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_pos, X_shap, show=False)
    plt.title("SHAP Beeswarm: Top Individual Features")
    plt.tight_layout()
    plt.savefig(ROOT_DIR / "figures" / "shap_beeswarm.png", dpi=300)
    plt.close()
    
    # 2. Grouped SHAP Analysis (by Feature Family)
    print("Grouping SHAP values by Feature Family (D, S, P, B, R)...")
    
    # Compute mean absolute SHAP value for each feature
    mean_abs_shap = np.abs(shap_values_pos).mean(axis=0)
    
    family_importance = {'Delinquency (D)': 0, 'Spend (S)': 0, 'Payment (P)': 0, 'Balance (B)': 0, 'Risk (R)': 0, 'Other': 0}
    
    for idx, col in enumerate(X_shap.columns):
        val = mean_abs_shap[idx]
        if col.startswith('D_'):
            family_importance['Delinquency (D)'] += val
        elif col.startswith('S_'):
            family_importance['Spend (S)'] += val
        elif col.startswith('P_'):
            family_importance['Payment (P)'] += val
        elif col.startswith('B_'):
            family_importance['Balance (B)'] += val
        elif col.startswith('R_'):
            family_importance['Risk (R)'] += val
        else:
            family_importance['Other'] += val
            
    families = list(family_importance.keys())
    importances = list(family_importance.values())
    
    plt.figure(figsize=(8, 5))
    plt.bar(families, importances, color='teal')
    plt.ylabel("Mean Absolute SHAP Value (Impact on Model)")
    plt.title("Feature Family Importance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(ROOT_DIR / "figures" / "shap_grouped.png", dpi=300)
    plt.close()
    
    # 3. Reason Code Generation (for 3 random high-risk customers)
    from src.explainability.reason_codes import print_reason_codes_for_top_risks
    
    preds = np.array(model.predict_proba(X_shap))[:, 1]
    feature_names = X_shap.columns.tolist()
    print_reason_codes_for_top_risks(preds, shap_values_pos, feature_names, n_customers=3)

if __name__ == "__main__":
    main()
