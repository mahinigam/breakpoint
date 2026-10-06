import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from src.config import PATHS, NEGATIVE_WEIGHT, SEED, ROOT_DIR
from src.models.baseline_lr import get_lr_pipeline
from src.models.lgbm import get_lgbm_model, optimize_lgbm
from src.evaluation.metrics import evaluate_model
from src.evaluation.delong import delong_roc_test

def main():
    print("Loading datasets...")
    features_df = pd.read_parquet(PATHS["processed_features"])
    labels_df = pd.read_parquet(PATHS["parquet_train_labels"])
    
    # Merge on customer_ID
    df = features_df.merge(labels_df, on="customer_ID", how="inner")
    
    # Check if S_2 (date) exists and drop it, we don't train on dates
    cols_to_drop = ["customer_ID", "target"]
    if "S_2" in df.columns:
        cols_to_drop.append("S_2")
        
    X = df.drop(columns=cols_to_drop)
    y = df["target"].values
    
    # Only keep numeric columns for now to simplify Baseline.
    # LightGBM handles categorical but LR needs OHE. Let's stick to numeric for this phase.
    X = X.select_dtypes(include=[np.number])
    
    # Sample weights
    sample_weights = np.where(y == 0, NEGATIVE_WEIGHT, 1.0)
    
    print(f"Dataset shape: {X.shape}")
    
    # Stratified K-Fold CV (equivalent to GroupKFold since we have 1 row per customer)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    lr_oof = np.zeros(len(y))
    lgb_oof = np.zeros(len(y))
    
    # Optimize LGBM on a subset to save time (fold 1)
    # Using defaults for now unless user wants tuning. We will just use standard defaults.
    lgb_params = None
    
    print("Starting 5-Fold Cross Validation...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"--- Fold {fold+1} ---")
        X_train, y_train, sw_train = X.iloc[train_idx], y[train_idx], sample_weights[train_idx]
        X_val, y_val, sw_val = X.iloc[val_idx], y[val_idx], sample_weights[val_idx]
        
        # 1. Train Logistic Regression
        lr = get_lr_pipeline(random_state=SEED)
        print("  Training Logistic Regression...")
        # Note: scikit-learn pipeline doesn't natively pass sample_weight to step easily in old versions,
        # but for LR we can pass it to fit parameters.
        lr.fit(X_train, y_train, lr__sample_weight=sw_train)
        lr_oof[val_idx] = lr.predict_proba(X_val)[:, 1]
        
        # 2. Train LightGBM
        print("  Training LightGBM...")
        lgb_model = get_lgbm_model(lgb_params)
        lgb_model.fit(
            X_train, y_train,
            sample_weight=sw_train,
            eval_set=[(X_val, y_val)],
            eval_sample_weight=[sw_val],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        lgb_oof[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
        
    print("\n--- Evaluation on Out-Of-Fold Predictions ---")
    lr_metrics = evaluate_model(y, lr_oof, negative_weight=NEGATIVE_WEIGHT)
    lgb_metrics = evaluate_model(y, lgb_oof, negative_weight=NEGATIVE_WEIGHT)
    
    print("\nLogistic Regression Baseline:")
    for k, v in lr_metrics.items():
        print(f"  {k}: {v:.4f}")
        
    print("\nLightGBM Model:")
    for k, v in lgb_metrics.items():
        print(f"  {k}: {v:.4f}")
        
    print("\nCalculating DeLong p-value for AUC difference...")
    # Note: DeLong can be very slow for 458k rows. We will sample 50k for the test if it's too large,
    # but the fastDeLong algorithm should handle 450k in seconds.
    p_val = delong_roc_test(y, lgb_oof, lr_oof)
    print(f"DeLong p-value: {p_val:.4e}")
    
    # Save OOF predictions for Phase 3 (Decisioning)
    oof_df = pd.DataFrame({
        "customer_ID": df["customer_ID"],
        "target": y,
        "lr_prob": lr_oof,
        "lgb_prob": lgb_oof
    })
    
    os.makedirs(ROOT_DIR / "data" / "processed", exist_ok=True)
    oof_path = ROOT_DIR / "data" / "processed" / "oof_predictions.parquet"
    oof_df.to_parquet(oof_path)
    print(f"Saved OOF predictions to {oof_path}")

if __name__ == "__main__":
    main()
