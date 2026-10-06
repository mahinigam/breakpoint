import lightgbm as lgb
import optuna
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def get_lgbm_model(params=None):
    """
    Returns a LightGBM model with given params or defaults.
    """
    if params is None:
        params = {
            'n_estimators': 300,
            'learning_rate': 0.05,
            'max_depth': 6,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
    
    return lgb.LGBMClassifier(**params)

def optimize_lgbm(X, y, sample_weight=None, n_trials=20):
    """
    Runs Optuna optimization to find best hyperparameters.
    """
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 500),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        # Split for tuning
        if sample_weight is not None:
            X_train, X_val, y_train, y_val, sw_train, sw_val = train_test_split(
                X, y, sample_weight, test_size=0.2, random_state=42, stratify=y
            )
            eval_sw = [sw_val]
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            sw_train = None
            eval_sw = None
            
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train, 
            sample_weight=sw_train,
            eval_set=[(X_val, y_val)],
            eval_sample_weight=eval_sw,
            eval_metric='auc',
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds, sample_weight=sw_val if sample_weight is not None else None)
        return auc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print("Best LightGBM params:", study.best_params)
    return study.best_params
