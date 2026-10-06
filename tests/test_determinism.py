import pytest
import numpy as np
import lightgbm as lgb
from sklearn.datasets import make_classification

def test_lgbm_determinism():
    """Verify that LightGBM produces exact same predictions given the same seed."""
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    
    model1 = lgb.LGBMClassifier(n_estimators=10, random_state=42, n_jobs=1)
    model1.fit(X, y)
    preds1 = model1.predict_proba(X)[:, 1]
    
    model2 = lgb.LGBMClassifier(n_estimators=10, random_state=42, n_jobs=1)
    model2.fit(X, y)
    preds2 = model2.predict_proba(X)[:, 1]
    
    np.testing.assert_array_equal(preds1, preds2)
