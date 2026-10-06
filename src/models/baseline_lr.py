from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

def get_lr_pipeline(random_state=42):
    """
    Returns an unfitted Logistic Regression pipeline
    with median imputation and standard scaling.
    """
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(
            C=0.1,
            max_iter=1000,
            random_state=random_state,
            n_jobs=-1
        ))
    ])
    return pipeline
