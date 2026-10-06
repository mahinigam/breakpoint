# Model Card: Profit-Optimal Credit Decisioning Model

## Model Details
- **Architecture:** LightGBM (Gradient Boosting Decision Tree) with 100 estimators, leaf-wise growth.
- **Task:** Binary classification (predicting credit default probability within 18 months).
- **Features:** 400+ point-in-time engineered SQL features derived from 13 statements (Delinquency, Spend, Payment, Balance, Risk metrics).
- **Optimization:** Hyperparameters tuned via Optuna with 5-Fold GroupKFold CV.

## Intended Use
- **Primary Use Case:** Determining approval/rejection thresholds for credit limits and account acquisition.
- **Out of Scope:** Making individual lending decisions purely based on raw scores without economic calibration, or using this model on non-US markets.

## Metrics
- **Performance:** 
  - AUC: 0.9584
  - Amex Metric (Gini + D@4%): 0.7811
- **Economic Value:**
  - MiniMax Regret Policy ensures worst-case profit loss is bounded to 4.7% across macro-economic extremes.

## Training Data
- **Dataset:** American Express Default Prediction dataset (~459k customers, ~5.5M statements).
- **Subsampling Correction:** The training labels were heavily subsampled (only 5% of negatives kept). We apply a **20x sample weight** to the negative class during metric evaluation, profit calculation, and calibration to accurately reflect the true population default rate of ~1.6%.

## Evaluation Data
- **Offline Evaluation:** 5-Fold Cross Validation.
- **Drift Monitoring:** PSI calculated on a 1% stratified sample of the test set.

## Ethical & Fairness Considerations
- **Anonymization:** All features are heavily anonymized. It is impossible to manually verify if features proxy protected classes (e.g., zip codes proxying race). In a production environment with named features, a full Disparate Impact and Equal Opportunity analysis is mandatory.
- **Fairness through Economics:** F1-score optimization unintentionally maximizes false positives in a way that harms the business and provides credit to unequipped borrowers during stress scenarios. The MiniMax regret policy balances economic stability with lending availability.

## Limitations
1. **Offline Only:** The power analysis and Champion-Challenger designs are offline simulations.
2. **DeLong Weighting Flaw:** Standard implementations of the DeLong test fail to compute valid variances on 20x subsampled data. We rely on Expected Profit curves instead of pure AUC p-values.
3. **Anonymized Reason Codes:** Plain-English reason codes rely on feature prefix families (e.g., D = Delinquency), as actual feature definitions are hidden.
