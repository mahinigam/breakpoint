import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss

def amex_metric(y_true, y_pred, negative_weight=20):
    """
    The official Amex competition metric, adjusted for the subsampling weight.
    M = 0.5 * (Normalized Gini + Default Rate Captured at 4%)
    """
    # Create sample weights
    weights = np.where(y_true == 0, negative_weight, 1.0)
    
    # 1. Normalized Gini
    # Gini = 2 * AUC - 1
    # Weighted AUC
    auc = roc_auc_score(y_true, y_pred, sample_weight=weights)
    gini = 2 * auc - 1
    
    # 2. Default Rate Captured at 4%
    # Sort by predicted probability descending
    indices = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[indices]
    weights_sorted = weights[indices]
    
    # Cumulative weight
    cum_weights = np.cumsum(weights_sorted)
    total_weight = np.sum(weights)
    
    # Find the index where cumulative weight reaches 4% of total weight
    cutoff_weight = 0.04 * total_weight
    cutoff_idx = np.searchsorted(cum_weights, cutoff_weight)
    
    # Calculate the recall up to this index (positives captured)
    # Since positives have weight 1, it's just the sum of y_true up to cutoff
    # divided by the total number of positives.
    positives_captured = np.sum(y_true_sorted[:cutoff_idx + 1])
    total_positives = np.sum(y_true)
    
    d_at_4 = positives_captured / total_positives
    
    return 0.5 * (gini + d_at_4)

def ks_statistic(y_true, y_pred, negative_weight=20):
    """
    Kolmogorov-Smirnov statistic with sample weights.
    """
    weights = np.where(y_true == 0, negative_weight, 1.0)
    
    indices = np.argsort(y_pred)
    y_true_sorted = y_true[indices]
    weights_sorted = weights[indices]
    
    # Cumulative distributions
    cum_pos = np.cumsum((y_true_sorted == 1) * weights_sorted) / np.sum((y_true == 1) * weights)
    cum_neg = np.cumsum((y_true_sorted == 0) * weights_sorted) / np.sum((y_true == 0) * weights)
    
    return np.max(np.abs(cum_pos - cum_neg))

def evaluate_model(y_true, y_pred, negative_weight=20):
    """
    Compute all required metrics.
    """
    weights = np.where(y_true == 0, negative_weight, 1.0)
    
    auc = roc_auc_score(y_true, y_pred, sample_weight=weights)
    ks = ks_statistic(y_true, y_pred, negative_weight=negative_weight)
    gini = 2 * auc - 1
    brier = brier_score_loss(y_true, y_pred, sample_weight=weights)
    amex = amex_metric(y_true, y_pred, negative_weight=negative_weight)
    
    return {
        "AUC": auc,
        "KS": ks,
        "Gini": gini,
        "Brier": brier,
        "Amex Metric": amex
    }
