import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

def get_ks_optimal_threshold(y_true, y_prob, neg_weight):
    """Returns the threshold that maximizes the KS statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, sample_weight=np.where(y_true==0, neg_weight, 1.0))
    ks_stat = tpr - fpr
    idx = np.argmax(ks_stat)
    return thresholds[idx]

def get_f1_optimal_threshold(y_true, y_prob, neg_weight):
    """Returns threshold that maximizes F1 score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob, sample_weight=np.where(y_true==0, neg_weight, 1.0))
    f1 = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-15)
    idx = np.argmax(f1)
    return thresholds[idx]

def get_top_pct_threshold(y_true, y_prob, neg_weight, pct=0.04):
    """Returns threshold that rejects exactly pct% of the volume."""
    weights = np.where(y_true == 0, neg_weight, 1.0)
    total_weight = np.sum(weights)
    
    indices = np.argsort(y_prob)[::-1]
    weights_sorted = weights[indices]
    
    cum_weights = np.cumsum(weights_sorted)
    cutoff_weight = pct * total_weight
    cutoff_idx = np.searchsorted(cum_weights, cutoff_weight)
    
    # Ensure cutoff_idx is within bounds
    cutoff_idx = min(cutoff_idx, len(y_prob) - 1)
    return y_prob[indices[cutoff_idx]]

def get_fixed_threshold(y_true, y_prob, neg_weight, threshold=0.5):
    """Returns a fixed threshold (e.g., 0.5)."""
    return threshold

def get_profit_optimal_threshold(y_true, y_prob, neg_weight, margin, loss):
    """Returns the theoretical profit-optimal threshold for a specific scenario."""
    r = loss / margin
    return 1.0 / (1.0 + r)
