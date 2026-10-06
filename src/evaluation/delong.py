import numpy as np
import scipy.stats

def compute_midrank(x):
    """Computes midranks."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    # Note: T is 0-indexed, so we add 1 for rank
    T2[J] = T + 1
    return T2

def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    Fast DeLong algorithm.
    """
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])

    tz_pos = tz[:, :m]
    tz_neg = tz[:, m:]
    
    # structural components
    v10 = tz_pos - tx
    v01 = ty - tz_neg
    
    # covariance matrix
    S10 = np.cov(v10)
    S01 = np.cov(v01)
    
    # If S10 or S01 are scalars (when k=1), force them to be 2D
    if k == 1:
        S10 = np.array([[S10]])
        S01 = np.array([[S01]])
        
    S = S10 / m + S01 / n
    return S

def calc_pvalue(aucs, SIGMA):
    """Computes p-value for the difference between two AUCs."""
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, SIGMA), l.T))
    pvalue = 2 * (1 - scipy.stats.norm.cdf(z))
    return pvalue[0][0]

def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes the p-value for the difference in AUC between two sets of predictions
    using the DeLong test.
    """
    # Align sorting with expected input for fastDeLong
    order = np.argsort(ground_truth)[::-1]
    ground_truth = ground_truth[order]
    predictions_one = predictions_one[order]
    predictions_two = predictions_two[order]
    
    label_1_count = np.sum(ground_truth)
    
    preds = np.vstack((predictions_one, predictions_two))
    
    # Compute AUCs manually
    auc1 = roc_auc_score(ground_truth, predictions_one)
    auc2 = roc_auc_score(ground_truth, predictions_two)
    aucs = np.array([auc1, auc2])
    
    # Compute Covariance Matrix
    SIGMA = fastDeLong(preds, label_1_count)
    
    # Compute p-value
    pvalue = calc_pvalue(aucs, SIGMA)
    return pvalue
    
from sklearn.metrics import roc_auc_score
