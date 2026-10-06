import pytest
import numpy as np
from src.decision.optimal_thresholds import compute_profit
from src.config import NEGATIVE_WEIGHT

def test_profit_weighting():
    """Verify that the 20x weight is applied correctly in profit calculations."""
    y_true = np.array([0, 0, 1]) # Two goods, one default
    y_prob = np.array([0.1, 0.9, 0.9]) # Threshold 0.5 approves the first good only
    
    # At cutoff 0.5, we approve y_prob <= 0.5 (the first good account)
    # So we get 1 good account approved.
    
    # Without weights, profit = 1 * margin
    # With 20x negative weight, the first good account counts as 20 accounts.
    # So profit = 20 * margin
    
    margin = 100
    loss = 500
    
    profit = compute_profit(y_true, y_prob, 0.5, margin, loss, NEGATIVE_WEIGHT)
    
    assert profit == 20 * margin, f"Expected {20*margin}, got {profit}"

def test_profit_bounds():
    """Verify that at cutoff=0.0, we reject everyone (profit=0)."""
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.4, 0.6, 0.2, 0.8])
    profit = compute_profit(y_true, y_prob, 0.0, 100, 500, NEGATIVE_WEIGHT)
    assert profit == 0
