import pytest
import numpy as np
from src.monitoring.psi import calculate_psi

def test_psi_identical():
    """Identical distributions should have PSI = 0."""
    train = np.random.normal(0, 1, 1000)
    test = train.copy()
    psi = calculate_psi(train, test)
    assert np.isclose(psi, 0.0)

def test_psi_shifted():
    """Shifted distributions should have PSI > 0."""
    train = np.random.normal(0, 1, 1000)
    test = np.random.normal(1, 1, 1000)
    psi = calculate_psi(train, test)
    assert psi > 0.1

def test_psi_empty_bins():
    """PSI handles distributions where some bins are completely empty."""
    train = np.random.normal(0, 1, 1000)
    test = np.array([5.0] * 1000) # Entire test set is in one bin
    psi = calculate_psi(train, test)
    assert not np.isnan(psi)
    assert psi > 1.0 # Should be a massive shift
