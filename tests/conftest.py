import pytest
import pandas as pd
from pathlib import Path
import os
import duckdb

ROOT_DIR = Path(__file__).resolve().parent.parent

@pytest.fixture(scope="session")
def sample_data():
    """Loads a 1% sample of the data for testing."""
    # Try to load the parquet sample if it exists
    sample_path = ROOT_DIR / "data" / "parquet" / "test_data_sample.parquet"
    if not sample_path.exists():
        # Generate a dummy dataset for CI if actual data is not available
        df = pd.DataFrame({
            "customer_ID": [f"CUST_{i}" for i in range(100)],
            "target": [0] * 95 + [1] * 5,
            "P_2": [0.8] * 100,
            "D_39": [0.0] * 100,
            "B_1": [0.01] * 100
        })
        return df
    return pd.read_parquet(sample_path)

@pytest.fixture
def test_db():
    """Provides a temporary DuckDB connection for testing."""
    conn = duckdb.connect(":memory:")
    yield conn
    conn.close()
