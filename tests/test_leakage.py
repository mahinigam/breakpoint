import duckdb
import pytest
import re
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

def test_no_future_leakage_in_sql():
    """
    Test that no window functions in feature_engineering.sql use 'FOLLOWING'.
    Any 'FOLLOWING' window definition allows features to "peek" into the future.
    """
    sql_path = ROOT_DIR / "sql" / "02_feature_engineering.sql"
    
    with open(sql_path, "r") as f:
        content = f.read()
        
    # Find all window definitions: 'OVER (...)'
    matches = re.finditer(r"OVER\s*\((.*?)\)", content, re.IGNORECASE | re.DOTALL)
    
    for match in matches:
        window_expr = match.group(1).upper()
        # It should not contain 'FOLLOWING'
        assert "FOLLOWING" not in window_expr, f"Leakage detected in window: OVER ({window_expr})"

def test_feature_table_structure():
    """
    Test that the output customer_features.parquet actually collapses to 1 row per customer.
    """
    processed_path = ROOT_DIR / "data" / "processed" / "customer_features.parquet"
    if not processed_path.exists():
        pytest.skip(f"Feature table not built yet: {processed_path}")
        
    # Check uniqueness of customer_ID
    conn = duckdb.connect()
    res = conn.execute(f"SELECT COUNT(*), COUNT(DISTINCT customer_ID) FROM parquet_scan('{processed_path}')").fetchone()
    
    total_rows = res[0]
    unique_customers = res[1]
    
    assert total_rows == unique_customers, "customer_features contains multiple rows per customer!"
    assert total_rows > 0, "customer_features is empty!"

