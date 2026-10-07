import pandas as pd
import duckdb
from src.config import PATHS
import sys

def load_parquet(filepath):
    """Loads a Parquet file into a Pandas DataFrame."""
    try:
        df = pd.read_parquet(filepath)
        print(f"Successfully loaded {filepath} with shape {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def validate_schema(filepath, expected_columns=None):
    """Validates the schema of a Parquet file using DuckDB."""
    try:
        # Just grab 1 row to get schema
        res = duckdb.sql(f"SELECT * FROM parquet_scan('{filepath}') LIMIT 1").df()
        columns = res.columns.tolist()
        
        if expected_columns:
            missing = set(expected_columns) - set(columns)
            if missing:
                print(f"Schema validation failed for {filepath}. Missing columns: {missing}")
                return False
                
        print(f"Schema validation passed for {filepath}. Found {len(columns)} columns.")
        return True
    except Exception as e:
        print(f"Error validating schema for {filepath}: {e}")
        return False

def main():
    print("Testing data loader...")
    # Test loading labels as it's small
    df = load_parquet(PATHS["parquet_train_labels"])
    if df is not None:
        validate_schema(PATHS["parquet_train_labels"], expected_columns=["customer_ID", "target"])

if __name__ == "__main__":
    main()
