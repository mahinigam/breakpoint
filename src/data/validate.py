import duckdb
import os
import sys
from src.config import PATHS

def count_csv_lines(filepath):
    try:
        res = duckdb.sql(f"SELECT COUNT(*) FROM read_csv_auto('{filepath}')").fetchone()[0]
        return res
    except Exception as e:
        print(f"Error reading CSV {filepath}: {e}")
        return None

def count_parquet_lines(filepath):
    try:
        res = duckdb.sql(f"SELECT COUNT(*) FROM parquet_scan('{filepath}')").fetchone()[0]
        return res
    except Exception as e:
        print(f"Error reading Parquet {filepath}: {e}")
        return None

def compare_counts(csv_path, parquet_path, name):
    if not os.path.exists(csv_path) or not os.path.exists(parquet_path):
        print(f"Skipping {name}: files missing.")
        return True
        
    print(f"Validating {name}...")
    csv_count = count_csv_lines(csv_path)
    parquet_count = count_parquet_lines(parquet_path)
    
    if csv_count == parquet_count:
        print(f"  ✅ {name} counts match: {csv_count}")
        return True
    else:
        print(f"  ❌ {name} mismatch: CSV={csv_count}, Parquet={parquet_count}")
        return False

def main():
    success = True
    
    success &= compare_counts(PATHS["raw_train_labels"], PATHS["parquet_train_labels"], "train_labels")
    success &= compare_counts(PATHS["raw_train_data"], PATHS["parquet_train_data"], "train_data")
    
    if not success:
        print("\nValidation FAILED.")
        sys.exit(1)
    else:
        print("\nValidation PASSED. You can now delete the raw CSV files.")

if __name__ == "__main__":
    main()
