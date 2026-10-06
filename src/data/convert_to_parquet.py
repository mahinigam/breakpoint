import duckdb
import os
from src.config import PATHS

def convert_csv_to_parquet(csv_path, parquet_path):
    if not os.path.exists(csv_path):
        print(f"Skipping {csv_path}: File not found.")
        return
        
    print(f"Converting {csv_path} to {parquet_path}...")
    
    # Using DuckDB to stream CSV to Parquet out-of-core
    query = f"COPY (SELECT * FROM read_csv_auto('{csv_path}')) TO '{parquet_path}' (FORMAT 'parquet');"
    
    duckdb.sql(query)
    print(f"Done. Saved to {parquet_path}.")

def main():
    # Convert train_labels
    convert_csv_to_parquet(PATHS["raw_train_labels"], PATHS["parquet_train_labels"])
    
    # Convert train_data
    convert_csv_to_parquet(PATHS["raw_train_data"], PATHS["parquet_train_data"])
    
    # Convert a sample of test_data for drift analysis
    if os.path.exists(PATHS["raw_test_data"]):
        print(f"Sampling 1% of {PATHS['raw_test_data']} to {PATHS['parquet_test_sample']}...")
        query = f"COPY (SELECT * FROM read_csv_auto('{PATHS['raw_test_data']}') USING SAMPLE 1%) TO '{PATHS['parquet_test_sample']}' (FORMAT 'parquet');"
        duckdb.sql(query)
        print("Done.")

if __name__ == "__main__":
    main()
