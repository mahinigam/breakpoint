import duckdb
import os
from pathlib import Path
from src.config import PATHS, ROOT_DIR

def run_feature_engineering():
    print("Running feature engineering...")
    input_parquet = str(PATHS["parquet_train_data"])
    output_parquet = str(PATHS["processed_features"])
    
    if not os.path.exists(input_parquet):
        print(f"Error: Input file {input_parquet} does not exist.")
        return

    # Load SQL scripts
    sql_dir = ROOT_DIR / "sql"
    with open(sql_dir / "02_feature_engineering.sql", "r") as f:
        sql_features = f.read()
    with open(sql_dir / "03_collapse_to_customer.sql", "r") as f:
        sql_collapse = f.read()
        
    # Replace variables in SQL
    # Note: duckdb parameter binding doesn't work for table names in some cases,
    # so we'll use simple string replacement here safely.
    sql_features = sql_features.replace("$input_parquet", f"'{input_parquet}'")
    sql_collapse = sql_collapse.replace("$output_parquet", f"'{output_parquet}'")
    
    print("Executing 02_feature_engineering.sql...")
    duckdb.sql(sql_features)
    
    print("Executing 03_collapse_to_customer.sql...")
    duckdb.sql(sql_collapse)
    
    print(f"Feature engineering complete. Output saved to {output_parquet}")

if __name__ == "__main__":
    run_feature_engineering()
