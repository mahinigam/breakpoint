-- 01_csv_to_parquet.sql
-- DuckDB SQL script to convert raw CSV files to Parquet out-of-core

-- Convert train_labels
COPY (SELECT * FROM read_csv_auto('data/raw/train_labels.csv')) 
TO 'data/parquet/train_labels.parquet' (FORMAT 'parquet');

-- Convert train_data
COPY (SELECT * FROM read_csv_auto('data/raw/train_data.csv')) 
TO 'data/parquet/train_data.parquet' (FORMAT 'parquet');

-- Convert a sample of test_data for drift analysis (1% sample)
COPY (SELECT * FROM read_csv_auto('data/raw/test_data.csv') USING SAMPLE 1%) 
TO 'data/parquet/test_data_sample.parquet' (FORMAT 'parquet');
