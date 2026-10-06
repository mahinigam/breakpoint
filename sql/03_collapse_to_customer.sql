-- 03_collapse_to_customer.sql
-- Collapse the statement-level features down to one row per customer, taking the most recent statement.

CREATE OR REPLACE TABLE customer_features AS
SELECT * EXCLUDE (row_num)
FROM (
    SELECT 
        *,
        ROW_NUMBER() OVER (PARTITION BY customer_ID ORDER BY S_2 DESC) as row_num
    FROM statement_features
)
WHERE row_num = 1;

-- Export to Parquet
COPY customer_features TO $output_parquet (FORMAT 'parquet');
