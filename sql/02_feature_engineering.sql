-- 02_feature_engineering.sql
-- Computes point-in-time safe features for each statement.
-- No window uses FOLLOWING to prevent data leakage.

CREATE OR REPLACE TABLE statement_features AS
SELECT 
    *,
    -- 1. Temporal Deltas
    P_2 - LAG(P_2, 1) OVER (PARTITION BY customer_ID ORDER BY S_2) AS P_2_lag1_diff,
    B_1 - LAG(B_1, 1) OVER (PARTITION BY customer_ID ORDER BY S_2) AS B_1_lag1_diff,
    D_39 - LAG(D_39, 1) OVER (PARTITION BY customer_ID ORDER BY S_2) AS D_39_lag1_diff,

    -- 2. Rolling 3-statement means
    AVG(P_2) OVER (PARTITION BY customer_ID ORDER BY S_2 ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS P_2_rolling3_mean,
    AVG(B_1) OVER (PARTITION BY customer_ID ORDER BY S_2 ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS B_1_rolling3_mean,
    
    -- 3. Rolling 6-statement means
    AVG(P_2) OVER (PARTITION BY customer_ID ORDER BY S_2 ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS P_2_rolling6_mean,
    AVG(B_1) OVER (PARTITION BY customer_ID ORDER BY S_2 ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS B_1_rolling6_mean,
    
    -- 4. Ratios
    CASE WHEN B_1 != 0 AND B_1 IS NOT NULL THEN P_2 / B_1 ELSE NULL END AS payment_to_balance_ratio,
    
    -- 5. Delinquency counts (using D_39 > 0 as a proxy)
    SUM(CASE WHEN D_39 > 0 THEN 1 ELSE 0 END) OVER (PARTITION BY customer_ID ORDER BY S_2 ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS D_39_count_3,
    SUM(CASE WHEN D_39 > 0 THEN 1 ELSE 0 END) OVER (PARTITION BY customer_ID ORDER BY S_2 ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS D_39_count_6,
    
    -- 6. Volatility (up to current row to prevent leakage)
    STDDEV_SAMP(P_2) OVER (PARTITION BY customer_ID ORDER BY S_2 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS P_2_volatility,
    STDDEV_SAMP(B_1) OVER (PARTITION BY customer_ID ORDER BY S_2 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS B_1_volatility

FROM read_parquet($input_parquet);
