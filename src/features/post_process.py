import pandas as pd
import numpy as np
from src.config import PATHS

def downcast_float(df):
    """Downcasts float64 columns to float32 to save memory."""
    float64_cols = df.select_dtypes(include=['float64']).columns
    if len(float64_cols) > 0:
        df[float64_cols] = df[float64_cols].astype(np.float32)
        print(f"Downcasted {len(float64_cols)} columns from float64 to float32.")
    return df

def generate_null_summary(df):
    """Returns a summary of missing values."""
    null_counts = df.isnull().sum()
    null_pct = null_counts / len(df) * 100
    
    summary = pd.DataFrame({
        'missing_count': null_counts,
        'missing_percent': null_pct
    })
    
    summary = summary[summary['missing_count'] > 0].sort_values('missing_percent', ascending=False)
    return summary

def main():
    print("Loading customer features...")
    df = pd.read_parquet(PATHS["customer_features"])
    
    initial_mem = df.memory_usage().sum() / 1024**2
    print(f"Initial memory usage: {initial_mem:.2f} MB")
    
    df = downcast_float(df)
    
    final_mem = df.memory_usage().sum() / 1024**2
    print(f"Final memory usage: {final_mem:.2f} MB ({(initial_mem - final_mem) / initial_mem * 100:.1f}% reduction)")
    
    print("\nNull Summary (Top 10):")
    null_summary = generate_null_summary(df)
    print(null_summary.head(10))
    
    # Save optimized dataframe back
    df.to_parquet(PATHS["customer_features"])
    print("\nSaved optimized customer features.")

if __name__ == "__main__":
    main()
