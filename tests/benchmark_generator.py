"""
Synthetic Dataset Benchmark & Stress Test Generator Utility.
Generates large-scale synthetic enterprise datasets with configurable row sizes,
missing value noise, mixed dates, and numeric categories to benchmark system latency.
"""
import os
import random
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def generate_benchmark_dataset(num_rows: int = 5000) -> pd.DataFrame:
    """
    Generates a synthetic enterprise dataset for stress testing.
    
    Args:
        num_rows (int): Number of records to generate.
        
    Returns:
        pd.DataFrame: Generated synthetic DataFrame.
    """
    np.random.seed(42)
    random.seed(42)
    
    regions = ["North America", "EMEA", "APAC", "LATAM"]
    categories = ["Technology", "Healthcare", "Finance", "Retail", "Energy"]
    payment_methods = ["Credit Card", "Wire Transfer", "PayPal", "ACH"]
    
    data = {
        "transaction_id": [f"TXN-{10000 + i}" for i in range(num_rows)],
        "timestamp": pd.date_range(start="2024-01-01", periods=num_rows, freq="h").astype(str),
        "region": [random.choice(regions) for _ in range(num_rows)],
        "category": [random.choice(categories) for _ in range(num_rows)],
        "sales_amount": np.round(np.random.uniform(10.0, 5000.0, size=num_rows), 2),
        "quantity": np.random.randint(1, 50, size=num_rows),
        "discount_pct": np.round(np.random.uniform(0.0, 0.3, size=num_rows), 2),
        "payment_method": [random.choice(payment_methods) for _ in range(num_rows)]
    }
    
    df = pd.DataFrame(data)
    
    # Introduce controlled missingness (2% missing values)
    for col in ["sales_amount", "payment_method"]:
        mask = np.random.rand(len(df)) < 0.02
        df.loc[mask, col] = np.nan
        
    logger.info(f"Generated synthetic benchmark dataset with shape {df.shape}")
    return df

def save_benchmark_csv(num_rows: int = 5000, filename: str = "synthetic_benchmark.csv") -> str:
    """Generates synthetic dataset and saves to CSV file."""
    df = generate_benchmark_dataset(num_rows)
    df.to_csv(filename, index=False)
    logger.info(f"Saved benchmark dataset to {filename}")
    return filename

if __name__ == "__main__":
    save_benchmark_csv(1000)
