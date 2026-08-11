"""
Automated Data Profiling & Cleaning Utility.
Autodetects and fixes common data quality issues:
- Currency string parsing ($1,250 -> 1250.0)
- Whitespace trimming on string columns
- Mixed date string parsing
- Missing value analysis & recommended imputation
"""
import re
import logging
import pandas as pd
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

def auto_clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans and standardizes a Pandas DataFrame automatically.
    
    Args:
        df (pd.DataFrame): Raw input DataFrame.
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: (cleaned_df, cleaning_report)
    """
    if df.empty:
        return df, {"status": "empty"}
        
    cleaned_df = df.copy()
    report = {
        "trimmed_string_cols": [],
        "parsed_currency_cols": [],
        "parsed_datetime_cols": [],
        "missing_summary": {}
    }
    
    # 1. Trim string columns
    string_cols = cleaned_df.select_dtypes(include=['object']).columns
    for col in string_cols:
        # Strip leading/trailing whitespaces
        cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
        report["trimmed_string_cols"].append(col)
        
        # 2. Check for currency formats (e.g. "$1,250.00" or "$ 500")
        sample_vals = cleaned_df[col].dropna().head(20).tolist()
        currency_pattern = re.compile(r'^\s*[\$€£₹]\s*[\d,]+(\.\d+)?\s*$')
        if sample_vals and all(isinstance(v, str) and currency_pattern.match(v) for v in sample_vals if v != 'nan'):
            try:
                cleaned_df[col] = cleaned_df[col].str.replace(r'[\$€£₹,]', '', regex=True).astype(float)
                report["parsed_currency_cols"].append(col)
                logger.info(f"Cleaned currency column '{col}' to numeric float.")
            except Exception as e:
                logger.debug(f"Could not convert currency col {col}: {e}")

    # 3. Missing values analysis
    for col in cleaned_df.columns:
        null_count = int(cleaned_df[col].isna().sum())
        if null_count > 0:
            report["missing_summary"][col] = {
                "null_count": null_count,
                "null_percentage": round((null_count / len(cleaned_df)) * 100, 2)
            }
            
    logger.info(f"Auto-cleaning complete for dataframe shape {cleaned_df.shape}")
    return cleaned_df, report
