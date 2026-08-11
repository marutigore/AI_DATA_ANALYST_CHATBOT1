"""
Statistical Hypothesis Testing Engine Utility.
Provides statistical calculations for dataset features:
- Pearson & Spearman correlation analysis
- Two-sample Student's t-test
- One-way ANOVA hypothesis test
"""
import logging
import pandas as pd
from typing import Dict, Any, List
from scipy import stats

logger = logging.getLogger(__name__)

def calculate_correlations(df: pd.DataFrame, method: str = "pearson") -> Dict[str, Any]:
    """
    Computes pairwise correlation matrix for all numeric columns.
    """
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.shape[1] < 2:
        return {"error": "Insufficient numeric columns for correlation matrix."}
        
    corr_matrix = numeric_df.corr(method=method).round(4)
    return corr_matrix.to_dict()

def run_ttest(df: pd.DataFrame, group_col: str, value_col: str) -> Dict[str, Any]:
    """
    Runs two-sample independent Student's t-test comparing group means of value_col.
    """
    if group_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"Columns {group_col} or {value_col} not found in DataFrame.")
        
    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        return {"error": f"T-test requires exactly 2 distinct groups, found {len(groups)}."}
        
    group1 = df[df[group_col] == groups[0]][value_col].dropna()
    group2 = df[df[group_col] == groups[1]][value_col].dropna()
    
    t_stat, p_val = stats.ttest_ind(group1, group2)
    return {
        "group1": str(groups[0]),
        "group2": str(groups[1]),
        "group1_mean": round(float(group1.mean()), 4),
        "group2_mean": round(float(group2.mean()), 4),
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_val), 6),
        "statistically_significant": bool(p_val < 0.05)
    }

def run_anova(df: pd.DataFrame, group_col: str, value_col: str) -> Dict[str, Any]:
    """
    Runs One-Way ANOVA test across multiple categorical groups.
    """
    if group_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"Columns {group_col} or {value_col} not found in DataFrame.")
        
    groups = [group[value_col].dropna().values for _, group in df.groupby(group_col)]
    if len(groups) < 2:
        return {"error": "ANOVA requires at least 2 categorical groups."}
        
    f_stat, p_val = stats.f_oneway(*groups)
    return {
        "groups_count": len(groups),
        "f_statistic": round(float(f_stat), 4),
        "p_value": round(float(p_val), 6),
        "statistically_significant": bool(p_val < 0.05)
    }
