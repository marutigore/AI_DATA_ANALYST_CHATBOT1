"""
DuckDB High-Performance Query Engine Utility.
Allows executing near-instant SQL queries directly over in-memory Pandas DataFrames
using DuckDB's C++ vectorized analytical query engine.
"""
import logging
import pandas as pd
import duckdb

logger = logging.getLogger(__name__)

def execute_sql_query(df: pd.DataFrame, sql_query: str) -> pd.DataFrame:
    """
    Executes an arbitrary SQL query on a Pandas DataFrame via DuckDB without memory copy.
    
    Args:
        df (pd.DataFrame): Target pandas DataFrame (exposed as 'dataset' or 'df' in SQL).
        sql_query (str): SQL SELECT query to execute.
        
    Returns:
        pd.DataFrame: Query result as a new Pandas DataFrame.
    """
    if df.empty:
        logger.warning("Empty DataFrame passed to query engine.")
        return pd.DataFrame()
        
    try:
        # Create an in-memory DuckDB connection
        con = duckdb.connect(database=':memory:')
        
        # Register the dataframe under both 'dataset' and 'df' table names
        con.register('dataset', df)
        con.register('df', df)
        
        logger.info(f"Executing SQL query via DuckDB: {sql_query[:100]}...")
        result_df = con.execute(sql_query).df()
        con.close()
        
        return result_df
    except Exception as e:
        logger.error(f"DuckDB SQL execution error: {e}")
        raise ValueError(f"SQL Execution Error: {str(e)}")
