"""
SQL Database Connector Interface Utility.
Provides native connections to SQLite databases (and standard SQL URIs),
enabling table introspection and direct data ingestion into Pandas DataFrames.
"""
import sqlite3
import logging
import pandas as pd
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

def connect_sqlite_db(db_path: str) -> Optional[sqlite3.Connection]:
    """
    Establishes a connection to a local SQLite database file.
    """
    try:
        conn = sqlite3.connect(db_path)
        logger.info(f"Successfully connected to SQLite database at {db_path}")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to SQLite DB {db_path}: {e}")
        return None

def get_db_table_names(conn: sqlite3.Connection) -> List[str]:
    """
    Retrieves all table names present in the connected SQLite database.
    """
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        return tables
    except Exception as e:
        logger.error(f"Failed to retrieve table names: {e}")
        return []

def query_db_to_dataframe(conn: sqlite3.Connection, query: str) -> pd.DataFrame:
    """
    Executes a SQL query against the database connection and returns a Pandas DataFrame.
    """
    try:
        df = pd.read_sql_query(query, conn)
        logger.info(f"Successfully queried DB into DataFrame with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Database query execution failed: {e}")
        raise ValueError(f"Database Query Error: {str(e)}")
