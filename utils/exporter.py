"""
Natural Language Data Export Engine Utility.
Converts query result DataFrames into downloadable file streams in multiple enterprise formats:
- CSV (.csv)
- Excel (.xlsx)
- JSON (.json)
"""
import io
import logging
import pandas as pd
from typing import Tuple

logger = logging.getLogger(__name__)

def export_dataframe(df: pd.DataFrame, file_format: str = "csv", base_filename: str = "exported_data") -> Tuple[bytes, str, str]:
    """
    Exports a DataFrame into binary file buffer stream.
    
    Args:
        df (pd.DataFrame): Target DataFrame.
        file_format (str): Desired format ('csv', 'xlsx', 'json').
        base_filename (str): Base file name.
        
    Returns:
        Tuple[bytes, str, str]: (binary_bytes, filename, mime_type)
    """
    fmt = file_format.lower().strip()
    
    if fmt == "excel" or fmt == "xlsx":
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name="Data_Export")
        bytes_data = buffer.getvalue()
        buffer.close()
        filename = f"{base_filename}.xlsx"
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        
    elif fmt == "json":
        json_str = df.to_json(orient="records", indent=2)
        bytes_data = json_str.encode("utf-8")
        filename = f"{base_filename}.json"
        mime = "application/json"
        
    else:  # Default CSV
        csv_str = df.to_csv(index=False)
        bytes_data = csv_str.encode("utf-8")
        filename = f"{base_filename}.csv"
        mime = "text/csv"
        
    logger.info(f"Exported DataFrame ({len(df)} rows) to format '{fmt}' ({len(bytes_data)} bytes)")
    return bytes_data, filename, mime
