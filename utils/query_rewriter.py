"""
Query Rewriting & HyDE (Hypothetical Document Embeddings) Generator Utility.
Transforms concise or ambiguous user queries into domain-enriched search queries
prior to vector embedding to maximize retrieval recall against tabular metadata.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def rewrite_query_for_search(query: str, schema_info: Optional[str] = None) -> str:
    """
    Expands and rewrites user prompts into rich search representations.
    
    Args:
        query (str): Original user prompt.
        schema_info (str, optional): Summary of available dataframe column names.
        
    Returns:
        str: Rewritten, enriched search prompt.
    """
    if not query or not query.strip():
        return ""
        
    query_clean = query.strip()
    
    # Common abbreviation and synonym expansions
    expansions = {
        "avg": "average mean summary statistics",
        "sum": "total aggregate summation sum",
        "max": "maximum top highest largest peak",
        "min": "minimum bottom lowest smallest",
        "std": "standard deviation variance dispersion",
        "cnt": "count total number frequency",
        "qty": "quantity volume amount count",
        "diff": "difference delta variance change",
    }
    
    words = query_clean.lower().split()
    expanded_words = []
    for word in words:
        expanded_words.append(word)
        if word in expansions:
            expanded_words.append(f"({expansions[word]})")
            
    enriched_query = " ".join(expanded_words)
    
    if schema_info:
        enriched_query = f"{enriched_query} [Available Schema Context: {schema_info}]"
        
    logger.info(f"Rewrote query '{query[:30]}' -> '{enriched_query[:50]}...'")
    return enriched_query
