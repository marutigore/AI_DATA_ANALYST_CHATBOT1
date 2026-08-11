"""
Hybrid BM25 + FAISS Vector Search Engine.
Combines sparse keyword matching (BM25 algorithm) with dense vector search (FAISS)
using Reciprocal Rank Fusion (RRF) to deliver superior retrieval accuracy for both exact column matching and semantic concepts.
"""
import math
import logging
from collections import Counter
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

import re

class BM25Scorer:
    """Lightweight self-contained BM25 text scorer."""
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.corpus_size = len(corpus)
        self.doc_tokens = [re.findall(r'\w+', doc.lower()) for doc in corpus]
        self.avg_doc_len = sum(len(doc) for doc in self.doc_tokens) / max(self.corpus_size, 1)
        self.doc_freqs: Dict[str, int] = Counter()
        for doc in self.doc_tokens:
            for token in set(doc):
                self.doc_freqs[token] += 1
                
    def get_scores(self, query: str) -> List[float]:
        query_tokens = re.findall(r'\w+', query.lower())
        scores = [0.0] * self.corpus_size

        
        for token in query_tokens:
            if token not in self.doc_freqs:
                continue
            # Calculate IDF
            df = self.doc_freqs[token]
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)
            
            for i, doc in enumerate(self.doc_tokens):
                tf = doc.count(token)
                if tf == 0:
                    continue
                doc_len = len(doc)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                scores[i] += idf * (numerator / denominator)
                
        return scores

def hybrid_rrf_search(
    query: str, 
    query_embedding: List[float], 
    documents: List[str], 
    vector_store_matches: List[Dict[str, Any]], 
    top_k: int = 3,
    rrf_k: int = 60
) -> List[Dict[str, Any]]:
    """
    Fuses BM25 keyword rankings with FAISS vector matches using Reciprocal Rank Fusion.
    """
    if not documents:
        return []
        
    # 1. BM25 Ranking
    bm25 = BM25Scorer(documents)
    bm25_scores = bm25.get_scores(query)
    
    # Sort docs by BM25 score descending
    bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
    doc_to_bm25_rank = {}
    for rank, (doc_idx, score) in enumerate(bm25_ranked):
        if score > 0:
            doc_to_bm25_rank[doc_idx] = rank
            
    # 2. Map FAISS matches to document indices
    doc_to_faiss_rank = {}
    for rank, match in enumerate(vector_store_matches):
        doc_content = match.get("content")
        if doc_content in documents:
            doc_idx = documents.index(doc_content)
            doc_to_faiss_rank[doc_idx] = rank
            
    # 3. Calculate RRF Score for each document
    rrf_scores = {}
    for doc_idx in range(len(documents)):
        bm25_rank = doc_to_bm25_rank.get(doc_idx, 9999)
        faiss_rank = doc_to_faiss_rank.get(doc_idx, 9999)
        
        bm25_score_part = 1.0 / (rrf_k + bm25_rank + 1) if bm25_rank != 9999 else 0.0
        faiss_score_part = 1.0 / (rrf_k + faiss_rank + 1) if faiss_rank != 9999 else 0.0
        
        rrf_scores[doc_idx] = bm25_score_part + faiss_score_part

        
    # 4. Sort documents by combined RRF score
    sorted_doc_indices = sorted(rrf_scores.keys(), key=lambda idx: rrf_scores[idx], reverse=True)[:top_k]
    
    results = []
    for idx in sorted_doc_indices:
        results.append({
            "content": documents[idx],
            "rrf_score": round(rrf_scores[idx], 6)
        })
        
    logger.info(f"Hybrid RRF Search fused top-{len(results)} matches for query '{query[:30]}'")
    return results
