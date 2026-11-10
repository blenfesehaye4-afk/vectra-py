"""Reciprocal Rank Fusion (RRF) combining dense semantic search and sparse lexical BM25 retrieval."""

from typing import Any, Dict, List, Optional, Tuple


def reciprocal_rank_fusion(
    dense_results: List[Tuple[str, float]],
    sparse_results: List[Tuple[str, float]],
    k: int = 60,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Merge dense vector results and sparse lexical results using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_score(doc) = dense_weight / (k + rank_dense) + sparse_weight / (k + rank_sparse)

    Args:
        dense_results: List of (doc_id, score) from semantic vector search.
        sparse_results: List of (doc_id, score) from BM25 lexical search.
        k: Smoothing constant (standard default is 60).
        dense_weight: Relative importance of semantic dense vector search.
        sparse_weight: Relative importance of sparse lexical BM25 search.
        top_k: Maximum number of fused items to return.

    Returns:
        List of dicts containing doc_id, rrf_score, dense_rank, and sparse_rank.
    """
    fused_scores: Dict[str, float] = {}
    dense_ranks: Dict[str, int] = {}
    sparse_ranks: Dict[str, int] = {}

    for rank, (doc_id, _) in enumerate(dense_results, start=1):
        dense_ranks[doc_id] = rank
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + (dense_weight / (k + rank))

    for rank, (doc_id, _) in enumerate(sparse_results, start=1):
        sparse_ranks[doc_id] = rank
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + (sparse_weight / (k + rank))

    sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    output: List[Dict[str, Any]] = []
    for doc_id, score in sorted_docs[:top_k]:
        output.append({
            "id": doc_id,
            "score": round(score, 6),
            "dense_rank": dense_ranks.get(doc_id),
            "sparse_rank": sparse_ranks.get(doc_id),
        })

    return output
