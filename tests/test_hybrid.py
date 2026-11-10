import pytest
from vectra.bm25 import BM25Index
from vectra.hybrid import reciprocal_rank_fusion


def test_bm25_indexing_and_ranking():
    bm = BM25Index()
    bm.index_document("doc1", "Distributed vector database with Raft consensus")
    bm.index_document("doc2", "Natural language processing with transformers and embeddings")
    bm.index_document("doc3", "Consensus algorithms and distributed systems architecture")

    # Query matching doc1 and doc3
    results = bm.search("distributed consensus", top_k=2)
    assert len(results) == 2
    matched_ids = [doc_id for doc_id, score in results]
    assert "doc1" in matched_ids
    assert "doc3" in matched_ids

    # Update doc1 text
    bm.index_document("doc1", "Only simple text")
    results_after = bm.search("consensus")
    assert "doc1" not in [r[0] for r in results_after]


def test_bm25_document_deletion():
    bm = BM25Index()
    bm.index_document("d1", "apple orange banana")
    bm.index_document("d2", "apple grape strawberry")
    assert len(bm) == 2

    assert bm.remove_document("d1") is True
    assert len(bm) == 1
    assert bm.remove_document("d_nonexistent") is False

    res = bm.search("banana")
    assert len(res) == 0


def test_reciprocal_rank_fusion_scoring():
    dense_results = [("doc_a", 0.95), ("doc_b", 0.85), ("doc_c", 0.75)]
    sparse_results = [("doc_b", 12.0), ("doc_a", 10.5), ("doc_d", 8.0)]

    fused = reciprocal_rank_fusion(dense_results, sparse_results, k=60, top_k=4)
    assert len(fused) == 4

    # Both doc_a and doc_b are high in both rankings, should be at the top
    top_ids = [item["id"] for item in fused]
    assert top_ids[0] in ("doc_a", "doc_b")
    assert top_ids[1] in ("doc_a", "doc_b")

    # Verify score provenance
    doc_b_entry = next(item for item in fused if item["id"] == "doc_b")
    assert doc_b_entry["dense_rank"] == 2
    assert doc_b_entry["sparse_rank"] == 1
