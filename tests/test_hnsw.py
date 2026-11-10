import math
import random
import pytest
from vectra.hnsw import HNSWIndex
from vectra.metrics import cosine_similarity, euclidean_distance


def test_hnsw_basic_insertion_and_search():
    index = HNSWIndex(dimension=4, metric="cosine", m=4, ef_construction=16, ef_search=16)

    # Insert distinct directional vectors
    index.insert("v1", [1.0, 0.0, 0.0, 0.0])
    index.insert("v2", [0.0, 1.0, 0.0, 0.0])
    index.insert("v3", [0.0, 0.0, 1.0, 0.0])
    index.insert("v4", [0.0, 0.0, 0.0, 1.0])

    assert len(index) == 4
    assert index.contains("v1") is True

    # Search query identical to v2
    results = index.search_knn([0.0, 1.0, 0.0, 0.0], k=2)
    assert len(results) == 2
    top_id, top_score, top_dist = results[0]
    assert top_id == "v2"
    assert math.isclose(top_score, 1.0, abs_tol=1e-5)


def test_hnsw_high_recall_vs_linear_scan():
    rng = random.Random(1337)
    dim = 8
    num_vectors = 100

    index = HNSWIndex(dimension=dim, metric="euclidean", m=8, ef_construction=32, ef_search=32, seed=42)
    raw_vectors = {}

    for i in range(num_vectors):
        vec = [rng.gauss(0, 1) for _ in range(dim)]
        vid = f"id_{i}"
        raw_vectors[vid] = vec
        index.insert(vid, vec)

    # Perform 10 random queries and evaluate top-1 recall against brute force
    recalls = 0
    total_queries = 10

    for _ in range(total_queries):
        q = [rng.gauss(0, 1) for _ in range(dim)]

        # Linear brute force ground truth
        brute_truth = sorted(
            raw_vectors.keys(),
            key=lambda k: euclidean_distance(q, raw_vectors[k])
        )[0]

        hnsw_res = index.search_knn(q, k=1)
        if hnsw_res and hnsw_res[0][0] == brute_truth:
            recalls += 1

    # Expect >= 90% top-1 recall
    assert (recalls / total_queries) >= 0.90


def test_hnsw_deletion_and_reassignment():
    index = HNSWIndex(dimension=2, metric="cosine")
    index.insert("a", [1.0, 0.0])
    index.insert("b", [0.0, 1.0])

    assert len(index) == 2
    assert index.delete("a") is True
    assert index.contains("a") is False
    assert len(index) == 1

    res = index.search_knn([1.0, 0.0], k=2)
    assert len(res) == 1
    assert res[0][0] == "b"


def test_hnsw_dimension_mismatch():
    index = HNSWIndex(dimension=3)
    with pytest.raises(ValueError, match="Vector dimension 2 does not match"):
        index.insert("bad", [1.0, 2.0])
