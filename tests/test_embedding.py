import pytest
import math
from vectra.embedding import TfIdfEmbedder, tokenize
from vectra import metrics


def test_tokenize():
    text = "FastAPI and Python 3.12 for High-Performance APIs!"
    tokens = tokenize(text)
    assert "fastapi" in tokens
    assert "python" in tokens
    # stop words like 'and', 'for' should be removed
    assert "and" not in tokens
    assert "for" not in tokens


def test_tfidf_embedder():
    embedder = TfIdfEmbedder(dim=64)
    assert embedder.dimension == 64

    vec = embedder.embed_text("Vector databases enable semantic similarity search")
    assert len(vec) == 64
    assert math.isclose(metrics.norm(vec), 1.0, abs_tol=1e-3)

    # Empty text check
    zero_vec = embedder.embed_text("")
    assert len(zero_vec) == 64
    assert all(x == 0.0 for x in zero_vec)


def test_semantic_similarity_ranking():
    embedder = TfIdfEmbedder(dim=128)
    v_ai = embedder.embed_text("deep neural networks artificial intelligence machine learning")
    v_ml = embedder.embed_text("machine learning neural network models")
    v_cooking = embedder.embed_text("baking delicious sourdough bread with yeast and flour")

    sim_ai_ml = metrics.cosine_similarity(v_ai, v_ml)
    sim_ai_cooking = metrics.cosine_similarity(v_ai, v_cooking)

    assert sim_ai_ml > sim_ai_cooking
