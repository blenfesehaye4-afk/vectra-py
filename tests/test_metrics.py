import pytest
import math
from vectra import metrics


def test_dot_product():
    assert metrics.dot_product([1.0, 2.0], [3.0, 4.0]) == 11.0
    with pytest.raises(ValueError):
        metrics.dot_product([1.0], [1.0, 2.0])


def test_norm_and_normalize():
    vec = [3.0, 4.0]
    assert math.isclose(metrics.norm(vec), 5.0)
    normed = metrics.normalize(vec)
    assert math.isclose(metrics.norm(normed), 1.0)
    assert metrics.normalize([0.0, 0.0]) == [0.0, 0.0]


def test_cosine_similarity():
    # Identical vectors
    assert math.isclose(metrics.cosine_similarity([1.0, 2.0], [1.0, 2.0]), 1.0)
    # Orthogonal vectors
    assert math.isclose(metrics.cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)
    # Opposite vectors
    assert math.isclose(metrics.cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0)
    # Zero vector
    assert metrics.cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_euclidean_distance():
    assert math.isclose(metrics.euclidean_distance([0.0, 0.0], [3.0, 4.0]), 5.0)


def test_manhattan_distance():
    assert metrics.manhattan_distance([1.0, 2.0], [4.0, 6.0]) == 7.0


def test_compute_distance_and_score():
    dist, score = metrics.compute_distance_and_score("cosine", [1.0, 0.0], [1.0, 0.0])
    assert math.isclose(dist, 0.0)
    assert math.isclose(score, 1.0)

    dist, score = metrics.compute_distance_and_score("euclidean", [0.0, 0.0], [0.0, 0.0])
    assert dist == 0.0
    assert score == 1.0

    with pytest.raises(ValueError):
        metrics.compute_distance_and_score("invalid_metric", [1.0], [1.0])
