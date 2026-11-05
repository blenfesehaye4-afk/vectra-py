"""Distance and similarity metrics for vector computations."""

from __future__ import annotations
import math
from typing import Sequence


def dot_product(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute dot product between two vectors."""
    if len(a) != len(b):
        raise ValueError(f"Vector dimensions do not match: {len(a)} != {len(b)}")
    return sum(x * y for x, y in zip(a, b))


def norm(v: Sequence[float]) -> float:
    """Compute Euclidean L2 norm of a vector."""
    return math.sqrt(sum(x * x for x in v))


def normalize(v: Sequence[float]) -> list[float]:
    """Normalize vector to unit length (L2 norm)."""
    n = norm(v)
    if n == 0.0:
        return [0.0] * len(v)
    return [x / n for x in v]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors (-1.0 to 1.0)."""
    if len(a) != len(b):
        raise ValueError(f"Vector dimensions do not match: {len(a)} != {len(b)}")
    norm_a = norm(a)
    norm_b = norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    val = dot_product(a, b) / (norm_a * norm_b)
    # Clamp due to floating point inaccuracies
    return max(-1.0, min(1.0, val))


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine distance: 1.0 - cosine_similarity (range 0.0 to 2.0)."""
    return 1.0 - cosine_similarity(a, b)


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute standard Euclidean (L2) distance."""
    if len(a) != len(b):
        raise ValueError(f"Vector dimensions do not match: {len(a)} != {len(b)}")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def manhattan_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute Manhattan (L1) distance."""
    if len(a) != len(b):
        raise ValueError(f"Vector dimensions do not match: {len(a)} != {len(b)}")
    return sum(abs(x - y) for x, y in zip(a, b))


def compute_distance_and_score(
    metric: str, a: Sequence[float], b: Sequence[float]
) -> tuple[float, float]:
    """
    Compute both distance and normalized ranking score for a given metric.
    Score is always higher-is-better (range 0.0 to 1.0).
    Distance is metric-specific (lower is closer).
    """
    metric = metric.lower()
    if metric == "cosine":
        sim = cosine_similarity(a, b)
        dist = 1.0 - sim
        score = (sim + 1.0) / 2.0  # map [-1, 1] to [0, 1]
        return dist, score
    elif metric in ("euclidean", "l2"):
        dist = euclidean_distance(a, b)
        score = 1.0 / (1.0 + dist)
        return dist, score
    elif metric in ("manhattan", "l1"):
        dist = manhattan_distance(a, b)
        score = 1.0 / (1.0 + dist)
        return dist, score
    elif metric in ("dot", "dot_product"):
        dp = dot_product(a, b)
        # Dot product distance is negative dot product; score normalized via sigmoid-like mapping
        dist = -dp
        score = 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, dp))))
        return dist, score
    else:
        raise ValueError(f"Unsupported metric '{metric}'. Expected 'cosine', 'euclidean', 'manhattan', or 'dot_product'.")
