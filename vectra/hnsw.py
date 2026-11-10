"""Hierarchical Navigable Small World (HNSW) graph index for log-time ANN search."""

import heapq
import math
import random
from typing import Dict, List, Optional, Set, Tuple

from .metrics import compute_distance, compute_score


class HNSWNode:
    """Node in an HNSW multi-layer graph representation."""

    def __init__(self, node_id: str, vector: List[float], level: int):
        self.node_id = node_id
        self.vector = vector
        self.level = level
        # Layer level -> Set of neighbor node_ids
        self.friends: Dict[int, Set[str]] = {lvl: set() for lvl in range(level + 1)}

    def get_neighbors(self, layer: int) -> Set[str]:
        return self.friends.get(layer, set())

    def add_neighbor(self, layer: int, neighbor_id: str) -> None:
        if layer in self.friends:
            self.friends[layer].add(neighbor_id)

    def set_neighbors(self, layer: int, neighbors: Set[str]) -> None:
        if layer in self.friends:
            self.friends[layer] = neighbors


class HNSWIndex:
    """Multi-layer Hierarchical Navigable Small World (HNSW) graph index."""

    def __init__(
        self,
        dimension: int,
        metric: str = "cosine",
        m: int = 16,
        ef_construction: int = 64,
        ef_search: int = 32,
        seed: Optional[int] = 42,
    ):
        self.dimension = dimension
        self.metric = metric
        self.m = m
        self.m0 = 2 * m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.m_l = 1.0 / math.log(m) if m > 1 else 1.0

        if seed is not None:
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random()

        self.enter_point: Optional[str] = None
        self.max_level = -1
        self.nodes: Dict[str, HNSWNode] = {}

    def _random_level(self) -> int:
        """Draw random level with exponential probability distribution."""
        unif = max(1e-9, self._rng.random())
        return int(-math.log(unif) * self.m_l)

    def _dist(self, v1: List[float], v2: List[float]) -> float:
        """Compute distance between two vectors."""
        return compute_distance(v1, v2, metric=self.metric)

    def __len__(self) -> int:
        return len(self.nodes)

    def contains(self, node_id: str) -> bool:
        return node_id in self.nodes
