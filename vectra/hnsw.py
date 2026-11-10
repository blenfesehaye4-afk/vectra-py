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

    def _search_layer(
        self,
        query: List[float],
        enter_points: List[str],
        ef: int,
        layer: int,
    ) -> List[Tuple[float, str]]:
        """
        Greedy search at a single graph layer.
        Returns list of (distance, node_id) sorted by distance ascending.
        """
        visited: Set[str] = set(enter_points)
        # candidates: min-heap of (dist, node_id)
        candidates: List[Tuple[float, str]] = []
        # nearest: max-heap of (-dist, node_id) storing best ef elements
        nearest: List[Tuple[float, str]] = []

        for ep in enter_points:
            d = self._dist(query, self.nodes[ep].vector)
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(nearest, (-d, ep))

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            furthest_dist = -nearest[0][0]

            if c_dist > furthest_dist and len(nearest) >= ef:
                break

            for neighbor_id in self.nodes[c_id].get_neighbors(layer):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    d_neighbor = self._dist(query, self.nodes[neighbor_id].vector)
                    furthest_dist = -nearest[0][0]

                    if d_neighbor < furthest_dist or len(nearest) < ef:
                        heapq.heappush(candidates, (d_neighbor, neighbor_id))
                        heapq.heappush(nearest, (-d_neighbor, neighbor_id))

                        if len(nearest) > ef:
                            heapq.heappop(nearest)

        result = [(-neg_d, nid) for neg_d, nid in nearest]
        result.sort(key=lambda x: x[0])
        return result

    def search_knn(
        self,
        query: List[float],
        k: int = 10,
        ef: Optional[int] = None,
    ) -> List[Tuple[str, float, float]]:
        """
        Search k nearest neighbors.
        Returns list of (node_id, score, distance).
        """
        if not self.nodes or self.enter_point is None:
            return []

        curr_ef = ef or max(self.ef_search, k)
        curr_ep = [self.enter_point]

        # Greedy zoom down from top layer to layer 1
        for layer in range(self.max_level, 0, -1):
            candidates = self._search_layer(query, curr_ep, ef=1, layer=layer)
            if candidates:
                curr_ep = [candidates[0][1]]

        # Layer 0 search with ef candidates
        nearest_layer0 = self._search_layer(query, curr_ep, ef=curr_ef, layer=0)

        # Slice top k and convert to scores
        top_k = nearest_layer0[:k]
        output: List[Tuple[str, float, float]] = []
        for dist, nid in top_k:
            score = compute_score(dist, metric=self.metric)
            output.append((nid, score, dist))

        return output
