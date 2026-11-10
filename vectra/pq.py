"""Product Quantization (PQ) for vector compression and fast Asymmetric Distance Computation (ADC)."""

import math
import random
from typing import List, Sequence, Tuple


class ProductQuantizer:
    """Product Quantizer compressing high-dimensional floating-point vectors into compact codes."""

    def __init__(self, dimension: int, num_subvectors: int = 4, num_centroids: int = 16, seed: int = 42):
        if dimension % num_subvectors != 0:
            raise ValueError(f"Dimension {dimension} must be divisible by num_subvectors {num_subvectors}")

        self.dimension = dimension
        self.num_subvectors = num_subvectors
        self.d_sub = dimension // num_subvectors
        self.num_centroids = num_centroids
        self._rng = random.Random(seed)

        # codebooks[m] -> list of centroids for m-th subvector, each centroid is list of length d_sub
        self.codebooks: List[List[List[float]]] = []
        self.is_fitted = False

    def fit(self, training_vectors: Sequence[Sequence[float]], max_iters: int = 10) -> None:
        """Train k-means centroids on each subspace using training vectors."""
        if len(training_vectors) < self.num_centroids:
            raise ValueError(f"Need at least {self.num_centroids} vectors to train PQ codebook")

        self.codebooks = []

        for m_idx in range(self.num_subvectors):
            start = m_idx * self.d_sub
            end = start + self.d_sub
            sub_vecs = [v[start:end] for v in training_vectors]

            # 1. Initialize centroids randomly from data
            centroids = [list(sub_vecs[i]) for i in self._rng.sample(range(len(sub_vecs)), self.num_centroids)]

            # 2. Lloyd-Forgy K-Means iterations
            for _ in range(max_iters):
                clusters: List[List[List[float]]] = [[] for _ in range(self.num_centroids)]

                for sv in sub_vecs:
                    best_c = 0
                    best_dist = float("inf")
                    for c_idx, c in enumerate(centroids):
                        d = sum((x - y) ** 2 for x, y in zip(sv, c))
                        if d < best_dist:
                            best_dist = d
                            best_c = c_idx
                    clusters[best_c].append(sv)

                # Recompute centroids
                for c_idx in range(self.num_centroids):
                    if clusters[c_idx]:
                        n = len(clusters[c_idx])
                        new_center = [sum(pt[i] for pt in clusters[c_idx]) / n for i in range(self.d_sub)]
                        centroids[c_idx] = new_center

            self.codebooks.append(centroids)

        self.is_fitted = True

    def encode(self, vector: Sequence[float]) -> List[int]:
        """Compress full-dimensional vector into M integer cluster IDs."""
        if not self.is_fitted:
            raise RuntimeError("ProductQuantizer must be fitted before encoding")

        codes: List[int] = []
        for m_idx in range(self.num_subvectors):
            start = m_idx * self.d_sub
            end = start + self.d_sub
            sv = vector[start:end]

            best_c = 0
            best_dist = float("inf")
            for c_idx, c in enumerate(self.codebooks[m_idx]):
                d = sum((x - y) ** 2 for x, y in zip(sv, c))
                if d < best_dist:
                    best_dist = d
                    best_c = c_idx
            codes.append(best_c)

        return codes

    def decode(self, codes: Sequence[int]) -> List[float]:
        """Reconstruct approximate vector from codes."""
        if not self.is_fitted:
            raise RuntimeError("ProductQuantizer must be fitted before decoding")

        recon: List[float] = []
        for m_idx, c_idx in enumerate(codes):
            recon.extend(self.codebooks[m_idx][c_idx])
        return recon

    def compute_distance_table(self, query: Sequence[float]) -> List[List[float]]:
        """Precompute squared distance table between query subvectors and all centroids."""
        table: List[List[float]] = []
        for m_idx in range(self.num_subvectors):
            start = m_idx * self.d_sub
            end = start + self.d_sub
            q_sub = query[start:end]

            sub_dists = [sum((qx - cx) ** 2 for qx, cx in zip(q_sub, c)) for c in self.codebooks[m_idx]]
            table.append(sub_dists)
        return table

    def asymmetric_distance_from_table(self, dist_table: List[List[float]], codes: Sequence[int]) -> float:
        """Asymmetric distance computation (ADC) in O(M) lookup operations."""
        sq_dist = sum(dist_table[m_idx][codes[m_idx]] for m_idx in range(self.num_subvectors))
        return math.sqrt(max(0.0, sq_dist))
