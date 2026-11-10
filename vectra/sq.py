"""Scalar Quantization (SQ8) converting 32-bit float vectors into compact 8-bit integers."""

import math
from typing import List, Sequence, Tuple


class ScalarQuantizer:
    """Scalar Quantizer mapping continuous floats to 8-bit signed integers [-128, 127]."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.min_vals: List[float] = []
        self.max_vals: List[float] = []
        self.is_fitted = False

    def fit(self, vectors: Sequence[Sequence[float]]) -> None:
        """Derive per-dimension dynamic range [min, max] from sample vectors."""
        if not vectors:
            raise ValueError("Cannot fit ScalarQuantizer with empty dataset")

        self.min_vals = [min(v[d] for v in vectors) for d in range(self.dimension)]
        self.max_vals = [max(v[d] for v in vectors) for d in range(self.dimension)]

        # Prevent zero division if constant column
        for d in range(self.dimension):
            if math.isclose(self.min_vals[d], self.max_vals[d], abs_tol=1e-7):
                self.max_vals[d] = self.min_vals[d] + 1e-4

        self.is_fitted = True

    def quantize(self, vector: Sequence[float]) -> List[int]:
        """Quantize float vector into int8 elements [-128, 127]."""
        if not self.is_fitted:
            raise RuntimeError("ScalarQuantizer must be fitted before quantizing")

        int8_vec: List[int] = []
        for d in range(self.dimension):
            val = vector[d]
            min_v = self.min_vals[d]
            max_v = self.max_vals[d]

            # Scale to [0, 255], shift to [-128, 127]
            normed = (val - min_v) / (max_v - min_v)
            clamped = max(0.0, min(1.0, normed))
            q = int(round(clamped * 255.0)) - 128
            int8_vec.append(q)

        return int8_vec

    def dequantize(self, quantized_vector: Sequence[int]) -> List[float]:
        """Reconstruct approximate float vector from int8 elements."""
        if not self.is_fitted:
            raise RuntimeError("ScalarQuantizer must be fitted before dequantizing")

        recon: List[float] = []
        for d in range(self.dimension):
            q = quantized_vector[d]
            min_v = self.min_vals[d]
            max_v = self.max_vals[d]

            normed = (q + 128) / 255.0
            recon.append(min_v + normed * (max_v - min_v))

        return recon

    def fast_cosine(self, q1: Sequence[int], q2: Sequence[int]) -> float:
        """Fast approximate cosine similarity directly on quantized integer vectors."""
        dot = sum(x * y for x, y in zip(q1, q2))
        norm1 = math.sqrt(sum(x * x for x in q1))
        norm2 = math.sqrt(sum(y * y for y in q2))
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return max(-1.0, min(1.0, dot / (norm1 * norm2)))
