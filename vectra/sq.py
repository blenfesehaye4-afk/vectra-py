"""Scalar Quantization (SQ8) converting 32-bit float vectors into compact 8-bit integers."""

import math
from typing import List, Sequence, Tuple


class ScalarQuantizer:
    """Symmetric 8-bit Scalar Quantizer [-127, 127] preserving angular geometric properties."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.scales: List[float] = []
        self.is_fitted = False

    def fit(self, vectors: Sequence[Sequence[float]]) -> None:
        """Derive per-dimension absolute maximum scales from sample vectors."""
        if not vectors:
            raise ValueError("Cannot fit ScalarQuantizer with empty dataset")

        self.scales = []
        for d in range(self.dimension):
            max_abs = max(abs(v[d]) for v in vectors)
            if max_abs <= 1e-9:
                self.scales.append(1.0)
            else:
                self.scales.append(max_abs)

        self.is_fitted = True

    def quantize(self, vector: Sequence[float]) -> List[int]:
        """Quantize float vector into int8 elements [-127, 127]."""
        if not self.is_fitted:
            raise RuntimeError("ScalarQuantizer must be fitted before quantizing")

        int8_vec: List[int] = []
        for d in range(self.dimension):
            val = vector[d]
            scale = self.scales[d]
            normalized = val / scale
            clamped = max(-1.0, min(1.0, normalized))
            q = int(round(clamped * 127.0))
            int8_vec.append(q)

        return int8_vec

    def dequantize(self, quantized_vector: Sequence[int]) -> List[float]:
        """Reconstruct approximate float vector from int8 elements."""
        if not self.is_fitted:
            raise RuntimeError("ScalarQuantizer must be fitted before dequantizing")

        recon: List[float] = []
        for d in range(self.dimension):
            q = quantized_vector[d]
            scale = self.scales[d]
            recon.append((q / 127.0) * scale)

        return recon

    def fast_cosine(self, q1: Sequence[int], q2: Sequence[int]) -> float:
        """Fast approximate cosine similarity directly on quantized integer vectors."""
        dot = sum(x * y for x, y in zip(q1, q2))
        norm1 = math.sqrt(sum(x * x for x in q1))
        norm2 = math.sqrt(sum(y * y for y in q2))
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return max(-1.0, min(1.0, dot / (norm1 * norm2)))
