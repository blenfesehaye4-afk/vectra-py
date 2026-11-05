"""Text embedding interface and lightweight built-in vectorizers."""

from __future__ import annotations
import math
import re
from abc import ABC, abstractmethod
from typing import Sequence


STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than",
    "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
    "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's",
    "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom",
    "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll",
    "you're", "you've", "your", "yours", "yourself", "yourselves"
}


class BaseEmbedder(ABC):
    """Abstract interface for text embedding models."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the generated vector."""
        ...

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Convert a single text string into a normalized dense vector."""
        ...

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Batch vectorization."""
        return [self.embed_text(t) for t in texts]


def tokenize(text: str) -> list[str]:
    """Clean, lowercase, and tokenize text into words."""
    cleaned = text.lower()
    tokens = re.findall(r"\b[a-z0-9_-]{2,}\b", cleaned)
    return [t for t in tokens if t not in STOP_WORDS]


class TfIdfEmbedder(BaseEmbedder):
    """
    Lightweight, deterministic feature-hashing TF-IDF text embedder.
    Requires no pre-trained weights or internet connection.
    Uses Murmur/FNV-style hashing with sublinear term-frequency scaling and L2 normalization.
    """

    def __init__(self, dim: int = 128, sublinear_tf: bool = True):
        if dim <= 0:
            raise ValueError("Embedding dimension must be positive")
        self._dim = dim
        self._sublinear_tf = sublinear_tf

    @property
    def dimension(self) -> int:
        return self._dim

    def _hash_token(self, token: str) -> tuple[int, int]:
        """Hash a token to (dimension_index, sign [-1, 1]) using 32-bit FNV-1a."""
        h = 2166136261
        for char in token:
            h = (h ^ ord(char)) * 16777619
            h &= 0xFFFFFFFF
        idx = h % self._dim
        # Use second bit for sign to produce unbiased expectation
        sign = 1 if (h & 0x80000000) else -1
        return idx, sign

    def embed_text(self, text: str) -> list[float]:
        tokens = tokenize(text)
        if not tokens:
            return [0.0] * self._dim

        # Calculate term frequencies
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1

        vec = [0.0] * self._dim
        for token, count in tf.items():
            idx, sign = self._hash_token(token)
            weight = (1.0 + math.log(count)) if self._sublinear_tf else float(count)
            vec[idx] += sign * weight

        # L2 normalize
        squared_sum = sum(x * x for x in vec)
        norm = math.sqrt(squared_sum)
        if norm > 0:
            vec = [round(x / norm, 6) for x in vec]
        return vec
