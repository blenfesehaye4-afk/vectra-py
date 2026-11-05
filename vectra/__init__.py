"""
Vectra: Lightweight, embeddable vector database & semantic similarity search engine.
"""

from .db import VectraDB
from .collection import Collection, Document, SearchResult
from .embedding import BaseEmbedder, TfIdfEmbedder
from .filter import matches_filter
from . import metrics

__version__ = "0.1.0"
__all__ = [
    "VectraDB",
    "Collection",
    "Document",
    "SearchResult",
    "BaseEmbedder",
    "TfIdfEmbedder",
    "matches_filter",
    "metrics",
]
