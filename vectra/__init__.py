"""
Vectra: Lightweight, embeddable vector database & hybrid search engine.
"""

from .db import VectraDB
from .collection import Collection, Document, SearchResult
from .embedding import BaseEmbedder, TfIdfEmbedder
from .filter import matches_filter
from .hnsw import HNSWIndex
from .pq import ProductQuantizer
from .sq import ScalarQuantizer
from .bm25 import BM25Index
from .hybrid import reciprocal_rank_fusion
from .storage import BinaryStorage
from .server import app as server_app
from . import metrics

__version__ = "0.3.0"
__all__ = [
    "VectraDB",
    "Collection",
    "Document",
    "SearchResult",
    "BaseEmbedder",
    "TfIdfEmbedder",
    "matches_filter",
    "HNSWIndex",
    "ProductQuantizer",
    "ScalarQuantizer",
    "BM25Index",
    "reciprocal_rank_fusion",
    "BinaryStorage",
    "server_app",
    "metrics",
]
