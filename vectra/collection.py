"""Vector collection managing documents, indexing, and similarity searches."""

from __future__ import annotations
from dataclasses import dataclass, field
import sys
from typing import Any, Mapping, Sequence
import uuid

from .metrics import compute_distance_and_score, normalize
from .filter import matches_filter
from .embedding import BaseEmbedder, TfIdfEmbedder


@dataclass
class Document:
    """A document containing a unique ID, dense vector embedding, optional raw text, and metadata."""
    id: str
    vector: list[float]
    text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "vector": self.vector,
            "text": self.text,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Document:
        return cls(
            id=str(data["id"]),
            vector=list(data["vector"]),
            text=data.get("text"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class SearchResult:
    """Result of a vector similarity search query."""
    id: str
    score: float
    distance: float
    text: str | None
    metadata: dict[str, Any]
    vector: list[float] | None = None

    def to_dict(self, include_vector: bool = False) -> dict[str, Any]:
        out = {
            "id": self.id,
            "score": round(self.score, 4),
            "distance": round(self.distance, 4),
            "text": self.text,
            "metadata": self.metadata,
        }
        if include_vector and self.vector is not None:
            out["vector"] = self.vector
        return out


class Collection:
    """
    Manages vector embeddings and provides fast exact nearest neighbor
    search with rich metadata filtering.
    """

    def __init__(
        self,
        name: str,
        dimension: int | None = None,
        metric: str = "cosine",
        embedder: BaseEmbedder | None = None,
    ):
        self.name = name
        self.metric = metric.lower()
        self._embedder = embedder or TfIdfEmbedder(dim=dimension or 128)
        self.dimension = dimension or self._embedder.dimension
        self._documents: dict[str, Document] = {}

    @property
    def embedder(self) -> BaseEmbedder:
        return self._embedder

    @embedder.setter
    def embedder(self, value: BaseEmbedder) -> None:
        self._embedder = value
        if self.dimension is None:
            self.dimension = value.dimension

    def __len__(self) -> int:
        return len(self._documents)

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._documents

    def add(
        self,
        id: str | None = None,
        vector: Sequence[float] | None = None,
        text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        """Insert or replace a document in the collection."""
        doc_id = str(id) if id is not None else str(uuid.uuid4())
        meta = dict(metadata) if metadata else {}

        if vector is None:
            if text is None:
                raise ValueError("Must provide either a vector or text to embed")
            vec = self._embedder.embed_text(text)
        else:
            vec = list(vector)

        if len(vec) != self.dimension:
            raise ValueError(
                f"Vector dimension {len(vec)} does not match collection dimension {self.dimension}"
            )

        doc = Document(id=doc_id, vector=vec, text=text, metadata=meta)
        self._documents[doc_id] = doc
        return doc

    def add_batch(self, docs: Sequence[Document | dict[str, Any]]) -> int:
        """Add multiple documents in a batch."""
        count = 0
        for item in docs:
            if isinstance(item, Document):
                if len(item.vector) != self.dimension:
                    raise ValueError(
                        f"Vector dimension {len(item.vector)} does not match collection dimension {self.dimension}"
                    )
                self._documents[item.id] = item
                count += 1
            elif isinstance(item, dict):
                self.add(
                    id=item.get("id"),
                    vector=item.get("vector"),
                    text=item.get("text"),
                    metadata=item.get("metadata"),
                )
                count += 1
        return count

    def get(self, doc_id: str) -> Document | None:
        """Retrieve a document by ID."""
        return self._documents.get(doc_id)

    def delete(self, doc_id: str) -> bool:
        """Delete a document by ID. Returns True if found and deleted."""
        return self._documents.pop(doc_id, None) is not None

    def delete_where(self, filter_dict: Mapping[str, Any]) -> int:
        """Delete all documents matching a metadata filter query."""
        to_delete = [
            doc_id
            for doc_id, doc in self._documents.items()
            if matches_filter(doc.metadata, filter_dict)
        ]
        for doc_id in to_delete:
            del self._documents[doc_id]
        return len(to_delete)

    def query(
        self,
        vector: Sequence[float] | None = None,
        text: str | None = None,
        top_k: int = 5,
        filter: Mapping[str, Any] | None = None,
        score_threshold: float | None = None,
        include_vectors: bool = False,
    ) -> list[SearchResult]:
        """
        Query the collection for the nearest neighbors.

        Args:
            vector: Query dense vector.
            text: Query text to embed with the collection embedder.
            top_k: Maximum number of search results to return.
            filter: Metadata filter dictionary.
            score_threshold: Minimum similarity score threshold (0.0 to 1.0).
            include_vectors: Whether to include vectors in search results.
        """
        if vector is None:
            if text is None:
                raise ValueError("Must provide either a query vector or query text")
            query_vec = self._embedder.embed_text(text)
        else:
            query_vec = list(vector)

        if len(query_vec) != self.dimension:
            raise ValueError(
                f"Query vector dimension {len(query_vec)} != collection dimension {self.dimension}"
            )

        candidates: list[SearchResult] = []
        for doc in self._documents.values():
            if filter and not matches_filter(doc.metadata, filter):
                continue

            dist, score = compute_distance_and_score(self.metric, query_vec, doc.vector)
            if score_threshold is not None and score < score_threshold:
                continue

            candidates.append(
                SearchResult(
                    id=doc.id,
                    score=score,
                    distance=dist,
                    text=doc.text,
                    metadata=doc.metadata,
                    vector=doc.vector if include_vectors else None,
                )
            )

        # Sort results: descending by score (higher is better)
        candidates.sort(key=lambda r: r.score, reverse=True)
        return candidates[:top_k]

    def deduplicate(
        self, threshold: float = 0.95
    ) -> list[tuple[str, str, float]]:
        """
        Find pairs of documents whose similarity score exceeds the threshold.
        Returns list of (id_a, id_b, similarity_score).
        """
        items = list(self._documents.values())
        n = len(items)
        duplicates: list[tuple[str, str, float]] = []

        for i in range(n):
            for j in range(i + 1, n):
                _, score = compute_distance_and_score(self.metric, items[i].vector, items[j].vector)
                if score >= threshold:
                    duplicates.append((items[i].id, items[j].id, round(score, 4)))

        duplicates.sort(key=lambda d: d[2], reverse=True)
        return duplicates

    def stats(self) -> dict[str, Any]:
        """Return operational statistics for the collection."""
        doc_count = len(self._documents)
        sample_meta_keys = set()
        for doc in self._documents.values():
            sample_meta_keys.update(doc.metadata.keys())

        approx_bytes = sys.getsizeof(self._documents) + sum(
            sys.getsizeof(d.vector) + sys.getsizeof(d.text or "") + sys.getsizeof(d.metadata)
            for d in self._documents.values()
        )

        return {
            "name": self.name,
            "count": doc_count,
            "dimension": self.dimension,
            "metric": self.metric,
            "approx_memory_bytes": approx_bytes,
            "metadata_fields": sorted(list(sample_meta_keys)),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize collection configuration and documents to dictionary."""
        return {
            "name": self.name,
            "dimension": self.dimension,
            "metric": self.metric,
            "documents": [doc.to_dict() for doc in self._documents.values()],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], embedder: BaseEmbedder | None = None) -> Collection:
        """Deserialize collection from dictionary."""
        col = cls(
            name=data["name"],
            dimension=data["dimension"],
            metric=data.get("metric", "cosine"),
            embedder=embedder,
        )
        for doc_data in data.get("documents", []):
            doc = Document.from_dict(doc_data)
            col._documents[doc.id] = doc
        return col
