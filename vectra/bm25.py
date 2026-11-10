"""Okapi BM25 sparse lexical search engine for keyword-based text retrieval."""

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple


class BM25Index:
    """Okapi BM25 inverted index with term frequency saturation and document length normalization."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        # doc_id -> token length
        self.doc_lengths: Dict[str, int] = {}
        # term -> doc_id -> count
        self.inverted_index: Dict[str, Dict[str, int]] = {}
        self.total_docs = 0
        self.avg_doc_len = 0.0

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Normalize and tokenize text into alphanumeric word stems."""
        return re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())

    def index_document(self, doc_id: str, text: str) -> None:
        """Add or update a document in the BM25 inverted index."""
        if doc_id in self.doc_lengths:
            self.remove_document(doc_id)

        tokens = self.tokenize(text)
        doc_len = len(tokens)
        self.doc_lengths[doc_id] = doc_len
        self.total_docs += 1

        counts = Counter(tokens)
        for term, freq in counts.items():
            if term not in self.inverted_index:
                self.inverted_index[term] = {}
            self.inverted_index[term][doc_id] = freq

        # Update average doc length
        self.avg_doc_len = sum(self.doc_lengths.values()) / max(1, self.total_docs)

    def remove_document(self, doc_id: str) -> bool:
        """Remove document from inverted index."""
        if doc_id not in self.doc_lengths:
            return False

        del self.doc_lengths[doc_id]
        self.total_docs -= 1

        for term in list(self.inverted_index.keys()):
            if doc_id in self.inverted_index[term]:
                del self.inverted_index[term][doc_id]
                if not self.inverted_index[term]:
                    del self.inverted_index[term]

        self.avg_doc_len = (sum(self.doc_lengths.values()) / max(1, self.total_docs)) if self.total_docs > 0 else 0.0
        return True

    def _idf(self, term: str) -> float:
        """Calculate Robertson-Spärck Jones Inverse Document Frequency."""
        n_q = len(self.inverted_index.get(term, {}))
        if n_q == 0:
            return 0.0
        return math.log(((self.total_docs - n_q + 0.5) / (n_q + 0.5)) + 1.0)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search documents scoring with BM25 formula."""
        if self.total_docs == 0:
            return []

        q_tokens = self.tokenize(query)
        if not q_tokens:
            return []

        doc_scores: Dict[str, float] = {}

        for term in q_tokens:
            if term not in self.inverted_index:
                continue

            idf = self._idf(term)
            postings = self.inverted_index[term]

            for doc_id, freq in postings.items():
                doc_len = self.doc_lengths[doc_id]
                denom = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1e-6, self.avg_doc_len)))
                score = idf * ((freq * (self.k1 + 1.0)) / max(1e-6, denom))
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score

        sorted_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def __len__(self) -> int:
        return self.total_docs
