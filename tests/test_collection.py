import pytest
from vectra.collection import Collection, Document


def test_collection_add_and_get():
    col = Collection("test_col", dimension=4, metric="cosine")
    doc = col.add(id="doc1", vector=[1.0, 0.0, 0.0, 0.0], text="Hello world", metadata={"author": "Blen"})
    assert len(col) == 1
    assert "doc1" in col
    assert col.get("doc1") == doc
    assert col.get("missing") is None


def test_collection_query():
    col = Collection("test_col", dimension=4, metric="cosine")
    col.add(id="doc1", vector=[1.0, 0.0, 0.0, 0.0], text="Unit 1", metadata={"status": "active"})
    col.add(id="doc2", vector=[0.0, 1.0, 0.0, 0.0], text="Unit 2", metadata={"status": "inactive"})
    col.add(id="doc3", vector=[0.8, 0.2, 0.0, 0.0], text="Unit 3", metadata={"status": "active"})

    # Query closest to [1, 0, 0, 0]
    results = col.query(vector=[1.0, 0.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0].id == "doc1"
    assert results[1].id == "doc3"

    # Query with metadata filter
    filtered = col.query(vector=[1.0, 0.0, 0.0, 0.0], filter={"status": "inactive"})
    assert len(filtered) == 1
    assert filtered[0].id == "doc2"


def test_collection_delete_and_delete_where():
    col = Collection("test_col", dimension=4)
    col.add("d1", [1, 0, 0, 0], metadata={"type": "a"})
    col.add("d2", [0, 1, 0, 0], metadata={"type": "b"})
    col.add("d3", [0, 0, 1, 0], metadata={"type": "a"})

    assert col.delete("d2") is True
    assert len(col) == 2

    deleted_count = col.delete_where({"type": "a"})
    assert deleted_count == 2
    assert len(col) == 0


def test_collection_deduplicate():
    col = Collection("test_col", dimension=4)
    col.add("d1", [1.0, 0.0, 0.0, 0.0])
    col.add("d2", [0.999, 0.001, 0.0, 0.0])  # Near duplicate
    col.add("d3", [0.0, 1.0, 0.0, 0.0])      # Dissimilar

    dupes = col.deduplicate(threshold=0.95)
    assert len(dupes) == 1
    assert (dupes[0][0] == "d1" and dupes[0][1] == "d2") or (dupes[0][0] == "d2" and dupes[0][1] == "d1")
