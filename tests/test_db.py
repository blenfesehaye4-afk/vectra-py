import pytest
from vectra.db import VectraDB


def test_db_in_memory():
    db = VectraDB()
    col = db.create_collection("docs", dimension=8)
    assert "docs" in db.list_collections()
    assert db.get_collection("docs") == col

    with pytest.raises(ValueError):
        db.create_collection("docs", dimension=8)

    assert db.delete_collection("docs") is True
    assert "docs" not in db.list_collections()


def test_db_persistence(tmp_path):
    storage_path = tmp_path / "vectra_data"
    db = VectraDB(storage_dir=storage_path)
    col = db.create_collection("articles", dimension=4, metric="cosine")
    col.add(id="art-1", vector=[1, 0, 0, 0], text="article 1", metadata={"cat": "news"})
    db.persist()

    # Load fresh DB instance from disk
    db2 = VectraDB(storage_dir=storage_path)
    assert "articles" in db2.list_collections()
    col2 = db2.get_collection("articles")
    assert len(col2) == 1
    doc = col2.get("art-1")
    assert doc is not None
    assert doc.text == "article 1"
    assert doc.metadata == {"cat": "news"}
