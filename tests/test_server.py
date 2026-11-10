import pytest
from fastapi.testclient import TestClient
from vectra.server import app, set_db
from vectra.db import VectraDB


@pytest.fixture
def client(tmp_path):
    db = VectraDB(storage_dir=str(tmp_path))
    set_db(db)
    return TestClient(app)


def test_health_check(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["service"] == "vectra-py"
    assert data["version"] == "0.3.0"


def test_metrics_endpoint(client):
    res = client.get("/v1/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "vectra_collections_count" in data
    assert "vectra_total_vectors" in data


def test_collection_lifecycle(client):
    # 1. Create collection
    res = client.post(
        "/v1/collections",
        json={"name": "test_items", "dimension": 3, "metric": "cosine", "index_type": "flat"},
    )
    assert res.status_code == 200
    assert res.json()["status"] == "created"

    # Duplicate creation error
    dup = client.post(
        "/v1/collections",
        json={"name": "test_items", "dimension": 3},
    )
    assert dup.status_code == 400

    # 2. List collections
    res = client.get("/v1/collections")
    assert res.status_code == 200
    cols = res.json()["collections"]
    assert any(c["name"] == "test_items" for c in cols)

    # 3. Get collection stats
    res = client.get("/v1/collections/test_items")
    assert res.status_code == 200
    stats = res.json()
    assert stats["name"] == "test_items"
    assert stats["dimension"] == 3

    # 4. Upsert documents
    docs = [
        {"id": "doc1", "vector": [1.0, 0.0, 0.0], "text": "hello world", "metadata": {"tag": "A"}},
        {"id": "doc2", "vector": [0.0, 1.0, 0.0], "text": "foo bar", "metadata": {"tag": "B"}},
        {"id": "doc3", "vector": [0.99, 0.01, 0.0], "text": "hello again", "metadata": {"tag": "A"}},
    ]
    res = client.post("/v1/collections/test_items/upsert", json={"documents": docs})
    assert res.status_code == 200
    assert res.json()["upserted"] == 3

    # 5. Query collection
    res = client.post(
        "/v1/collections/test_items/query",
        json={"vector": [1.0, 0.0, 0.0], "top_k": 2, "filter": {"tag": "A"}},
    )
    assert res.status_code == 200
    results = res.json()["results"]
    assert len(results) == 2
    assert results[0]["id"] == "doc1"

    # 6. Hybrid query
    res = client.post(
        "/v1/collections/test_items/hybrid",
        json={"text": "hello", "top_k": 2},
    )
    assert res.status_code == 200
    hybrid_results = res.json()["results"]
    assert len(hybrid_results) >= 1
    assert any(r["id"] in ["doc1", "doc3"] for r in hybrid_results)

    # 7. Deduplicate
    res = client.post(
        "/v1/collections/test_items/deduplicate",
        params={"threshold": 0.95},
    )
    assert res.status_code == 200
    dups = res.json()["duplicates"]
    assert len(dups) >= 1
    pair = dups[0]
    assert {pair["id_a"], pair["id_b"]} == {"doc1", "doc3"}

    # 8. Delete collection
    res = client.delete("/v1/collections/test_items")
    assert res.status_code == 200
    assert res.json()["status"] == "deleted"

    # 404 when querying deleted collection
    res = client.get("/v1/collections/test_items")
    assert res.status_code == 404
