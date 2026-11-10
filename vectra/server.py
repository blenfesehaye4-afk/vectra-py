"""High-performance FastAPI REST API server for VectraDB."""

from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .db import VectraDB

app = FastAPI(title="VectraDB REST API", version="0.3.0")

# In-memory global instance (or path-backed)
_DB = VectraDB()


def get_db() -> VectraDB:
    return _DB


def set_db(db: VectraDB) -> None:
    global _DB
    _DB = db


# -----------------------------------------------------------------------------
# Request & Response Schemas
# -----------------------------------------------------------------------------

class CreateCollectionRequest(BaseModel):
    name: str
    dimension: int
    metric: str = "cosine"
    index_type: str = "flat"


class DocumentPayload(BaseModel):
    id: Optional[str] = None
    vector: Optional[List[float]] = None
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class UpsertRequest(BaseModel):
    documents: List[DocumentPayload]


class QueryRequest(BaseModel):
    vector: Optional[List[float]] = None
    text: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=100)
    filter: Optional[Dict[str, Any]] = None
    score_threshold: Optional[float] = None
    include_vectors: bool = False


class HybridQueryRequest(BaseModel):
    text: str
    top_k: int = Field(default=5, ge=1, le=100)
    filter: Optional[Dict[str, Any]] = None
    dense_weight: float = 0.5
    sparse_weight: float = 0.5


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

def _fetch_collection(name: str):
    try:
        return _DB.get_collection(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")


@app.get("/health")
def health():
    return {"status": "healthy", "service": "vectra-py", "version": "0.3.0"}


@app.get("/v1/metrics")
def metrics():
    names = _DB.list_collections()
    total_vectors = sum(len(_DB.get_collection(n)) for n in names)
    return {
        "vectra_collections_count": len(names),
        "vectra_total_vectors": total_vectors,
    }


@app.post("/v1/collections")
def create_collection(req: CreateCollectionRequest):
    if req.name in _DB.list_collections():
        raise HTTPException(status_code=400, detail=f"Collection '{req.name}' already exists")
    col = _DB.create_collection(
        name=req.name,
        dimension=req.dimension,
        metric=req.metric,
        index_type=req.index_type,
    )
    return {"status": "created", "collection": col.stats()}


@app.get("/v1/collections")
def list_collections():
    names = _DB.list_collections()
    return {"collections": [_DB.get_collection(n).stats() for n in names]}


@app.get("/v1/collections/{name}")
def get_collection(name: str):
    col = _fetch_collection(name)
    return col.stats()


@app.delete("/v1/collections/{name}")
def delete_collection(name: str):
    if not _DB.delete_collection(name):
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    return {"status": "deleted", "name": name}


@app.post("/v1/collections/{name}/upsert")
def upsert_documents(name: str, req: UpsertRequest):
    col = _fetch_collection(name)
    docs_data = [d.model_dump() for d in req.documents]
    count = col.add_batch(docs_data)
    return {"status": "success", "upserted": count}


@app.post("/v1/collections/{name}/query")
def query_collection(name: str, req: QueryRequest):
    col = _fetch_collection(name)
    try:
        results = col.query(
            vector=req.vector,
            text=req.text,
            top_k=req.top_k,
            filter=req.filter,
            score_threshold=req.score_threshold,
            include_vectors=req.include_vectors,
        )
        return {
            "results": [r.to_dict(include_vector=req.include_vectors) for r in results]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/v1/collections/{name}/hybrid")
def hybrid_query(name: str, req: HybridQueryRequest):
    col = _fetch_collection(name)
    fused = col.hybrid_query(
        text=req.text,
        top_k=req.top_k,
        filter=req.filter,
        dense_weight=req.dense_weight,
        sparse_weight=req.sparse_weight,
    )
    return {"results": fused}


@app.post("/v1/collections/{name}/deduplicate")
def deduplicate_collection(name: str, threshold: float = Query(0.95, ge=0.0, le=1.0)):
    col = _fetch_collection(name)
    pairs = col.deduplicate(threshold=threshold)
    return {
        "duplicates": [
            {"id_a": a, "id_b": b, "similarity": score}
            for a, b, score in pairs
        ]
    }
