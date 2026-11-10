# Vectra ⚡

[![CI](https://github.com/blenfesehaye4-afk/vectra-py/actions/workflows/ci.yml/badge.svg)](https://github.com/blenfesehaye4-afk/vectra-py/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Lightweight, embeddable vector database & hybrid search engine for Python AI applications with zero heavy external dependencies.**

Vectra is engineered for developers needing fast, local vector storage, approximate nearest neighbor search (HNSW), hybrid dense-sparse retrieval (BM25 + RRF), and vector compression (PQ / SQ8) for LLM memory, RAG pipelines, or semantic deduplication without running heavy database clusters.

---

## 🌟 Key Features

- 🚀 **Zero Heavy Dependencies**: Pure Python algorithms with optional `fastapi` and `rich` extensions. No mandatory C++ compilers, CUDA toolkits, or external Daemons.
- ⚡ **HNSW Graph Indexing**: Hierarchical Navigable Small World graph for sub-millisecond approximate nearest neighbor (ANN) search with logarithmic $O(\log N)$ query complexity.
- 🔀 **Hybrid Search (BM25 + Dense RRF)**: Built-in Okapi BM25 lexical engine combined with dense semantic embeddings via Reciprocal Rank Fusion (RRF).
- 🗜️ **Vector Quantization & Compression**:
  - **Product Quantization (PQ)**: Subspace vector decomposition with k-means centroids and Asymmetric Distance Computation (ADC).
  - **Scalar Quantization (SQ8)**: Uniform float32 to signed int8 quantization reducing vector memory footprint by 75% with >0.98 cosine correlation.
- 💾 **Binary Serialization Engine**: High-performance binary file format with CRC32 payload checksum integrity validation.
- 🌐 **Production FastAPI REST API**: Async HTTP endpoints for collection management, batch upserts, vector queries, hybrid search, and Prometheus-compatible metrics.
- 📐 **Multiple Distance & Similarity Metrics**: Cosine Similarity, Euclidean Distance ($L_2$), Manhattan Distance ($L_1$), and Dot Product.
- 🔍 **Rich Metadata Filtering**: MongoDB-style query operators (`$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$contains`, `$and`, `$or`, `$not`).
- 🧹 **Semantic Deduplication**: Automated duplicate detection above customizable similarity thresholds.
- 💻 **Rich CLI Suite**: Interactive benchmarking, hybrid search, collection inspection, and server hosting.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             VectraDB Engine                                 │
│          (Multi-Collection Manager, Binary Storage & REST Server)           │
└───────────────────────────────────────┬─────────────────────────────────────┘
                                        │
           ┌────────────────────────────┴────────────────────────────┐
           ▼                                                         ▼
┌─────────────────────────────────────┐   ┌──────────────────────────────────┐
│      Collection (Dense Index)       │   │     BM25 Lexical Index (Sparse)  │
│  - Flat Exact Scan                  │   │  - Term Frequency (TF)           │
│  - HNSW Multi-Layer ANN Graph       │   │  - Inverted Document Index (IDF) │
│  - Quantizer (PQ / SQ8 int8)        │   │  - Sublinear Length Normalization│
└──────────────────┬──────────────────┘   └────────────────┬─────────────────┘
                   │                                       │
                   └───────────────────┬───────────────────┘
                                       │
                                       ▼
                   ┌───────────────────────────────────────┐
                   │    Reciprocal Rank Fusion (RRF)       │
                   │    Score = Σ (1 / (k + rank_i))       │
                   └───────────────────┬───────────────────┘
                                       │
                                       ▼
                   ┌───────────────────────────────────────┐
                   │          Filter & Result Engine       │
                   │    MongoDB-style Metadata Filtering   │
                   └───────────────────────────────────────┘
```

---

## 📦 Installation

```bash
pip install vectra-py
```

To include FastAPI REST server and CLI tools:

```bash
pip install "vectra-py[server,cli]"
```

Or install in development mode from source:

```bash
git clone https://github.com/blenfesehaye4-afk/vectra-py.git
cd vectra-py
pip install -e .[dev]
```

---

## 🚀 Quickstart

### 1. Approximate Nearest Neighbors with HNSW

```python
from vectra import VectraDB

# Initialize database
db = VectraDB(storage_dir="./data/vectra_store")

# Create collection configured with HNSW graph indexing
collection = db.get_or_create_collection(
    name="documents_hnsw",
    dimension=128,
    metric="cosine",
    index_type="hnsw"  # Enables HNSW graph index
)

# Insert documents
collection.add_batch([
    {"id": "doc-1", "vector": [0.1] * 128, "text": "FastAPI async REST framework", "metadata": {"tag": "web"}},
    {"id": "doc-2", "vector": [-0.2] * 128, "text": "PostgreSQL ACID database storage", "metadata": {"tag": "db"}},
    {"id": "doc-3", "vector": [0.11] * 128, "text": "Microservices in Docker containers", "metadata": {"tag": "devops"}},
])

# Query nearest neighbors via HNSW
results = collection.query(vector=[0.1] * 128, top_k=2)
for r in results:
    print(f"[{r.score:.4f}] {r.id}: {r.text}")
```

### 2. Hybrid Sparse-Dense Search (BM25 + RRF)

```python
# Hybrid query combines semantic embeddings and BM25 lexical keyword matching
hybrid_results = collection.hybrid_query(
    text="database storage",
    top_k=2,
    dense_weight=0.5,
    sparse_weight=0.5
)

for item in hybrid_results:
    print(f"Rank {item['rank']}: {item['id']} (Score: {item['score']:.4f}) -> {item['text']}")
```

### 3. Product & Scalar Quantization (PQ / SQ8)

```python
from vectra import ProductQuantizer, ScalarQuantizer

vectors = [[0.1, -0.4, 0.8, 0.2] for _ in range(100)]

# Scalar Quantization (32-bit float to 8-bit int)
sq = ScalarQuantizer()
sq.fit(vectors)
compressed_int8 = sq.quantize(vectors[0])
reconstructed = sq.dequantize(compressed_int8)

# Product Quantization
pq = ProductQuantizer(num_subvectors=2, num_centroids=16)
pq.fit(vectors)
codes = pq.encode(vectors[0])
score = pq.asymmetric_distance(query_vector=[0.1, -0.4, 0.8, 0.2], code=codes)
```

### 4. Binary Storage Serialization

```python
from vectra import BinaryStorage

# Persist collection in compact binary format with CRC32 integrity verification
BinaryStorage.save_collection(
    "collection.bin",
    dimension=128,
    metric="cosine",
    records=[{"id": "v1", "vector": [0.5] * 128, "metadata": {"label": "test"}}]
)

dim, metric, loaded = BinaryStorage.load_collection("collection.bin")
```

---

## 🌐 FastAPI REST API Server

Run the production REST API server:

```bash
vectra serve --host 0.0.0.0 --port 8000
```

### Key Endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check endpoint |
| `GET` | `/v1/metrics` | Prometheus-compatible vector & collection metrics |
| `POST` | `/v1/collections` | Create a new vector collection |
| `GET` | `/v1/collections` | List all available collections and stats |
| `POST` | `/v1/collections/{name}/upsert` | Batch upsert documents with vectors & metadata |
| `POST` | `/v1/collections/{name}/query` | Dense vector similarity search with filters |
| `POST` | `/v1/collections/{name}/hybrid` | Hybrid BM25 + dense search with RRF fusion |
| `POST` | `/v1/collections/{name}/deduplicate` | Semantic deduplication above threshold |

Interactive Swagger UI documentation is available at `http://localhost:8000/docs`.

---

## 🖥️ Command Line Interface (CLI)

```bash
# Benchmark HNSW ANN graph against flat exact scan
vectra benchmark-hnsw --vectors 500 --dim 64 --queries 20

# Run hybrid BM25 + dense search from terminal
vectra hybrid tech-kb --query "microservice orchestration" --top-k 5

# Start REST API server
vectra serve --port 8000

# Run interactive knowledge base demo
vectra demo
```

---

## 🧪 Running Tests

The test suite thoroughly covers all modules (HNSW graph indexing, BM25, PQ, SQ8, binary storage, REST API server, and CLI):

```bash
pytest -v
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
