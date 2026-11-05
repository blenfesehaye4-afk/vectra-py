# Vectra ⚡

[![CI](https://github.com/blenfesehaye4-afk/vectra-py/actions/workflows/ci.yml/badge.svg)](https://github.com/blenfesehaye4-afk/vectra-py/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Lightweight, embeddable vector database & semantic similarity search engine for Python AI applications with zero heavy external dependencies.**

Vectra is designed for developers who need fast, local, embeddable vector storage and nearest-neighbor search for LLM long-term memory, RAG (Retrieval-Augmented Generation), document search, or semantic deduplication without spinning up heavy external database clusters.

---

## 🌟 Key Features

- 🚀 **Zero Heavy Dependencies**: Runs directly on modern Python standard library with optional `rich` formatting. No C++ compilers, CUDA requirements, or heavy binary runtimes needed.
- 📐 **Multiple Distance & Similarity Metrics**:
  - **Cosine Similarity** (`cosine`): Ideal for text and normalized semantic embeddings.
  - **Euclidean Distance** (`euclidean` / `l2`): Geometric spatial distance.
  - **Manhattan Distance** (`manhattan` / `l1`): Grid-based absolute difference.
  - **Dot Product** (`dot_product`): Raw linear projection.
- 🔍 **Rich Metadata Filtering Engine**: Filter results with MongoDB-style operators (`$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$contains`, `$and`, `$or`, `$not`).
- 🧠 **Built-in Text Vectorizer**: Includes `TfIdfEmbedder` with sublinear term-frequency scaling and L2 normalization for instant out-of-the-box text vectorization without downloading gigabyte weights. Pluggable interface for HuggingFace, OpenAI, Cohere, or local ONNX models.
- 💾 **Atomic File-Backed Persistence**: In-memory speed with atomic disk snapshotting and automatic recovery across restarts.
- 🧹 **Semantic Deduplication**: Detect duplicate and near-duplicate documents exceeding customizable similarity thresholds.
- 💻 **Interactive CLI**: Rich terminal commands to inspect, query, deduplicate, and manage collections.

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────┐
│                      VectraDB                          │
│   (Multi-Collection Manager & Atomic Persistence)      │
└───────────────────────────┬────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
    ┌───────────────────┐       ┌───────────────────┐
    │ Collection: docs  │       │ Collection: memory│
    │  (dim=128, cos)   │       │  (dim=384, l2)    │
    └─────────┬─────────┘       └─────────┬─────────┘
              │                           │
    ┌─────────┴─────────┐       ┌─────────┴─────────┐
    │  Embedding Engine │       │  Metadata Filter  │
    │ (TfIdf / Custom)  │       │ ($gt, $in, $and)  │
    └───────────────────┘       └───────────────────┘
```

---

## 📦 Installation

```bash
pip install vectra-py
```

Or install in development mode from source:

```bash
git clone https://github.com/blenfesehaye4-afk/vectra-py.git
cd vectra-py
pip install -e .[dev]
```

---

## 🚀 Quickstart

### 1. Python API

```python
from vectra import VectraDB

# Initialize database with persistent local storage
db = VectraDB(storage_dir="./data/vectra_store")

# Create or load a vector collection
collection = db.get_or_create_collection(
    name="kb_articles",
    dimension=64,
    metric="cosine"
)

# Ingest documents with metadata
collection.add(
    id="doc-1",
    text="FastAPI is an asynchronous web framework for Python.",
    metadata={"category": "engineering", "views": 1500}
)
collection.add(
    id="doc-2",
    text="PostgreSQL provides robust ACID transactions and indexing.",
    metadata={"category": "database", "views": 2800}
)
collection.add(
    id="doc-3",
    text="Docker containers package application code and dependencies.",
    metadata={"category": "devops", "views": 950}
)

# Persist to disk
db.persist()

# Perform semantic similarity search with metadata filter
results = collection.query(
    text="relational database storage and sql",
    top_k=2,
    filter={"views": {"$gte": 1000}}
)

for r in results:
    print(f"[{r.score:.4f}] {r.id}: {r.text} | meta={r.metadata}")
```

### 2. Semantic Deduplication

```python
# Detect duplicate or paraphrased records with similarity >= 0.90
duplicates = collection.deduplicate(threshold=0.90)
for doc_a, doc_b, score in duplicates:
    print(f"Match: {doc_a} <-> {doc_b} (Score: {score:.4f})")
```

---

## 🖥️ Command Line Interface (CLI)

Vectra includes a complete command-line suite:

### Run Interactive Demo
```bash
vectra demo
```

### Manage Collections
```bash
# Create a collection
vectra create technical-docs --dim 128 --metric cosine

# List all collections
vectra list

# Inspect collection statistics
vectra stats technical-docs
```

### Insert & Query
```bash
# Add a document
vectra add technical-docs \
  --id "doc-101" \
  --text "Transformers utilize self-attention mechanisms for language understanding." \
  --metadata '{"author": "Blen", "year": 2026}'

# Search with metadata filtering
vectra search technical-docs \
  --query "attention mechanisms in deep learning" \
  --top-k 5 \
  --filter '{"year": {"$gte": 2025}}'
```

---

## 🧪 Running Tests

The test suite covers metrics, metadata filtering, text vectorization, collection persistence, and CLI commands:

```bash
pytest -v
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
