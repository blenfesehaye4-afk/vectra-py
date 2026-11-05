"""VectraDB database manager with multi-collection persistence."""

from __future__ import annotations
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

from .collection import Collection
from .embedding import BaseEmbedder, TfIdfEmbedder


class VectraDB:
    """
    Embedded vector database engine.
    Manages collections with persistent on-disk storage and atomic saves.
    """

    def __init__(self, storage_dir: str | Path | None = None):
        self.storage_dir = Path(storage_dir) if storage_dir else None
        self._collections: dict[str, Collection] = {}
        if self.storage_dir and self.storage_dir.exists():
            self.load()

    def create_collection(
        self,
        name: str,
        dimension: int = 128,
        metric: str = "cosine",
        embedder: BaseEmbedder | None = None,
    ) -> Collection:
        """Create a new collection. Raises ValueError if collection already exists."""
        if name in self._collections:
            raise ValueError(f"Collection '{name}' already exists")
        col = Collection(name=name, dimension=dimension, metric=metric, embedder=embedder)
        self._collections[name] = col
        if self.storage_dir:
            self.persist(name)
        return col

    def get_collection(self, name: str) -> Collection:
        """Retrieve an existing collection by name."""
        if name not in self._collections:
            raise KeyError(f"Collection '{name}' not found")
        return self._collections[name]

    def get_or_create_collection(
        self,
        name: str,
        dimension: int = 128,
        metric: str = "cosine",
        embedder: BaseEmbedder | None = None,
    ) -> Collection:
        """Retrieve existing collection or create if missing."""
        if name in self._collections:
            return self._collections[name]
        return self.create_collection(
            name=name, dimension=dimension, metric=metric, embedder=embedder
        )

    def list_collections(self) -> list[str]:
        """List all loaded collection names."""
        return sorted(list(self._collections.keys()))

    def delete_collection(self, name: str) -> bool:
        """Delete a collection and its disk file if storage_dir is configured."""
        if name not in self._collections:
            return False
        del self._collections[name]
        if self.storage_dir:
            file_path = self.storage_dir / f"{name}.vectra.json"
            if file_path.exists():
                file_path.unlink()
        return True

    def persist(self, collection_name: str | None = None) -> None:
        """Atomically persist collections to disk as JSON files."""
        if not self.storage_dir:
            return

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        targets = (
            [collection_name]
            if collection_name
            else list(self._collections.keys())
        )

        for name in targets:
            if name not in self._collections:
                continue
            col = self._collections[name]
            payload = col.to_dict()
            dest_file = self.storage_dir / f"{name}.vectra.json"

            # Atomic write via tempfile and replace
            with tempfile.NamedTemporaryFile(
                "w", dir=str(self.storage_dir), delete=False, encoding="utf-8"
            ) as tf:
                json.dump(payload, tf, indent=2)
                temp_name = tf.name

            os.replace(temp_name, dest_file)

    def load(self) -> None:
        """Load all collections from storage directory."""
        if not self.storage_dir or not self.storage_dir.exists():
            return

        for json_path in self.storage_dir.glob("*.vectra.json"):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                name = data["name"]
                col = Collection.from_dict(data)
                self._collections[name] = col
            except Exception as e:
                # Corrupt or invalid file skip
                continue
