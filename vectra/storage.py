"""Binary storage serialization and CRC32 integrity validation for Vectra."""

import json
import os
import struct
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Binary File Layout:
# -------------------------------------------------------------
# [Header: 24 Bytes]
#   magic: 4B (0x56454354 -> "VECT")
#   version: 2B (e.g. 1)
#   metric_code: 2B (0=cosine, 1=euclidean, 2=manhattan, 3=dot)
#   dimension: 4B
#   count: 8B
#   payload_crc32: 4B
# [Body: Packed Records]
#   For each vector:
#     [id_len: 2B] [id_bytes]
#     [meta_len: 4B] [meta_json_utf8]
#     [dim * 4B float32]
HEADER_STRUCT = struct.Struct(">IHHIQI")
MAGIC = 0x56454354
VERSION = 1

METRIC_MAP = {
    "cosine": 0,
    "euclidean": 1,
    "l2": 1,
    "manhattan": 2,
    "l1": 2,
    "dot_product": 3,
    "dot": 3,
}
REVERSE_METRIC_MAP = {0: "cosine", 1: "euclidean", 2: "manhattan", 3: "dot_product"}


class BinaryStorage:
    """Manages high-throughput binary serialization and crash-safe atomic saves."""

    @classmethod
    def save_collection(
        cls,
        filepath: Union[str, Path],
        dimension: int,
        metric: str,
        records: List[Dict[str, Any]],
    ) -> int:
        """
        Serialize collection to binary file atomically with CRC32 verification.
        Returns total bytes written.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")

        metric_code = METRIC_MAP.get(metric.lower(), 0)
        body_chunks = []

        for r in records:
            id_bytes = str(r["id"]).encode("utf-8")
            meta_bytes = json.dumps(r.get("metadata", {})).encode("utf-8")
            vec = r["vector"]

            if len(vec) != dimension:
                raise ValueError(f"Vector dimension {len(vec)} does not match collection dimension {dimension}")

            rec_header = struct.pack(">HI", len(id_bytes), len(meta_bytes))
            vec_bytes = struct.pack(f">{dimension}f", *vec)

            body_chunks.append(rec_header)
            body_chunks.append(id_bytes)
            body_chunks.append(meta_bytes)
            body_chunks.append(vec_bytes)

        body_payload = b"".join(body_chunks)
        payload_crc = zlib.crc32(body_payload) & 0xFFFFFFFF

        header = HEADER_STRUCT.pack(
            MAGIC,
            VERSION,
            metric_code,
            dimension,
            len(records),
            payload_crc,
        )

        with open(tmp_path, "wb") as f:
            f.write(header)
            f.write(body_payload)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, path)
        return len(header) + len(body_payload)

    @classmethod
    def load_collection(
        cls,
        filepath: Union[str, Path],
    ) -> Tuple[int, str, List[Dict[str, Any]]]:
        """
        Load collection from binary file, verifying header magic and payload CRC32.
        Returns (dimension, metric, records).
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Binary collection file '{path}' not found")

        with open(path, "rb") as f:
            header_bytes = f.read(HEADER_STRUCT.size)
            if len(header_bytes) < HEADER_STRUCT.size:
                raise ValueError("Corrupt file: Header truncated")

            magic, version, metric_code, dimension, count, expected_crc = HEADER_STRUCT.unpack(header_bytes)
            if magic != MAGIC:
                raise ValueError(f"Invalid magic number {hex(magic)}, expected {hex(MAGIC)}")

            body_payload = f.read()

        actual_crc = zlib.crc32(body_payload) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ValueError(f"CRC32 Checksum Mismatch! File is corrupted (expected {expected_crc}, got {actual_crc})")

        records: List[Dict[str, Any]] = []
        offset = 0
        total_len = len(body_payload)

        for _ in range(count):
            if offset + 6 > total_len:
                break
            id_len, meta_len = struct.unpack_from(">HI", body_payload, offset)
            offset += 6

            id_str = body_payload[offset:offset + id_len].decode("utf-8")
            offset += id_len

            meta_str = body_payload[offset:offset + meta_len].decode("utf-8")
            meta = json.loads(meta_str) if meta_str else {}
            offset += meta_len

            vec_size = dimension * 4
            vec = list(struct.unpack_from(f">{dimension}f", body_payload, offset))
            offset += vec_size

            records.append({
                "id": id_str,
                "vector": vec,
                "metadata": meta,
            })

        metric_name = REVERSE_METRIC_MAP.get(metric_code, "cosine")
        return dimension, metric_name, records
