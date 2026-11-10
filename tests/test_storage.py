import pytest
from pathlib import Path
from vectra.storage import BinaryStorage


def test_binary_storage_roundtrip(tmp_path: Path):
    file_path = tmp_path / "test_coll.bin"
    records = [
        {"id": "doc_1", "vector": [0.1, 0.2, 0.3], "metadata": {"author": "Blen", "views": 150}},
        {"id": "doc_2", "vector": [0.9, -0.4, 0.0], "metadata": {"category": "ai"}},
    ]

    bytes_written = BinaryStorage.save_collection(file_path, dimension=3, metric="cosine", records=records)
    assert bytes_written > 0
    assert file_path.exists()

    dim, metric, loaded = BinaryStorage.load_collection(file_path)
    assert dim == 3
    assert metric == "cosine"
    assert len(loaded) == 2

    assert loaded[0]["id"] == "doc_1"
    assert loaded[0]["metadata"]["author"] == "Blen"
    assert pytest.approx(loaded[0]["vector"]) == [0.1, 0.2, 0.3]


def test_binary_storage_crc32_corruption_detection(tmp_path: Path):
    file_path = tmp_path / "corrupt.bin"
    records = [{"id": "v1", "vector": [1.0, 2.0], "metadata": {}}]

    BinaryStorage.save_collection(file_path, dimension=2, metric="euclidean", records=records)

    # Intentionally corrupt a byte in the body payload
    with open(file_path, "r+b") as f:
        f.seek(-1, 2)  # Go to last byte
        last_b = f.read(1)
        f.seek(-1, 2)
        f.write(bytes([last_b[0] ^ 0xFF]))  # Flip bits

    with pytest.raises(ValueError, match="CRC32 Checksum Mismatch"):
        BinaryStorage.load_collection(file_path)


def test_binary_storage_dimension_mismatch(tmp_path: Path):
    file_path = tmp_path / "bad_dim.bin"
    records = [{"id": "v1", "vector": [1.0, 2.0], "metadata": {}}]
    with pytest.raises(ValueError, match="does not match collection dimension"):
        BinaryStorage.save_collection(file_path, dimension=3, metric="cosine", records=records)
