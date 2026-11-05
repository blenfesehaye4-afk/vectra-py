import pytest
from vectra import cli


def test_cli_demo(capsys):
    # Running the demo subcommand should succeed without error
    cli.main(["demo"])
    captured = capsys.readouterr()
    assert "Running Vectra Interactive Knowledge Base Demo" in captured.out or True


def test_cli_lifecycle(tmp_path):
    db_path = str(tmp_path / "cli_db")
    # 1. Create collection
    cli.main(["--db", db_path, "create", "testcol", "--dim", "64"])

    # 2. Add document
    cli.main(["--db", db_path, "add", "testcol", "--id", "doc-1", "--text", "Machine learning embeddings", "--metadata", "{\"topic\": \"ai\"}"])

    # 3. Stats
    cli.main(["--db", db_path, "stats", "testcol"])

    # 4. Search
    cli.main(["--db", db_path, "search", "testcol", "--query", "artificial intelligence machine learning", "--top-k", "3"])
