"""Rich command-line interface for Vectra vector database."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

from .db import VectraDB
from .metrics import cosine_similarity


console = Console()
DEFAULT_DB_PATH = Path(".vectra_store")


def get_db(path_str: str | None = None) -> VectraDB:
    path = Path(path_str) if path_str else DEFAULT_DB_PATH
    return VectraDB(storage_dir=path)


def cmd_create(args: argparse.Namespace) -> None:
    db = get_db(args.db)
    try:
        col = db.create_collection(name=args.name, dimension=args.dim, metric=args.metric)
        console.print(
            f"[bold green]✓ Created collection[/bold green] '[cyan]{col.name}[/cyan]' "
            f"(dim={col.dimension}, metric={col.metric})"
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


def cmd_list(args: argparse.Namespace) -> None:
    db = get_db(args.db)
    cols = db.list_collections()
    if not cols:
        console.print("[yellow]No collections found.[/yellow] Use 'vectra create <name>' to create one.")
        return

    table = Table(title="Vectra Collections", show_header=True, header_style="bold magenta")
    table.add_column("Collection Name", style="cyan")
    table.add_column("Documents", justify="right")
    table.add_column("Dimension", justify="right")
    table.add_column("Metric", style="green")

    for name in cols:
        col = db.get_collection(name)
        stats = col.stats()
        table.add_row(name, str(stats["count"]), str(stats["dimension"]), stats["metric"])

    console.print(table)


def cmd_add(args: argparse.Namespace) -> None:
    db = get_db(args.db)
    try:
        col = db.get_collection(args.collection)
    except KeyError:
        console.print(f"[bold red]Error:[/bold red] Collection '{args.collection}' not found.")
        sys.exit(1)

    metadata = {}
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError as e:
            console.print(f"[bold red]Invalid JSON metadata:[/bold red] {e}")
            sys.exit(1)

    doc = col.add(id=args.id, text=args.text, metadata=metadata)
    db.persist(col.name)
    console.print(
        f"[bold green]✓ Inserted document[/bold green] [cyan]{doc.id}[/cyan] into '[yellow]{col.name}[/yellow]'."
    )


def cmd_search(args: argparse.Namespace) -> None:
    db = get_db(args.db)
    try:
        col = db.get_collection(args.collection)
    except KeyError:
        console.print(f"[bold red]Error:[/bold red] Collection '{args.collection}' not found.")
        sys.exit(1)

    filter_dict = None
    if args.filter:
        try:
            filter_dict = json.loads(args.filter)
        except json.JSONDecodeError as e:
            console.print(f"[bold red]Invalid JSON filter:[/bold red] {e}")
            sys.exit(1)

    results = col.query(
        text=args.query,
        top_k=args.top_k,
        filter=filter_dict,
        score_threshold=args.min_score,
    )

    if not results:
        console.print(f"[yellow]No matches found for query:[/yellow] '{args.query}'")
        return

    table = Table(
        title=f"Search Results: '{args.query}' (Collection: {col.name})",
        show_header=True,
        header_style="bold blue",
    )
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("ID", style="cyan")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Distance", justify="right", style="dim")
    table.add_column("Text Preview", style="white")
    table.add_column("Metadata", style="yellow")

    for i, res in enumerate(results, 1):
        preview = (res.text[:60] + "...") if res.text and len(res.text) > 60 else (res.text or "-")
        meta_str = json.dumps(res.metadata, ensure_ascii=False)
        table.add_row(
            str(i),
            res.id,
            f"{res.score:.4f}",
            f"{res.distance:.4f}",
            preview,
            meta_str,
        )

    console.print(table)


def cmd_stats(args: argparse.Namespace) -> None:
    db = get_db(args.db)
    try:
        col = db.get_collection(args.collection)
    except KeyError:
        console.print(f"[bold red]Error:[/bold red] Collection '{args.collection}' not found.")
        sys.exit(1)

    s = col.stats()
    panel = Panel(
        f"[bold cyan]Collection:[/bold cyan] {s['name']}\n"
        f"[bold]Total Documents:[/bold] {s['count']}\n"
        f"[bold]Vector Dimension:[/bold] {s['dimension']}\n"
        f"[bold]Distance Metric:[/bold] {s['metric']}\n"
        f"[bold]Approx Memory:[/bold] {s['approx_memory_bytes']} bytes\n"
        f"[bold]Metadata Fields:[/bold] {', '.join(s['metadata_fields']) if s['metadata_fields'] else 'None'}",
        title="Vectra Collection Statistics",
        border_style="green",
    )
    console.print(panel)


def cmd_dedupe(args: argparse.Namespace) -> None:
    db = get_db(args.db)
    try:
        col = db.get_collection(args.collection)
    except KeyError:
        console.print(f"[bold red]Error:[/bold red] Collection '{args.collection}' not found.")
        sys.exit(1)

    dupes = col.deduplicate(threshold=args.threshold)
    if not dupes:
        console.print(f"[green]✓ No duplicate documents found at threshold {args.threshold}[/green]")
        return

    table = Table(
        title=f"Potential Duplicates (Similarity >= {args.threshold})",
        show_header=True,
        header_style="bold red",
    )
    table.add_column("Doc A ID", style="cyan")
    table.add_column("Doc B ID", style="cyan")
    table.add_column("Similarity Score", justify="right", style="bold yellow")

    for id_a, id_b, score in dupes:
        table.add_row(id_a, id_b, f"{score:.4f}")

    console.print(table)


def cmd_demo(args: argparse.Namespace) -> None:
    console.print("[bold cyan]🚀 Running Vectra Interactive Knowledge Base Demo...[/bold cyan]\n")
    db = VectraDB(storage_dir=None)  # In-memory
    col = db.create_collection("tech-kb", dimension=64, metric="cosine")

    sample_articles = [
        ("art-1", "FastAPI is a modern, fast web framework for building APIs with Python 3.8+ based on standard Python type hints.", {"tag": "python", "category": "web"}),
        ("art-2", "PostgreSQL is a powerful, open-source object-relational database system known for reliability and feature robustness.", {"tag": "database", "category": "infra"}),
        ("art-3", "Docker containers package software into standardized units for development, shipment, and deployment.", {"tag": "devops", "category": "infra"}),
        ("art-4", "Vector embeddings represent words, sentences, or images as continuous high-dimensional coordinate vectors.", {"tag": "ai", "category": "ml"}),
        ("art-5", "Large Language Models use transformer architectures and self-attention mechanisms to generate human-like text.", {"tag": "ai", "category": "ml"}),
        ("art-6", "Kubernetes is an open-source container orchestration system for automating deployment, scaling, and management.", {"tag": "devops", "category": "infra"}),
    ]

    for doc_id, text, meta in sample_articles:
        col.add(id=doc_id, text=text, metadata=meta)

    console.print(f"[green]✓ Seeded {len(sample_articles)} technical knowledge articles into collection 'tech-kb'[/green]\n")

    test_queries = [
        ("neural network transformer models", {"category": "ml"}),
        ("deploying microservices in containers", None),
    ]

    for q, flt in test_queries:
        console.print(f"[bold yellow]🔍 Query:[/bold yellow] '[bold white]{q}[/bold white]' (Filter: {flt})")
        results = col.query(text=q, top_k=2, filter=flt)
        for rank, r in enumerate(results, 1):
            console.print(f"  {rank}. [cyan]{r.id}[/cyan] (Score: [green]{r.score:.4f}[/green]) -> {r.text}")
        console.print()

    console.print("[bold green]✓ Demo completed successfully![/bold green]")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="vectra",
        description="Lightweight embeddable vector database & semantic similarity search engine.",
    )
    parser.add_argument("--db", default=None, help="Directory path to database store (default: .vectra_store)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # create
    p_create = subparsers.add_parser("create", help="Create a new vector collection")
    p_create.add_argument("name", help="Collection name")
    p_create.add_argument("--dim", type=int, default=128, help="Vector dimension (default: 128)")
    p_create.add_argument("--metric", choices=["cosine", "euclidean", "manhattan", "dot_product"], default="cosine")
    p_create.set_defaults(func=cmd_create)

    # list
    p_list = subparsers.add_parser("list", help="List all collections")
    p_list.set_defaults(func=cmd_list)

    # add
    p_add = subparsers.add_parser("add", help="Add a document to a collection")
    p_add.add_argument("collection", help="Target collection name")
    p_add.add_argument("--id", default=None, help="Document ID (auto-generated if omitted)")
    p_add.add_argument("--text", required=True, help="Raw text to embed")
    p_add.add_argument("--metadata", default=None, help="JSON metadata string, e.g. '{\"category\": \"ai\"}'")
    p_add.set_defaults(func=cmd_add)

    # search
    p_search = subparsers.add_parser("search", help="Semantic similarity search")
    p_search.add_argument("collection", help="Collection to search")
    p_search.add_argument("--query", required=True, help="Search query string")
    p_search.add_argument("--top-k", type=int, default=5, help="Number of results to return (default: 5)")
    p_search.add_argument("--filter", default=None, help="JSON filter dictionary")
    p_search.add_argument("--min-score", type=float, default=None, help="Minimum score threshold")
    p_search.set_defaults(func=cmd_search)

    # stats
    p_stats = subparsers.add_parser("stats", help="Show collection statistics")
    p_stats.add_argument("collection", help="Collection name")
    p_stats.set_defaults(func=cmd_stats)

    # dedupe
    p_dedupe = subparsers.add_parser("dedupe", help="Find duplicate items above similarity threshold")
    p_dedupe.add_argument("collection", help="Collection name")
    p_dedupe.add_argument("--threshold", type=float, default=0.90, help="Similarity threshold (default: 0.90)")
    p_dedupe.set_defaults(func=cmd_dedupe)

    # demo
    p_demo = subparsers.add_parser("demo", help="Run interactive demonstration")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
