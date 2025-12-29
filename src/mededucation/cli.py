"""MedEducation CLI - Medical Education RAG Assistant.

Command-line interface for ingesting medical textbooks and querying them
with natural language questions.

Usage:
    mededucation ingest <pdf_path> --source-id <id> --source-name <name>
    mededucation index --source-id <id>
    mededucation sources
    mededucation query "What are the signs of shock?"
    mededucation chat
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(
    name="mededucation",
    help="Medical Education RAG Assistant - Query medical textbooks with citations",
    add_completion=False,
)
console = Console()

# Default paths
DEFAULT_DATA_DIR = Path("./data")
DEFAULT_VECTORDB_PATH = DEFAULT_DATA_DIR / "vectordb"
DEFAULT_TEXTBOOKS_PATH = DEFAULT_DATA_DIR / "textbooks"
DEFAULT_CONFIG_PATH = Path("./config/sources.yaml")


def get_project_root() -> Path:
    """Find the project root (where pyproject.toml or config dir exists)."""
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "pyproject.toml").exists() or (parent / "config").exists():
            return parent
    return cwd


@app.command()
def ingest(
    file_path: str = typer.Argument(..., help="Path to PDF or EPUB file"),
    source_id: str = typer.Option(..., "--source-id", "-s", help="Unique identifier for this source"),
    source_name: str = typer.Option(None, "--source-name", "-n", help="Short name for citations (defaults to source_id)"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory for extracted JSON"),
):
    """Ingest a PDF or EPUB file and extract text with page tracking."""
    from mededucation.ingest import PDFProcessor, EPUBProcessor

    file_path = Path(file_path)
    if not file_path.exists():
        console.print(f"[red]Error:[/red] File not found: {file_path}")
        raise typer.Exit(1)

    source_name = source_name or source_id

    # Determine output path
    if output_dir:
        out_path = Path(output_dir)
    else:
        project_root = get_project_root()
        out_path = project_root / "data" / "extracted"

    out_path.mkdir(parents=True, exist_ok=True)
    json_output = out_path / f"{source_id}_extracted.json"

    console.print(f"\n[bold]Ingesting:[/bold] {file_path.name}")
    console.print(f"[bold]Source ID:[/bold] {source_id}")
    console.print(f"[bold]Source Name:[/bold] {source_name}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting text...", total=None)

        try:
            if file_path.suffix.lower() == ".pdf":
                processor = PDFProcessor(file_path)
            elif file_path.suffix.lower() == ".epub":
                processor = EPUBProcessor(file_path)
            else:
                console.print(f"[red]Error:[/red] Unsupported file type: {file_path.suffix}")
                raise typer.Exit(1)

            metadata = processor.extract()
            processor.save_json(json_output)

            progress.update(task, description="Complete!")

        except Exception as e:
            console.print(f"[red]Error during extraction:[/red] {e}")
            raise typer.Exit(1)

    # Print summary
    console.print(f"\n[green]Extraction complete![/green]")
    console.print(f"  Pages: {metadata.total_pages:,}")
    console.print(f"  Words: {metadata.total_words:,}")
    console.print(f"  Chapters: {len(metadata.chapters)}")
    console.print(f"  Output: {json_output}")

    console.print(f"\n[yellow]Next step:[/yellow] Run 'mededucation index --source-id {source_id}' to build the vector index")


@app.command()
def index(
    source_id: str = typer.Option(..., "--source-id", "-s", help="Source ID to index"),
    source_name: Optional[str] = typer.Option(None, "--source-name", "-n", help="Short name for citations"),
    extracted_json: Optional[str] = typer.Option(None, "--input", "-i", help="Path to extracted JSON (auto-detected if not provided)"),
    vectordb_path: Optional[str] = typer.Option(None, "--vectordb", help="Path to vector database"),
    rebuild: bool = typer.Option(False, "--rebuild", "-r", help="Rebuild index (delete existing chunks)"),
):
    """Build vector index from extracted text for semantic search."""
    import json
    from mededucation.models.content import ExtractedPage
    from mededucation.chunking import HighQualityChunker
    from mededucation.storage import VectorStore

    project_root = get_project_root()
    source_name = source_name or source_id

    # Find extracted JSON
    if extracted_json:
        json_path = Path(extracted_json)
    else:
        json_path = project_root / "data" / "extracted" / f"{source_id}_extracted.json"

    if not json_path.exists():
        console.print(f"[red]Error:[/red] Extracted JSON not found: {json_path}")
        console.print(f"[yellow]Hint:[/yellow] Run 'mededucation ingest' first")
        raise typer.Exit(1)

    # Vector DB path
    if vectordb_path:
        vdb_path = Path(vectordb_path)
    else:
        vdb_path = project_root / "data" / "vectordb"

    console.print(f"\n[bold]Indexing:[/bold] {source_id}")
    console.print(f"[bold]Source:[/bold] {json_path}")
    console.print(f"[bold]Vector DB:[/bold] {vdb_path}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Load extracted data
        task = progress.add_task("Loading extracted text...", total=None)
        with open(json_path, "r") as f:
            data = json.load(f)
        pages = [ExtractedPage(**p) for p in data["pages"]]
        progress.update(task, description=f"Loaded {len(pages)} pages")

        # Chunk
        task = progress.add_task("Chunking text...", total=None)
        chunker = HighQualityChunker()
        chunks = chunker.chunk_pages(pages, source_name=source_name)
        progress.update(task, description=f"Created {len(chunks)} chunks")

        # Store
        task = progress.add_task("Building vector index...", total=None)
        store = VectorStore(persist_directory=str(vdb_path))

        if rebuild:
            deleted = store.delete_source(source_id)
            if deleted > 0:
                console.print(f"  Deleted {deleted} existing chunks")

        added = store.add_chunks(chunks, source_id=source_id)
        progress.update(task, description=f"Indexed {added} chunks")

    console.print(f"\n[green]Indexing complete![/green]")
    console.print(f"  Chunks indexed: {added}")
    console.print(f"  Total chunks in DB: {store.get_chunk_count()}")

    console.print(f"\n[yellow]Ready![/yellow] You can now query with 'mededucation query \"your question\"'")


@app.command()
def sources(
    vectordb_path: Optional[str] = typer.Option(None, "--vectordb", help="Path to vector database"),
):
    """List all indexed sources."""
    from mededucation.storage import VectorStore

    project_root = get_project_root()
    vdb_path = Path(vectordb_path) if vectordb_path else project_root / "data" / "vectordb"

    if not vdb_path.exists():
        console.print("[yellow]No vector database found.[/yellow]")
        console.print("Run 'mededucation ingest' and 'mededucation index' first.")
        return

    store = VectorStore(persist_directory=str(vdb_path))
    source_list = store.get_sources()

    if not source_list:
        console.print("[yellow]No sources indexed yet.[/yellow]")
        return

    table = Table(title="Indexed Sources")
    table.add_column("Source ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Chunks", justify="right")
    table.add_column("Chapters", justify="right")

    for src in source_list:
        chapters = src.get("chapters", [])
        chapter_str = f"{len(chapters)}" if chapters else "-"
        table.add_row(
            src["source_id"],
            src["source_name"],
            str(src["chunk_count"]),
            chapter_str,
        )

    console.print(table)
    console.print(f"\nTotal chunks: {store.get_chunk_count()}")


@app.command()
def profiles():
    """List available user profiles."""
    from mededucation.prompts.system import PROFILES

    table = Table(title="Available Profiles")
    table.add_column("Profile ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Description")

    for key, info in PROFILES.items():
        table.add_row(key, info["name"], info["description"])

    console.print(table)
    console.print("\n[dim]Set profile in config/sources.yaml or use --profile flag[/dim]")


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    source_id: Optional[str] = typer.Option(None, "--source", "-s", help="Filter to specific source"),
    top_k: int = typer.Option(12, "--top-k", "-k", help="Number of chunks to retrieve"),
    profile: Optional[str] = typer.Option(None, "--profile", "-p", help="User profile (flight_critical_care, medical_student, etc.)"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config file"),
    vectordb_path: Optional[str] = typer.Option(None, "--vectordb", help="Path to vector database"),
    show_sources: bool = typer.Option(True, "--sources/--no-sources", help="Show source citations"),
):
    """Query the medical textbooks with a question."""
    from mededucation.chat import ChatEngine

    project_root = get_project_root()

    # Resolve paths
    if config_path:
        cfg_path = Path(config_path)
    else:
        cfg_path = project_root / "config" / "sources.yaml"
        if not cfg_path.exists():
            cfg_path = None

    if vectordb_path:
        vdb_path = Path(vectordb_path)
    else:
        vdb_path = project_root / "data" / "vectordb"

    if not vdb_path.exists():
        console.print("[red]Error:[/red] No vector database found.")
        console.print("Run 'mededucation ingest' and 'mededucation index' first.")
        raise typer.Exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Searching textbooks...", total=None)

        engine = ChatEngine(
            vectordb_path=str(vdb_path),
            config_path=str(cfg_path) if cfg_path else None,
            top_k=top_k,
            profile=profile or "flight_critical_care",
        )

        progress.update(task, description="Generating answer...")
        response = engine.query(question=question, source_id=source_id)

    # Display answer
    console.print()
    console.print(Panel(Markdown(response.answer), title="Answer", border_style="green"))

    if show_sources and response.chunks_used:
        console.print()
        console.print(f"[dim]{response.sources_formatted}[/dim]")


@app.command()
def chat(
    source_id: Optional[str] = typer.Option(None, "--source", "-s", help="Filter to specific source"),
    top_k: int = typer.Option(12, "--top-k", "-k", help="Number of chunks to retrieve"),
    profile: Optional[str] = typer.Option(None, "--profile", "-p", help="User profile (flight_critical_care, medical_student, etc.)"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config file"),
    vectordb_path: Optional[str] = typer.Option(None, "--vectordb", help="Path to vector database"),
):
    """Start an interactive chat session with the medical textbooks."""
    from mededucation.chat import ChatEngine

    project_root = get_project_root()

    # Resolve paths
    if config_path:
        cfg_path = Path(config_path)
    else:
        cfg_path = project_root / "config" / "sources.yaml"
        if not cfg_path.exists():
            cfg_path = None

    if vectordb_path:
        vdb_path = Path(vectordb_path)
    else:
        vdb_path = project_root / "data" / "vectordb"

    if not vdb_path.exists():
        console.print("[red]Error:[/red] No vector database found.")
        console.print("Run 'mededucation ingest' and 'mededucation index' first.")
        raise typer.Exit(1)

    active_profile = profile or "flight_critical_care"

    engine = ChatEngine(
        vectordb_path=str(vdb_path),
        config_path=str(cfg_path) if cfg_path else None,
        top_k=top_k,
        profile=active_profile,
    )

    # Get profile name for display
    from mededucation.prompts.system import PROFILES
    profile_name = PROFILES.get(active_profile, {}).get("name", active_profile)

    # Show welcome message
    console.print()
    console.print(Panel(
        f"[bold]MedEducation Chat[/bold]\n"
        f"[dim]Profile: {profile_name}[/dim]\n\n"
        "Ask questions about your medical textbooks.\n"
        "Type 'exit' or 'quit' to end the session.\n"
        "Type 'sources' to see indexed sources.\n"
        "Type 'profile' to see current profile.",
        border_style="blue",
    ))

    if source_id:
        console.print(f"[dim]Filtering to source: {source_id}[/dim]")

    while True:
        console.print()
        try:
            question = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        if question.lower() == "sources":
            source_list = engine.get_sources()
            if source_list:
                for src in source_list:
                    console.print(f"  • {src['source_id']}: {src['chunk_count']} chunks")
            else:
                console.print("[yellow]No sources indexed.[/yellow]")
            continue

        if question.lower() == "profile":
            console.print(f"[bold]Current Profile:[/bold] {profile_name}")
            console.print(f"[dim]Profile ID: {active_profile}[/dim]")
            console.print("\n[dim]Features: Detailed explanations, differentials, drug dosages, clinical pearls, mnemonics[/dim]")
            continue

        # Query
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Thinking...", total=None)
            response = engine.query(question=question, source_id=source_id)

        console.print()
        console.print("[bold green]Assistant:[/bold green]")
        console.print(Markdown(response.answer))

        if response.chunks_used:
            console.print()
            console.print(f"[dim]{response.sources_formatted}[/dim]")


@app.command()
def test(
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """Test the LLM connection."""
    from mededucation.chat import ChatEngine

    project_root = get_project_root()

    if config_path:
        cfg_path = Path(config_path)
    else:
        cfg_path = project_root / "config" / "sources.yaml"
        if not cfg_path.exists():
            cfg_path = None

    console.print("[bold]Testing LLM connection...[/bold]")

    engine = ChatEngine(
        vectordb_path=str(project_root / "data" / "vectordb"),
        config_path=str(cfg_path) if cfg_path else None,
    )

    result = engine.test_connection()

    if result["status"] == "connected":
        console.print(f"[green]Connected![/green]")
        console.print(f"  Base URL: {result['base_url']}")
        console.print(f"  Model: {result['configured_model']}")
        if result.get("available_models"):
            console.print(f"  Available models: {', '.join(result['available_models'][:5])}")
    else:
        console.print(f"[red]Connection failed![/red]")
        console.print(f"  Base URL: {result['base_url']}")
        console.print(f"  Error: {result.get('error', 'Unknown error')}")
        console.print("\n[yellow]Make sure Ollama is running:[/yellow]")
        console.print("  ollama serve")
        console.print("  ollama pull qwen3:14b")


if __name__ == "__main__":
    app()
