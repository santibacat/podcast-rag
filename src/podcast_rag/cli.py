from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from podcast_rag.config import build_settings
from podcast_rag.db import add_episode, init_db
from podcast_rag.search import get_episode_segments, list_episodes, list_topics, search_chunks
from podcast_rag.timecode import format_timestamp
from podcast_rag.transcripts import parse_transcript_file

app = typer.Typer(no_args_is_help=True)
console = Console()


def data_dir_option() -> Path:
    return typer.Option(Path("data"), "--data-dir", help="Directory for SQLite DB and local artifacts.")


@app.command("init-db")
def init_db_command(data_dir: Path = data_dir_option()) -> None:
    """Create the local SQLite database."""
    settings = build_settings(data_dir)
    init_db(settings.db_path)
    console.print(f"Initialized database at [bold]{settings.db_path}[/bold]")


@app.command("ingest-transcript")
def ingest_transcript_command(
    transcript_path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    title: str = typer.Option(..., "--title", "-t", help="Episode title."),
    source_url: str | None = typer.Option(None, "--source-url", help="Original media URL."),
    author: str | None = typer.Option(None, "--author", help="Podcast author or channel."),
    language: str | None = typer.Option(None, "--language", help="Transcript language code."),
    data_dir: Path = data_dir_option(),
) -> None:
    """Import a plain text or timestamped transcript."""
    settings = build_settings(data_dir)
    segments = parse_transcript_file(transcript_path)
    if not segments:
        raise typer.BadParameter("Transcript did not contain any text.")
    episode_id = add_episode(
        settings.db_path,
        title=title,
        segments=segments,
        source_url=source_url,
        author=author,
        language=language,
    )
    console.print(f"Imported episode [bold]{episode_id}[/bold] with {len(segments)} transcript segments.")


@app.command("episodes")
def episodes_command(data_dir: Path = data_dir_option()) -> None:
    """List ingested episodes."""
    settings = build_settings(data_dir)
    episodes = list_episodes(settings.db_path)
    table = Table("ID", "Title", "Segments", "Author", "Language", "Source")
    for episode in episodes:
        table.add_row(
            str(episode["id"]),
            str(episode["title"]),
            str(episode["segment_count"]),
            str(episode["author"] or ""),
            str(episode["language"] or ""),
            str(episode["source_url"] or ""),
        )
    console.print(table)


@app.command("search")
def search_command(
    query: str = typer.Argument(..., help="FTS query. Use quotes for phrases."),
    limit: int = typer.Option(10, "--limit", "-n", min=1, max=50),
    data_dir: Path = data_dir_option(),
) -> None:
    """Search transcript chunks."""
    settings = build_settings(data_dir)
    results = search_chunks(settings.db_path, query=query, limit=limit)
    if not results:
        console.print("No results.")
        return

    for result in results:
        timestamp = format_timestamp(result["start_seconds"])
        console.print(f"[bold]Episode {result['episode_id']} - {result['title']}[/bold] [{timestamp}]")
        if result["source_url"]:
            console.print(str(result["source_url"]))
        console.print(str(result["snippet"]))
        console.print()


@app.command("show")
def show_command(
    episode_id: int = typer.Argument(..., help="Episode ID."),
    data_dir: Path = data_dir_option(),
) -> None:
    """Show an episode transcript."""
    settings = build_settings(data_dir)
    try:
        episode, segments = get_episode_segments(settings.db_path, episode_id)
    except LookupError as exc:
        raise typer.BadParameter(str(exc)) from exc

    console.print(f"[bold]{episode['title']}[/bold]")
    if episode["source_url"]:
        console.print(str(episode["source_url"]))
    console.print()

    for segment in segments:
        console.print(f"[dim]{format_timestamp(segment['start_seconds'])}[/dim] {segment['text']}")


@app.command("topics")
def topics_command(
    limit: int = typer.Option(50, "--limit", "-n", min=1, max=200),
    data_dir: Path = data_dir_option(),
) -> None:
    """List candidate entities and topics."""
    settings = build_settings(data_dir)
    topics = list_topics(settings.db_path, limit=limit)
    table = Table("Topic", "Mentions", "Episodes")
    for topic in topics:
        table.add_row(str(topic["name"]), str(topic["mentions"]), str(topic["episodes"]))
    console.print(table)


if __name__ == "__main__":
    app()
