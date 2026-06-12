from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from podcast_rag.config import build_settings
from podcast_rag.db import add_episode, init_db
from podcast_rag.discovery import PlaylistMode, PlaylistOrder, discover_sources
from podcast_rag.ingest import ingest_url
from podcast_rag.search import get_episode_segments, list_episodes, list_topics, search_chunks
from podcast_rag.timecode import format_timestamp
from podcast_rag.transcription import transcribe_audio
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


@app.command("ingest-url")
def ingest_url_command(
    url: str = typer.Argument(..., help="Media URL, YouTube video, playlist, or page containing media links."),
    playlist_mode: PlaylistMode = typer.Option(
        PlaylistMode.single,
        "--playlist-mode",
        help="Use 'single' for one item or 'all' to expand playlists/pages.",
    ),
    playlist_order: PlaylistOrder = typer.Option(
        PlaylistOrder.source,
        "--playlist-order",
        help="For expanded playlists, keep source order or prefer newest metadata.",
    ),
    max_items: int | None = typer.Option(
        None,
        "--max-items",
        min=1,
        help="Maximum playlist/page items to ingest. Source ordering is preserved.",
    ),
    whisper_model: str = typer.Option("small", "--whisper-model", help="faster-whisper model size."),
    device: str = typer.Option("auto", "--device", help="Whisper device: auto, cpu, cuda, or mps."),
    compute_type: str = typer.Option("auto", "--compute-type", help="Whisper compute type."),
    language: str | None = typer.Option(None, "--language", help="Optional language code, e.g. es."),
    no_skip_existing: bool = typer.Option(False, "--no-skip-existing", help="Reprocess URLs already in the DB."),
    data_dir: Path = data_dir_option(),
) -> None:
    """Download URL audio, transcribe it locally, and import the transcript."""
    settings = build_settings(data_dir)
    results = ingest_url(
        url=url,
        settings=settings,
        playlist_mode=playlist_mode,
        playlist_order=playlist_order,
        max_items=max_items,
        whisper_model=whisper_model,
        device=device,
        compute_type=compute_type,
        language=language,
        skip_existing=not no_skip_existing,
    )
    table = Table("Status", "Episode", "Title", "Source", "Message")
    for result in results:
        table.add_row(
            result.status,
            str(result.episode_id or ""),
            str(result.title or ""),
            result.source_url,
            str(result.message or ""),
        )
    console.print(table)


@app.command("discover-url")
def discover_url_command(
    url: str = typer.Argument(..., help="Media URL, playlist, or page containing media links."),
    playlist_mode: PlaylistMode = typer.Option(PlaylistMode.all, "--playlist-mode"),
    playlist_order: PlaylistOrder = typer.Option(PlaylistOrder.source, "--playlist-order"),
    max_items: int | None = typer.Option(None, "--max-items", min=1),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
) -> None:
    """Preview media sources found in a URL without downloading them."""
    sources = discover_sources(
        url,
        playlist_mode=playlist_mode,
        playlist_order=playlist_order,
        max_items=max_items,
    )
    if as_json:
        console.print_json(
            json.dumps(
                [
                    {
                        "url": source.url,
                        "title": source.title,
                        "webpage_url": source.webpage_url,
                        "source_type": source.source_type,
                    }
                    for source in sources
                ],
                ensure_ascii=False,
            )
        )
        return

    table = Table("No.", "Type", "Title", "URL")
    for index, source in enumerate(sources, start=1):
        table.add_row(str(index), source.source_type, str(source.title or ""), source.webpage_url or source.url)
    console.print(table)


@app.command("transcribe-audio")
def transcribe_audio_command(
    audio_path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    title: str = typer.Option(..., "--title", "-t", help="Episode title."),
    source_url: str | None = typer.Option(None, "--source-url", help="Original media URL."),
    author: str | None = typer.Option(None, "--author", help="Podcast author or channel."),
    whisper_model: str = typer.Option("small", "--whisper-model", help="faster-whisper model size."),
    device: str = typer.Option("auto", "--device", help="Whisper device: auto, cpu, cuda, or mps."),
    compute_type: str = typer.Option("auto", "--compute-type", help="Whisper compute type."),
    language: str | None = typer.Option(None, "--language", help="Optional language code, e.g. es."),
    data_dir: Path = data_dir_option(),
) -> None:
    """Transcribe a local audio file with faster-whisper and import it."""
    settings = build_settings(data_dir)
    segments, detected_language = transcribe_audio(
        audio_path,
        model_size=whisper_model,
        device=device,
        compute_type=compute_type,
        language=language,
        transcript_dir=settings.transcript_dir,
    )
    episode_id = add_episode(
        settings.db_path,
        title=title,
        segments=segments,
        source_url=source_url,
        author=author,
        language=language or detected_language,
    )
    console.print(f"Transcribed and imported episode [bold]{episode_id}[/bold] with {len(segments)} segments.")


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
