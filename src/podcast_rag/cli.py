from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from podcast_rag.agent_tools import AgentToolConfig, AskMode, agentic_research
from podcast_rag.analytics import (
    corpus_stats,
    entity_profile,
    entity_timeline,
    episode_insights,
    graph_export,
    profile_explanation,
    quality_report,
    system_status,
    topic_episode_matrix,
)
from podcast_rag.chunking import describe_chunking_strategy
from podcast_rag.config import build_settings
from podcast_rag.corpora import create_corpus, get_corpus, load_corpora, resolve_corpus_settings
from podcast_rag.db import add_episode, init_db, rebuild_entities
from podcast_rag.discovery import PlaylistMode, PlaylistOrder, discover_sources
from podcast_rag.domain_profiles import DEFAULT_DOMAIN_PROFILE, list_domain_profiles
from podcast_rag.embeddings import DEFAULT_EMBEDDING_MODEL, build_embedder, rebuild_chunk_embeddings, semantic_search
from podcast_rag.ingest import ingest_url
from podcast_rag.retrieval_qdrant import (
    DEFAULT_QDRANT_COLLECTION,
    DEFAULT_QDRANT_DENSE_MODEL,
    DEFAULT_QDRANT_SPARSE_MODEL,
    QdrantIndexConfig,
    index_qdrant_chunks,
    qdrant_hybrid_search,
    retrieve_evidence,
)
from podcast_rag.search import (
    entity_connections,
    get_chunk_context,
    get_episode_segments,
    list_episodes,
    list_topics,
    related_topics,
    search_chunks,
)
from podcast_rag.timecode import format_timestamp
from podcast_rag.transcription import transcribe_audio
from podcast_rag.transcripts import parse_transcript_file
from podcast_rag.workflows import process_url_workflow

app = typer.Typer(no_args_is_help=True)
console = Console()


def data_dir_option() -> Path:
    return typer.Option(Path("data"), "--data-dir", help="Directory for SQLite DB and local artifacts.")


def corpus_option() -> str | None:
    return typer.Option(None, "--corpus", help="Registered corpus id. Uses --data-dir directly when omitted.")


@app.command("init-db")
def init_db_command(data_dir: Path = data_dir_option()) -> None:
    """Create the local SQLite database."""
    settings = build_settings(data_dir)
    init_db(settings.db_path)
    console.print(f"Initialized database at [bold]{settings.db_path}[/bold]")


@app.command("corpora")
def corpora_command(data_dir: Path = data_dir_option()) -> None:
    """List registered corpus configurations."""
    table = Table("ID", "Name", "Data Dir", "Profile", "Qdrant", "Tags")
    table.add_row("default", "Default corpus", str(data_dir), "", "", "")
    for corpus in load_corpora(data_dir):
        table.add_row(
            corpus.id,
            corpus.name,
            corpus.data_dir,
            str(corpus.domain_profile or ""),
            str(corpus.qdrant_url or ""),
            ", ".join(corpus.tags),
        )
    console.print(table)


@app.command("create-corpus")
def create_corpus_command(
    corpus_id: str = typer.Argument(..., help="Stable corpus id, e.g. memorias-tambor."),
    name: str | None = typer.Option(None, "--name", help="Display name."),
    corpus_data_dir: Path | None = typer.Option(None, "--corpus-data-dir", help="Storage directory for this corpus."),
    description: str | None = typer.Option(None, "--description"),
    domain_profile: str | None = typer.Option(None, "--domain-profile"),
    qdrant_url: str | None = typer.Option(None, "--qdrant-url"),
    tag: list[str] | None = typer.Option(None, "--tag", help="Repeatable corpus tag."),
    data_dir: Path = data_dir_option(),
) -> None:
    """Create a registered corpus configuration."""
    try:
        corpus = create_corpus(
            base_data_dir=data_dir,
            corpus_id=corpus_id,
            name=name,
            data_dir=corpus_data_dir,
            description=description,
            domain_profile=domain_profile,
            qdrant_url=qdrant_url,
            tags=tag or [],
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    init_db(Path(corpus.data_dir) / "podcast_rag.sqlite3")
    console.print(f"Created corpus [bold]{corpus.id}[/bold] at [bold]{corpus.data_dir}[/bold].")


@app.command("domain-profiles")
def domain_profiles_command() -> None:
    """List available entity extraction domain profiles."""
    table = Table("Profile")
    for profile in list_domain_profiles():
        table.add_row(profile)
    console.print(table)


@app.command("profile-info")
def profile_info_command(
    domain_profile: str = typer.Argument(DEFAULT_DOMAIN_PROFILE, help="Domain profile name."),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
) -> None:
    """Explain a domain profile's entity extraction rules."""
    try:
        profile = profile_explanation(domain_profile)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if as_json:
        console.print_json(json.dumps(profile, ensure_ascii=False))
        return

    console.print(f"[bold]{profile['name']}[/bold]")
    table = Table("Rule Group", "Values")
    for key, value in profile.items():
        if key == "name":
            continue
        if isinstance(value, list):
            rendered = ", ".join(str(item) for item in value[:40])
        else:
            rendered = str(value)
        table.add_row(key, rendered)
    console.print(table)


@app.command("chunking-info")
def chunking_info_command(as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON.")) -> None:
    """Explain the current transcript chunking strategy."""
    info = describe_chunking_strategy()
    if as_json:
        console.print_json(json.dumps(info, ensure_ascii=False))
        return
    table = Table("Setting", "Value")
    for key, value in info.items():
        table.add_row(key, str(value))
    console.print(table)


@app.command("qdrant-health")
def qdrant_health_command(
    qdrant_url: str = typer.Option("http://localhost:6333", "--qdrant-url", help="Qdrant server URL."),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
) -> None:
    """Check a Qdrant server deployment."""
    import httpx

    url = qdrant_url.rstrip("/")
    try:
        response = httpx.get(f"{url}/", timeout=5)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise typer.BadParameter(f"Qdrant server is not reachable at {url}: {exc}") from exc

    result = {"url": url, "status": "ok", "response": payload}
    if as_json:
        console.print_json(json.dumps(result, ensure_ascii=False))
        return
    console.print(f"Qdrant server [bold]OK[/bold] at [bold]{url}[/bold]")
    console.print_json(json.dumps(payload, ensure_ascii=False))


@app.command("ingest-transcript")
def ingest_transcript_command(
    transcript_path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    title: str = typer.Option(..., "--title", "-t", help="Episode title."),
    source_url: str | None = typer.Option(None, "--source-url", help="Original media URL."),
    author: str | None = typer.Option(None, "--author", help="Podcast author or channel."),
    language: str | None = typer.Option(None, "--language", help="Transcript language code."),
    domain_profile: str = typer.Option(DEFAULT_DOMAIN_PROFILE, "--domain-profile", help="Entity extraction domain profile."),
    corpus: str | None = corpus_option(),
    data_dir: Path = data_dir_option(),
) -> None:
    """Import a plain text or timestamped transcript."""
    settings = resolve_corpus_settings(data_dir, corpus)
    if corpus and domain_profile == DEFAULT_DOMAIN_PROFILE:
        domain_profile = get_corpus(data_dir, corpus).domain_profile or domain_profile
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
        domain_profile=domain_profile,
    )
    console.print(f"Imported episode [bold]{episode_id}[/bold] with {len(segments)} transcript segments.")


@app.command("rebuild-entities")
def rebuild_entities_command(
    domain_profile: str = typer.Option(DEFAULT_DOMAIN_PROFILE, "--domain-profile", help="Entity extraction domain profile."),
    corpus: str | None = corpus_option(),
    data_dir: Path = data_dir_option(),
) -> None:
    """Rebuild entities, mentions, and relations for all chunks using a domain profile."""
    settings = resolve_corpus_settings(data_dir, corpus)
    if corpus and domain_profile == DEFAULT_DOMAIN_PROFILE:
        domain_profile = get_corpus(data_dir, corpus).domain_profile or domain_profile
    try:
        result = rebuild_entities(settings.db_path, domain_profile=domain_profile)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(
        f"Rebuilt entities with [bold]{domain_profile}[/bold]: "
        f"{result['entities']} entities, {result['mentions']} mentions, {result['relations']} relations."
    )


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
    device: str = typer.Option("cpu", "--device", help="Whisper device: cpu, auto, cuda, or mps."),
    compute_type: str = typer.Option("int8", "--compute-type", help="Whisper compute type."),
    language: str | None = typer.Option(None, "--language", help="Optional language code, e.g. es."),
    transcribe_seconds: int | None = typer.Option(
        None,
        "--transcribe-seconds",
        min=1,
        help="Only transcribe the first N seconds. Useful for smoke tests.",
    ),
    domain_profile: str = typer.Option(DEFAULT_DOMAIN_PROFILE, "--domain-profile", help="Entity extraction domain profile."),
    no_skip_existing: bool = typer.Option(False, "--no-skip-existing", help="Reprocess URLs already in the DB."),
    corpus: str | None = corpus_option(),
    data_dir: Path = data_dir_option(),
) -> None:
    """Download URL audio, transcribe it locally, and import the transcript."""
    settings = resolve_corpus_settings(data_dir, corpus)
    if corpus and domain_profile == DEFAULT_DOMAIN_PROFILE:
        domain_profile = get_corpus(data_dir, corpus).domain_profile or domain_profile
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
        domain_profile=domain_profile,
        skip_existing=not no_skip_existing,
        transcribe_seconds=transcribe_seconds,
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


@app.command("process-url")
def process_url_command(
    url: str = typer.Argument(..., help="Media URL, YouTube URL, playlist, or page containing podcast/video links."),
    corpus: str | None = corpus_option(),
    create_corpus_if_missing: bool = typer.Option(False, "--create-corpus", help="Create --corpus automatically if it does not exist."),
    corpus_name: str | None = typer.Option(None, "--corpus-name", help="Display name when --create-corpus is used."),
    playlist_mode: PlaylistMode = typer.Option(PlaylistMode.all, "--playlist-mode", help="Expand pages/playlists by default."),
    playlist_order: PlaylistOrder = typer.Option(PlaylistOrder.source, "--playlist-order"),
    max_items: int | None = typer.Option(None, "--max-items", min=1),
    whisper_model: str = typer.Option("small", "--whisper-model", help="faster-whisper model size."),
    device: str = typer.Option("cpu", "--device", help="Whisper device: cpu, auto, cuda, or mps."),
    compute_type: str = typer.Option("int8", "--compute-type", help="Whisper compute type."),
    language: str | None = typer.Option(None, "--language", help="Optional language code, e.g. es."),
    transcribe_seconds: int | None = typer.Option(None, "--transcribe-seconds", min=1, help="Only transcribe first N seconds."),
    domain_profile: str = typer.Option(DEFAULT_DOMAIN_PROFILE, "--domain-profile", help="Entity extraction domain profile."),
    qdrant_url: str | None = typer.Option(None, "--qdrant-url", help="Qdrant server URL. Defaults to QDRANT_URL or local storage."),
    collection: str = typer.Option(DEFAULT_QDRANT_COLLECTION, "--collection", help="Qdrant collection name."),
    batch_size: int = typer.Option(64, "--batch-size", min=1, max=512),
    force_index: bool = typer.Option(False, "--force-index", help="Recreate the Qdrant collection after processing."),
    no_rebuild_entities: bool = typer.Option(False, "--no-rebuild-entities", help="Skip entity/relation rebuild."),
    no_index: bool = typer.Option(False, "--no-index", help="Skip Qdrant indexing."),
    no_skip_existing: bool = typer.Option(False, "--no-skip-existing", help="Reprocess URLs already in the DB."),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
    data_dir: Path = data_dir_option(),
) -> None:
    """Process a URL end-to-end: ingest, rebuild entities, and index retrieval."""
    result = process_url_workflow(
        url=url,
        data_dir=data_dir,
        corpus=corpus,
        create_missing_corpus=create_corpus_if_missing,
        corpus_name=corpus_name,
        playlist_mode=playlist_mode,
        playlist_order=playlist_order,
        max_items=max_items,
        whisper_model=whisper_model,
        device=device,
        compute_type=compute_type,
        language=language,
        domain_profile=domain_profile,
        skip_existing=not no_skip_existing,
        transcribe_seconds=transcribe_seconds,
        qdrant_url=qdrant_url,
        collection=collection,
        batch_size=batch_size,
        force_index=force_index,
        rebuild=not no_rebuild_entities,
        index=not no_index,
    )
    if as_json:
        console.print_json(json.dumps(result, ensure_ascii=False))
        return

    console.print(f"Processed [bold]{result['source_url']}[/bold]")
    console.print(f"Corpus: [bold]{result['corpus']}[/bold] | Data dir: [bold]{result['data_dir']}[/bold]")
    if result.get("created_corpus"):
        console.print(f"Created corpus [bold]{result['created_corpus']['id']}[/bold].")
    table = Table("Status", "Episode", "Title", "Message")
    for item in result["ingest"]:
        table.add_row(str(item["status"]), str(item.get("episode_id") or ""), str(item.get("title") or ""), str(item.get("message") or ""))
    console.print(table)
    if result.get("entities"):
        entities = result["entities"]
        console.print(
            f"Entities rebuilt: [bold]{entities['entities']}[/bold] entities, "
            f"[bold]{entities['mentions']}[/bold] mentions, [bold]{entities['relations']}[/bold] relations."
        )
    if result.get("index", {}).get("enabled"):
        console.print(
            f"Indexed [bold]{result['index']['indexed_chunks']}[/bold] chunks into "
            f"[bold]{result['index']['collection']}[/bold]."
        )


@app.command("transcribe-audio")
def transcribe_audio_command(
    audio_path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    title: str = typer.Option(..., "--title", "-t", help="Episode title."),
    source_url: str | None = typer.Option(None, "--source-url", help="Original media URL."),
    author: str | None = typer.Option(None, "--author", help="Podcast author or channel."),
    whisper_model: str = typer.Option("small", "--whisper-model", help="faster-whisper model size."),
    device: str = typer.Option("cpu", "--device", help="Whisper device: cpu, auto, cuda, or mps."),
    compute_type: str = typer.Option("int8", "--compute-type", help="Whisper compute type."),
    language: str | None = typer.Option(None, "--language", help="Optional language code, e.g. es."),
    transcribe_seconds: int | None = typer.Option(
        None,
        "--transcribe-seconds",
        min=1,
        help="Only transcribe the first N seconds. Useful for smoke tests.",
    ),
    domain_profile: str = typer.Option(DEFAULT_DOMAIN_PROFILE, "--domain-profile", help="Entity extraction domain profile."),
    corpus: str | None = corpus_option(),
    data_dir: Path = data_dir_option(),
) -> None:
    """Transcribe a local audio file with faster-whisper and import it."""
    settings = resolve_corpus_settings(data_dir, corpus)
    if corpus and domain_profile == DEFAULT_DOMAIN_PROFILE:
        domain_profile = get_corpus(data_dir, corpus).domain_profile or domain_profile
    segments, detected_language = transcribe_audio(
        audio_path,
        model_size=whisper_model,
        device=device,
        compute_type=compute_type,
        language=language,
        transcript_dir=settings.transcript_dir,
        transcribe_seconds=transcribe_seconds,
    )
    episode_id = add_episode(
        settings.db_path,
        title=title,
        segments=segments,
        source_url=source_url,
        author=author,
        language=language or detected_language,
        domain_profile=domain_profile,
    )
    console.print(f"Transcribed and imported episode [bold]{episode_id}[/bold] with {len(segments)} segments.")


@app.command("episodes")
def episodes_command(corpus: str | None = corpus_option(), data_dir: Path = data_dir_option()) -> None:
    """List ingested episodes."""
    settings = resolve_corpus_settings(data_dir, corpus)
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
    corpus: str | None = corpus_option(),
    data_dir: Path = data_dir_option(),
) -> None:
    """Search transcript chunks."""
    settings = resolve_corpus_settings(data_dir, corpus)
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


@app.command("index-embeddings")
def index_embeddings_command(
    model_name: str = typer.Option(DEFAULT_EMBEDDING_MODEL, "--model", help="sentence-transformers model name."),
    batch_size: int = typer.Option(32, "--batch-size", min=1, max=256),
    force: bool = typer.Option(False, "--force", help="Rebuild existing embeddings for this model."),
    data_dir: Path = data_dir_option(),
) -> None:
    """Build or refresh the local semantic embedding index."""
    settings = build_settings(data_dir)
    embedder = build_embedder(model_name)
    indexed = rebuild_chunk_embeddings(settings.db_path, embedder=embedder, batch_size=batch_size, force=force)
    console.print(f"Indexed [bold]{indexed}[/bold] transcript chunks with [bold]{model_name}[/bold].")


@app.command("semantic-search")
def semantic_search_command(
    query: str = typer.Argument(..., help="Natural language semantic query."),
    limit: int = typer.Option(10, "--limit", "-n", min=1, max=50),
    model_name: str = typer.Option(DEFAULT_EMBEDDING_MODEL, "--model", help="sentence-transformers model name."),
    data_dir: Path = data_dir_option(),
) -> None:
    """Search transcript chunks by semantic similarity."""
    settings = build_settings(data_dir)
    embedder = build_embedder(model_name)
    results = semantic_search(settings.db_path, query=query, embedder=embedder, limit=limit)
    if not results:
        console.print("No semantic results. Run index-embeddings first.")
        return

    for result in results:
        timestamp = format_timestamp(result["start_seconds"])
        score = float(result["score"])
        console.print(f"[bold]Episode {result['episode_id']} - {result['title']}[/bold] [{timestamp}] score={score:.3f}")
        if result["source_url"]:
            console.print(str(result["source_url"]))
        console.print(str(result["text"]))
        console.print()


@app.command("index-retrieval")
def index_retrieval_command(
    collection: str = typer.Option(DEFAULT_QDRANT_COLLECTION, "--collection", help="Qdrant collection name."),
    dense_model: str = typer.Option(DEFAULT_QDRANT_DENSE_MODEL, "--dense-model", help="FastEmbed dense model."),
    sparse_model: str = typer.Option(DEFAULT_QDRANT_SPARSE_MODEL, "--sparse-model", help="FastEmbed sparse/BM25 model."),
    qdrant_url: str | None = typer.Option(None, "--qdrant-url", help="Qdrant server URL. Defaults to QDRANT_URL or local storage."),
    batch_size: int = typer.Option(64, "--batch-size", min=1, max=512),
    force: bool = typer.Option(False, "--force", help="Recreate the Qdrant collection before indexing."),
    corpus: str | None = corpus_option(),
    data_dir: Path = data_dir_option(),
) -> None:
    """Index transcript chunks into local Qdrant for hybrid retrieval."""
    settings = resolve_corpus_settings(data_dir, corpus, qdrant_url=qdrant_url)
    config = QdrantIndexConfig(
        collection_name=collection,
        dense_model=dense_model,
        sparse_model=sparse_model,
        url=settings.qdrant_url,
    )
    indexed = index_qdrant_chunks(
        db_path=settings.db_path,
        qdrant_dir=settings.qdrant_dir,
        config=config,
        batch_size=batch_size,
        force=force,
    )
    console.print(f"Indexed [bold]{indexed}[/bold] chunks into Qdrant collection [bold]{collection}[/bold].")


@app.command("hybrid-search")
def hybrid_search_command(
    query: str = typer.Argument(..., help="Natural language or keyword query."),
    limit: int = typer.Option(10, "--limit", "-n", min=1, max=50),
    prefetch_limit: int = typer.Option(40, "--prefetch-limit", min=1, max=500),
    collection: str = typer.Option(DEFAULT_QDRANT_COLLECTION, "--collection", help="Qdrant collection name."),
    dense_model: str = typer.Option(DEFAULT_QDRANT_DENSE_MODEL, "--dense-model", help="FastEmbed dense model."),
    sparse_model: str = typer.Option(DEFAULT_QDRANT_SPARSE_MODEL, "--sparse-model", help="FastEmbed sparse/BM25 model."),
    qdrant_url: str | None = typer.Option(None, "--qdrant-url", help="Qdrant server URL. Defaults to QDRANT_URL or local storage."),
    episode_id: int | None = typer.Option(None, "--episode-id", help="Restrict results to one episode."),
    topic: str | None = typer.Option(None, "--topic", help="Restrict results to chunks mentioning this topic/entity."),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
    corpus: str | None = corpus_option(),
    data_dir: Path = data_dir_option(),
) -> None:
    """Search with Qdrant hybrid dense+sparse retrieval."""
    settings = resolve_corpus_settings(data_dir, corpus, qdrant_url=qdrant_url)
    config = QdrantIndexConfig(
        collection_name=collection,
        dense_model=dense_model,
        sparse_model=sparse_model,
        url=settings.qdrant_url,
    )
    results = qdrant_hybrid_search(
        query=query,
        qdrant_dir=settings.qdrant_dir,
        config=config,
        limit=limit,
        prefetch_limit=prefetch_limit,
        episode_id=episode_id,
        topic=topic,
    )
    if not results:
        console.print("No hybrid results. Run index-retrieval first.")
        return

    if as_json:
        console.print_json(json.dumps(results, ensure_ascii=False))
        return

    for result in results:
        timestamp = format_timestamp(_optional_float(result.get("start_seconds")))
        score = float(result["score"])
        console.print(f"[bold]Episode {result['episode_id']} - {result['title']}[/bold] [{timestamp}] score={score:.3f}")
        if result.get("source_url"):
            console.print(str(result["source_url"]))
        if result.get("entities"):
            console.print(f"[dim]Topics: {', '.join(str(item) for item in result['entities'])}[/dim]")
        console.print(str(result["text"]))
        console.print()


@app.command("retrieve")
def retrieve_command(
    query: str = typer.Argument(..., help="Question or retrieval query."),
    limit: int = typer.Option(5, "--limit", "-n", min=1, max=25),
    prefetch_limit: int = typer.Option(40, "--prefetch-limit", min=1, max=500),
    before_segments: int = typer.Option(2, "--before", min=0, max=20, help="Segments before each hit."),
    after_segments: int = typer.Option(2, "--after", min=0, max=20, help="Segments after each hit."),
    episode_id: int | None = typer.Option(None, "--episode-id", help="Restrict results to one episode."),
    topic: str | None = typer.Option(None, "--topic", help="Restrict results to chunks mentioning this topic/entity."),
    collection: str = typer.Option(DEFAULT_QDRANT_COLLECTION, "--collection", help="Qdrant collection name."),
    dense_model: str = typer.Option(DEFAULT_QDRANT_DENSE_MODEL, "--dense-model", help="FastEmbed dense model."),
    sparse_model: str = typer.Option(DEFAULT_QDRANT_SPARSE_MODEL, "--sparse-model", help="FastEmbed sparse/BM25 model."),
    qdrant_url: str | None = typer.Option(None, "--qdrant-url", help="Qdrant server URL. Defaults to QDRANT_URL or local storage."),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
    corpus: str | None = corpus_option(),
    data_dir: Path = data_dir_option(),
) -> None:
    """Retrieve hybrid evidence with expanded transcript context."""
    settings = resolve_corpus_settings(data_dir, corpus, qdrant_url=qdrant_url)
    config = QdrantIndexConfig(
        collection_name=collection,
        dense_model=dense_model,
        sparse_model=sparse_model,
        url=settings.qdrant_url,
    )
    evidence = retrieve_evidence(
        query=query,
        db_path=settings.db_path,
        qdrant_dir=settings.qdrant_dir,
        config=config,
        limit=limit,
        prefetch_limit=prefetch_limit,
        before_segments=before_segments,
        after_segments=after_segments,
        episode_id=episode_id,
        topic=topic,
    )
    if not evidence:
        console.print("No evidence found. Run index-retrieval first.")
        return

    if as_json:
        console.print_json(json.dumps(evidence, ensure_ascii=False))
        return

    for item in evidence:
        timestamp = format_timestamp(_optional_float(item.get("start_seconds")))
        score = float(item["score"])
        console.print(f"[bold]Episode {item['episode_id']} - {item['title']}[/bold] [{timestamp}] score={score:.3f}")
        if item.get("source_url"):
            console.print(str(item["source_url"]))
        if item.get("entities"):
            console.print(f"[dim]Topics: {', '.join(str(entity) for entity in item['entities'])}[/dim]")
        console.print(str(item["context_text"]))
        console.print()


@app.command("ask")
def ask_command(
    question: str = typer.Argument(..., help="Question to investigate with local retrieval tools."),
    limit: int = typer.Option(5, "--limit", "-n", min=1, max=25),
    mode: AskMode = typer.Option(AskMode.local, "--mode", help="Use local retrieval only or add LLM synthesis."),
    collection: str = typer.Option(DEFAULT_QDRANT_COLLECTION, "--collection", help="Qdrant collection name."),
    dense_model: str = typer.Option(DEFAULT_QDRANT_DENSE_MODEL, "--dense-model", help="FastEmbed dense model."),
    sparse_model: str = typer.Option(DEFAULT_QDRANT_SPARSE_MODEL, "--sparse-model", help="FastEmbed sparse/BM25 model."),
    qdrant_url: str | None = typer.Option(None, "--qdrant-url", help="Qdrant server URL. Defaults to QDRANT_URL or local storage."),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
    corpus: str | None = corpus_option(),
    data_dir: Path = data_dir_option(),
) -> None:
    """Run a local agentic retrieval workflow over the indexed podcast corpus."""
    settings = resolve_corpus_settings(data_dir, corpus, qdrant_url=qdrant_url)
    result = agentic_research(
        question=question,
        limit=limit,
        mode=mode,
        config=AgentToolConfig(
            data_dir=settings.data_dir,
            qdrant_url=settings.qdrant_url,
            collection=collection,
            dense_model=dense_model,
            sparse_model=sparse_model,
        ),
    )
    if as_json:
        console.print_json(json.dumps(result, ensure_ascii=False))
        return

    console.print("[bold]Agentic Retrieval Brief[/bold]")
    console.print(result["brief"])
    if result["tool_calls"]:
        console.print()
        table = Table("Tool", "Topic", "Results")
        for call in result["tool_calls"]:
            table.add_row(str(call["tool"]), str(call.get("topic") or ""), str(call.get("result_count", "")))
        console.print(table)


@app.command("query")
def query_command(
    question: str = typer.Argument(..., help="Question to investigate."),
    limit: int = typer.Option(5, "--limit", "-n", min=1, max=25),
    mode: AskMode = typer.Option(AskMode.local, "--mode", help="Use local retrieval only or add LLM synthesis."),
    collection: str = typer.Option(DEFAULT_QDRANT_COLLECTION, "--collection", help="Qdrant collection name."),
    dense_model: str = typer.Option(DEFAULT_QDRANT_DENSE_MODEL, "--dense-model", help="FastEmbed dense model."),
    sparse_model: str = typer.Option(DEFAULT_QDRANT_SPARSE_MODEL, "--sparse-model", help="FastEmbed sparse/BM25 model."),
    qdrant_url: str | None = typer.Option(None, "--qdrant-url", help="Qdrant server URL. Defaults to QDRANT_URL or local storage."),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
    corpus: str | None = corpus_option(),
    data_dir: Path = data_dir_option(),
) -> None:
    """Alias for ask: query an indexed corpus with local or LLM-assisted retrieval."""
    ask_command(
        question=question,
        limit=limit,
        mode=mode,
        collection=collection,
        dense_model=dense_model,
        sparse_model=sparse_model,
        qdrant_url=qdrant_url,
        as_json=as_json,
        corpus=corpus,
        data_dir=data_dir,
    )


@app.command("show")
def show_command(
    episode_id: int = typer.Argument(..., help="Episode ID."),
    corpus: str | None = corpus_option(),
    data_dir: Path = data_dir_option(),
) -> None:
    """Show an episode transcript."""
    settings = resolve_corpus_settings(data_dir, corpus)
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


@app.command("context")
def context_command(
    chunk_id: int = typer.Argument(..., help="Transcript chunk ID."),
    before_segments: int = typer.Option(2, "--before", min=0, max=20),
    after_segments: int = typer.Option(2, "--after", min=0, max=20),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
    corpus: str | None = corpus_option(),
    data_dir: Path = data_dir_option(),
) -> None:
    """Show transcript context around a chunk."""
    settings = resolve_corpus_settings(data_dir, corpus)
    try:
        context = get_chunk_context(
            settings.db_path,
            chunk_id,
            before_segments=before_segments,
            after_segments=after_segments,
        )
    except LookupError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if as_json:
        console.print_json(json.dumps(context, ensure_ascii=False))
        return

    console.print(f"[bold]Episode {context['episode_id']} - {context['title']}[/bold]")
    for segment in context["segments"]:
        console.print(f"[dim]{format_timestamp(segment['start_seconds'])}[/dim] {segment['text']}")


@app.command("topics")
def topics_command(
    limit: int = typer.Option(50, "--limit", "-n", min=1, max=200),
    corpus: str | None = corpus_option(),
    data_dir: Path = data_dir_option(),
) -> None:
    """List candidate entities and topics."""
    settings = resolve_corpus_settings(data_dir, corpus)
    topics = list_topics(settings.db_path, limit=limit)
    table = Table("Topic", "Type", "Confidence", "Mentions", "Episodes")
    for topic in topics:
        table.add_row(
            str(topic["name"]),
            str(topic["entity_type"]),
            f"{float(topic['confidence']):.2f}",
            str(topic["mentions"]),
            str(topic["episodes"]),
        )
    console.print(table)


@app.command("related")
def related_command(
    topic: str = typer.Argument(..., help="Topic/entity name, e.g. 'Felipe II'."),
    limit: int = typer.Option(25, "--limit", "-n", min=1, max=100),
    corpus: str | None = corpus_option(),
    data_dir: Path = data_dir_option(),
) -> None:
    """Show topics that co-occur with a topic in transcript chunks."""
    settings = resolve_corpus_settings(data_dir, corpus)
    rows = related_topics(settings.db_path, topic, limit=limit)
    if not rows:
        console.print("No related topics found.")
        return
    table = Table("Related Topic", "Mentions", "Shared Chunks", "Episodes")
    for row in rows:
        table.add_row(str(row["name"]), str(row["mentions"]), str(row["shared_chunks"]), str(row["episodes"]))
    console.print(table)


@app.command("connections")
def connections_command(
    topic: str | None = typer.Option(None, "--topic", help="Show only connections involving this topic/entity."),
    limit: int = typer.Option(50, "--limit", "-n", min=1, max=200),
    corpus: str | None = corpus_option(),
    data_dir: Path = data_dir_option(),
) -> None:
    """Show automatically populated semantic connections between entities."""
    settings = resolve_corpus_settings(data_dir, corpus)
    rows = entity_connections(settings.db_path, name=topic, limit=limit)
    if not rows:
        console.print("No entity connections found.")
        return

    table = Table("Source", "Type", "Target", "Type", "Relation", "Weight", "Chunks")
    for row in rows:
        table.add_row(
            str(row["source"]),
            str(row["source_type"]),
            str(row["target"]),
            str(row["target_type"]),
            str(row["relation_type"]),
            f"{float(row['weight']):.2f}",
            str(row["shared_chunks"]),
        )
    console.print(table)


@app.command("corpus-stats")
def corpus_stats_command(
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
    corpus: str | None = corpus_option(),
    data_dir: Path = data_dir_option(),
) -> None:
    """Show corpus-wide analytics and density metrics."""
    settings = resolve_corpus_settings(data_dir, corpus)
    stats = corpus_stats(settings.db_path)
    if as_json:
        console.print_json(json.dumps(stats, ensure_ascii=False))
        return

    counts_table = Table("Metric", "Value")
    for key, value in stats["counts"].items():
        counts_table.add_row(key, str(value))
    counts_table.add_row("avg_entities_per_episode", f"{float(stats['avg_entities_per_episode']):.2f}")
    console.print(counts_table)

    types_table = Table("Entity Type", "Count")
    for row in stats["entity_types"]:
        types_table.add_row(str(row["entity_type"]), str(row["count"]))
    console.print(types_table)

    episodes_table = Table("Episode", "Title", "Unique Entities", "Mentions", "Chunks")
    for row in stats["richest_episodes"]:
        episodes_table.add_row(
            str(row["episode_id"]),
            str(row["title"]),
            str(row["unique_entities"]),
            str(row["mentions"]),
            str(row["chunks"]),
        )
    console.print(episodes_table)


@app.command("entity-profile")
def entity_profile_command(
    name: str = typer.Argument(..., help="Entity/topic name."),
    limit: int = typer.Option(10, "--limit", "-n", min=1, max=100),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
    corpus: str | None = corpus_option(),
    data_dir: Path = data_dir_option(),
) -> None:
    """Show mentions, timestamps, episodes, and connections for an entity."""
    settings = resolve_corpus_settings(data_dir, corpus)
    try:
        profile = entity_profile(settings.db_path, name=name, limit=limit)
    except LookupError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if as_json:
        console.print_json(json.dumps(profile, ensure_ascii=False))
        return

    entity = profile["entity"]
    console.print(
        f"[bold]{entity['name']}[/bold] type={entity['entity_type']} confidence={float(entity['confidence']):.2f}"
    )
    if entity.get("evidence"):
        console.print(f"[dim]{entity['evidence']}[/dim]")

    mentions_table = Table("Episode", "Title", "Time", "Chunk", "Text")
    for row in profile["mentions"]:
        mentions_table.add_row(
            str(row["episode_id"]),
            str(row["title"]),
            format_timestamp(_optional_float(row["start_seconds"])),
            str(row["chunk_id"]),
            str(row["text"])[:120],
        )
    console.print(mentions_table)

    connections_table = Table("Source", "Target", "Relation", "Weight", "Chunks")
    for row in profile["connections"]:
        connections_table.add_row(
            str(row["source"]),
            str(row["target"]),
            str(row["relation_type"]),
            f"{float(row['weight']):.2f}",
            str(row["shared_chunks"]),
        )
    console.print(connections_table)


@app.command("timeline")
def timeline_command(
    topic: str | None = typer.Option(None, "--topic", help="Restrict to a topic/entity."),
    episode_id: int | None = typer.Option(None, "--episode-id", help="Restrict to one episode."),
    limit: int = typer.Option(200, "--limit", "-n", min=1, max=1000),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
    corpus: str | None = corpus_option(),
    data_dir: Path = data_dir_option(),
) -> None:
    """Show an entity/topic timeline across episodes and timestamps."""
    settings = resolve_corpus_settings(data_dir, corpus)
    rows = entity_timeline(settings.db_path, topic=topic, episode_id=episode_id, limit=limit)
    if as_json:
        console.print_json(json.dumps(rows, ensure_ascii=False))
        return

    table = Table("Episode", "Title", "Time", "Topic", "Type", "Count", "Text")
    for row in rows:
        table.add_row(
            str(row["episode_id"]),
            str(row["title"]),
            format_timestamp(_optional_float(row["start_seconds"])),
            str(row["name"]),
            str(row["entity_type"]),
            str(row["count"]),
            str(row["text"])[:90],
        )
    console.print(table)


@app.command("episode-insights")
def episode_insights_command(
    episode_id: int = typer.Argument(..., help="Episode ID."),
    limit: int = typer.Option(20, "--limit", "-n", min=1, max=100),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
    corpus: str | None = corpus_option(),
    data_dir: Path = data_dir_option(),
) -> None:
    """Show granular analytics for one episode."""
    settings = resolve_corpus_settings(data_dir, corpus)
    try:
        insights = episode_insights(settings.db_path, episode_id=episode_id, limit=limit)
    except LookupError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if as_json:
        console.print_json(json.dumps(insights, ensure_ascii=False))
        return

    console.print(f"[bold]Episode {episode_id}: {insights['episode']['title']}[/bold]")
    top_table = Table("Entity", "Type", "Mentions", "Chunks")
    for row in insights["top_entities"]:
        top_table.add_row(str(row["name"]), str(row["entity_type"]), str(row["mentions"]), str(row["chunks"]))
    console.print(top_table)

    density_table = Table("Chunk", "Time", "Unique Entities", "Text")
    for row in insights["entity_density"]:
        density_table.add_row(
            str(row["chunk_id"]),
            format_timestamp(_optional_float(row["start_seconds"])),
            str(row["unique_entities"]),
            str(row["text"])[:100],
        )
    console.print(density_table)


@app.command("topic-matrix")
def topic_matrix_command(
    limit_entities: int = typer.Option(50, "--limit-entities", min=1, max=500),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
    corpus: str | None = corpus_option(),
    data_dir: Path = data_dir_option(),
) -> None:
    """Show entity-by-episode mention matrix data."""
    settings = resolve_corpus_settings(data_dir, corpus)
    matrix = topic_episode_matrix(settings.db_path, limit_entities=limit_entities)
    if as_json:
        console.print_json(json.dumps(matrix, ensure_ascii=False))
        return

    table = Table("Entity", "Type", "Total Mentions", "Episode Mentions")
    episode_titles = {int(row["id"]): row["title"] for row in matrix["episodes"]}
    cells_by_entity: dict[int, list[str]] = {}
    for cell in matrix["cells"]:
        cells_by_entity.setdefault(int(cell["entity_id"]), []).append(
            f"{episode_titles.get(int(cell['episode_id']), cell['episode_id'])}: {cell['mentions']}"
        )
    for entity in matrix["entities"]:
        table.add_row(
            str(entity["name"]),
            str(entity["entity_type"]),
            str(entity["mentions"]),
            "; ".join(cells_by_entity.get(int(entity["id"]), [])),
        )
    console.print(table)


@app.command("graph-export")
def graph_export_command(
    output_path: Path = typer.Argument(..., help="Output JSON path."),
    min_weight: float = typer.Option(0.0, "--min-weight", min=0.0),
    limit: int = typer.Option(1000, "--limit", min=1),
    corpus: str | None = corpus_option(),
    data_dir: Path = data_dir_option(),
) -> None:
    """Export entity graph JSON for visualization."""
    settings = resolve_corpus_settings(data_dir, corpus)
    graph = graph_export(settings.db_path, min_weight=min_weight, limit=limit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"Exported graph with {len(graph['nodes'])} nodes and {len(graph['edges'])} edges to [bold]{output_path}[/bold].")


@app.command("quality-report")
def quality_report_command(
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
    corpus: str | None = corpus_option(),
    data_dir: Path = data_dir_option(),
) -> None:
    """Show quality/debug signals for transcripts, entities, and chunks."""
    settings = resolve_corpus_settings(data_dir, corpus)
    report = quality_report(settings.db_path)
    if as_json:
        console.print_json(json.dumps(report, ensure_ascii=False))
        return

    for section, rows in report.items():
        console.print(f"[bold]{section}[/bold]: {len(rows)}")
        table = Table(*([str(key) for key in rows[0].keys()] if rows else ["No issues"]))
        for row in rows:
            table.add_row(*(str(value)[:100] for value in row.values()))
        console.print(table)


@app.command("system-status")
def system_status_command(
    qdrant_url: str | None = typer.Option(None, "--qdrant-url", help="Qdrant server URL. Defaults to QDRANT_URL or local storage."),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
    corpus: str | None = corpus_option(),
    data_dir: Path = data_dir_option(),
) -> None:
    """Explain corpus/index state and recommended next actions."""
    settings = resolve_corpus_settings(data_dir, corpus, qdrant_url=qdrant_url)
    status = system_status(settings.db_path, settings.data_dir, qdrant_url=settings.qdrant_url)
    if as_json:
        console.print_json(json.dumps(status, ensure_ascii=False))
        return

    console.print("[bold]Podcast RAG Status[/bold]")
    console.print(f"Data dir: {status['data_dir']}")
    console.print(f"SQLite: {status['sqlite_db']}")
    console.print(f"Qdrant mode: {status['qdrant']['mode']}")

    counts_table = Table("Metric", "Value")
    for key, value in status["stats"]["counts"].items():
        counts_table.add_row(key, str(value))
    console.print(counts_table)

    quality_table = Table("Quality Signal", "Count")
    for key, value in status["quality_counts"].items():
        quality_table.add_row(key, str(value))
    console.print(quality_table)

    rec_table = Table("Recommended Next Action")
    for recommendation in status["recommendations"]:
        rec_table.add_row(str(recommendation))
    console.print(rec_table)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


if __name__ == "__main__":
    app()
