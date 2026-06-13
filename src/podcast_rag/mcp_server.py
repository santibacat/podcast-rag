from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

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
from podcast_rag.corpora import create_corpus as create_corpus_config
from podcast_rag.corpora import corpus_to_dict, get_corpus, load_corpora, resolve_corpus_set, resolve_corpus_settings
from podcast_rag.db import init_db, rebuild_entities
from podcast_rag.discovery import PlaylistMode, PlaylistOrder, discover_sources
from podcast_rag.ingest import ingest_url as run_ingest_url
from podcast_rag.retrieval_qdrant import (
    DEFAULT_QDRANT_COLLECTION,
    DEFAULT_QDRANT_DENSE_MODEL,
    DEFAULT_QDRANT_SPARSE_MODEL,
    QdrantIndexConfig,
    index_qdrant_chunks,
)
from podcast_rag.agent_tools import (
    AgentToolConfig,
    agentic_research,
    tool_connections,
    tool_context,
    tool_lexical_search,
    tool_list_episodes,
    tool_related,
    tool_retrieve,
    tool_topics,
)

mcp = FastMCP("podcast-rag")


def _config(data_dir: str = "data", qdrant_url: str | None = None) -> AgentToolConfig:
    return AgentToolConfig(data_dir=Path(data_dir), qdrant_url=qdrant_url)


def _corpus_config(base_data_dir: str = "data", corpus: str | None = None, qdrant_url: str | None = None) -> AgentToolConfig:
    settings = resolve_corpus_settings(Path(base_data_dir), corpus, qdrant_url=qdrant_url)
    return AgentToolConfig(data_dir=settings.data_dir, qdrant_url=settings.qdrant_url)


@mcp.tool()
def list_corpora(data_dir: str = "data") -> dict[str, Any]:
    """List registered corpus configurations available to agents and dashboards."""
    base = Path(data_dir)
    return {
        "default": {"id": "default", "name": "Default corpus", "data_dir": str(base)},
        "corpora": [corpus_to_dict(corpus) for corpus in load_corpora(base)],
    }


@mcp.tool()
def create_corpus(
    corpus_id: str,
    name: str | None = None,
    description: str | None = None,
    domain_profile: str | None = None,
    qdrant_url: str | None = None,
    tags: list[str] | None = None,
    data_dir: str = "data",
    corpus_data_dir: str | None = None,
) -> dict[str, Any]:
    """Create a named corpus configuration and initialize its SQLite database."""
    corpus = create_corpus_config(
        base_data_dir=Path(data_dir),
        corpus_id=corpus_id,
        name=name,
        data_dir=Path(corpus_data_dir) if corpus_data_dir else None,
        description=description,
        domain_profile=domain_profile,
        qdrant_url=qdrant_url,
        tags=tags or [],
    )
    init_db(Path(corpus.data_dir) / "podcast_rag.sqlite3")
    return corpus_to_dict(corpus)


@mcp.tool()
def discover_url(
    url: str,
    playlist_mode: str = "all",
    playlist_order: str = "source",
    max_items: int | None = None,
) -> list[dict[str, Any]]:
    """Discover media, YouTube videos, playlist entries, or podcast files from a URL without downloading."""
    sources = discover_sources(
        url,
        playlist_mode=PlaylistMode(playlist_mode),
        playlist_order=PlaylistOrder(playlist_order),
        max_items=max_items,
    )
    return [
        {
            "url": source.url,
            "title": source.title,
            "webpage_url": source.webpage_url,
            "source_type": source.source_type,
        }
        for source in sources
    ]


@mcp.tool()
def ingest_url(
    url: str,
    corpus: str | None = None,
    playlist_mode: str = "single",
    playlist_order: str = "source",
    max_items: int | None = None,
    whisper_model: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = None,
    transcribe_seconds: int | None = None,
    domain_profile: str | None = None,
    skip_existing: bool = True,
    data_dir: str = "data",
) -> list[dict[str, Any]]:
    """Download/transcribe/import a URL into a corpus. Supports pages, media files, YouTube videos, and playlists."""
    base = Path(data_dir)
    settings = resolve_corpus_settings(base, corpus)
    if corpus and domain_profile is None:
        domain_profile = get_corpus(base, corpus).domain_profile
    results = run_ingest_url(
        url=url,
        settings=settings,
        playlist_mode=PlaylistMode(playlist_mode),
        playlist_order=PlaylistOrder(playlist_order),
        max_items=max_items,
        whisper_model=whisper_model,
        device=device,
        compute_type=compute_type,
        language=language,
        domain_profile=domain_profile,
        skip_existing=skip_existing,
        transcribe_seconds=transcribe_seconds,
    )
    return [result.__dict__ for result in results]


@mcp.tool()
def rebuild_corpus_entities(corpus: str | None = None, domain_profile: str | None = None, data_dir: str = "data") -> dict[str, int]:
    """Rebuild entities, mentions, and relations for a corpus."""
    base = Path(data_dir)
    settings = resolve_corpus_settings(base, corpus)
    if corpus and domain_profile is None:
        domain_profile = get_corpus(base, corpus).domain_profile
    return rebuild_entities(settings.db_path, domain_profile=domain_profile)


@mcp.tool()
def index_retrieval(
    corpus: str | None = None,
    collection: str = DEFAULT_QDRANT_COLLECTION,
    dense_model: str = DEFAULT_QDRANT_DENSE_MODEL,
    sparse_model: str = DEFAULT_QDRANT_SPARSE_MODEL,
    qdrant_url: str | None = None,
    batch_size: int = 64,
    force: bool = False,
    data_dir: str = "data",
) -> dict[str, Any]:
    """Index a corpus into Qdrant for hybrid dense+sparse retrieval."""
    settings = resolve_corpus_settings(Path(data_dir), corpus, qdrant_url=qdrant_url)
    config = QdrantIndexConfig(collection_name=collection, dense_model=dense_model, sparse_model=sparse_model, url=settings.qdrant_url)
    indexed = index_qdrant_chunks(settings.db_path, settings.qdrant_dir, config=config, batch_size=batch_size, force=force)
    return {"corpus": corpus or "default", "collection": collection, "indexed_chunks": indexed, "data_dir": str(settings.data_dir)}


@mcp.tool()
def episodes(data_dir: str = "data", corpus: str | None = None) -> list[dict[str, Any]]:
    """List ingested podcast/video episodes."""
    return tool_list_episodes(_corpus_config(data_dir, corpus))


@mcp.tool()
def topics(limit: int = 50, data_dir: str = "data", corpus: str | None = None) -> list[dict[str, Any]]:
    """List detected entities/topics with type, confidence, and counts."""
    return tool_topics(limit=limit, config=_corpus_config(data_dir, corpus))


@mcp.tool()
def connections(topic: str | None = None, limit: int = 50, data_dir: str = "data", corpus: str | None = None) -> list[dict[str, Any]]:
    """Show semantic connections between detected entities."""
    return tool_connections(topic=topic, limit=limit, config=_corpus_config(data_dir, corpus))


@mcp.tool()
def related(topic: str, limit: int = 25, data_dir: str = "data", corpus: str | None = None) -> list[dict[str, Any]]:
    """Show topics that co-occur with a topic in transcript chunks."""
    return tool_related(topic=topic, limit=limit, config=_corpus_config(data_dir, corpus))


@mcp.tool()
def context(
    chunk_id: int,
    before_segments: int = 2,
    after_segments: int = 2,
    data_dir: str = "data",
    corpus: str | None = None,
) -> dict[str, Any]:
    """Return transcript context around a chunk."""
    return tool_context(
        chunk_id=chunk_id,
        before_segments=before_segments,
        after_segments=after_segments,
        config=_corpus_config(data_dir, corpus),
    )


@mcp.tool()
def lexical_search(query: str, limit: int = 10, data_dir: str = "data", corpus: str | None = None) -> list[dict[str, Any]]:
    """Run SQLite FTS/BM25 lexical search over transcript chunks."""
    return tool_lexical_search(query=query, limit=limit, config=_corpus_config(data_dir, corpus))


@mcp.tool()
def retrieve(
    query: str,
    limit: int = 5,
    before_segments: int = 2,
    after_segments: int = 2,
    episode_id: int | None = None,
    topic: str | None = None,
    data_dir: str = "data",
    corpus: str | None = None,
    qdrant_url: str | None = None,
) -> list[dict[str, Any]]:
    """Run Qdrant hybrid retrieval and return evidence with expanded transcript context."""
    return tool_retrieve(
        query=query,
        limit=limit,
        before_segments=before_segments,
        after_segments=after_segments,
        episode_id=episode_id,
        topic=topic,
        config=_corpus_config(data_dir, corpus, qdrant_url=qdrant_url),
    )


@mcp.tool()
def research(
    question: str,
    limit: int = 5,
    mode: str = "local",
    data_dir: str = "data",
    corpus: str | None = None,
    qdrant_url: str | None = None,
) -> dict[str, Any]:
    """Run agentic retrieval and optionally add LLM synthesis when mode='llm'."""
    return agentic_research(question=question, limit=limit, mode=mode, config=_corpus_config(data_dir, corpus, qdrant_url=qdrant_url))


@mcp.tool()
def research_across_corpora(
    question: str,
    corpus_selector: str = "all",
    limit: int = 5,
    mode: str = "local",
    data_dir: str = "data",
    qdrant_url: str | None = None,
) -> dict[str, Any]:
    """Run agentic retrieval across several registered corpora and return per-corpus briefs."""
    corpora = resolve_corpus_set(Path(data_dir), corpus_selector)
    results = []
    for corpus in corpora:
        settings = resolve_corpus_settings(Path(data_dir), None if corpus.id == "default" else corpus.id, qdrant_url=qdrant_url)
        try:
            result = agentic_research(
                question=question,
                limit=limit,
                mode=mode,
                config=AgentToolConfig(data_dir=settings.data_dir, qdrant_url=settings.qdrant_url),
            )
        except Exception as exc:
            result = {"brief": f"Failed: {exc}", "tool_calls": [], "evidence": []}
        result["corpus_id"] = corpus.id
        result["corpus_name"] = corpus.name
        results.append(result)
    return {
        "question": question,
        "corpus_selector": corpus_selector,
        "brief": "\n\n".join(f"[{item['corpus_name']}]\n{item['brief']}" for item in results),
        "results": results,
    }


@mcp.tool()
def analytics_corpus_stats(data_dir: str = "data", corpus: str | None = None) -> dict[str, Any]:
    """Return corpus-wide analytics and density metrics."""
    settings = resolve_corpus_settings(Path(data_dir), corpus)
    return corpus_stats(settings.db_path)


@mcp.tool()
def analytics_entity_profile(name: str, limit: int = 10, data_dir: str = "data", corpus: str | None = None) -> dict[str, Any]:
    """Return mentions, timestamps, episodes, and connections for an entity."""
    settings = resolve_corpus_settings(Path(data_dir), corpus)
    return entity_profile(settings.db_path, name=name, limit=limit)


@mcp.tool()
def analytics_timeline(
    topic: str | None = None,
    episode_id: int | None = None,
    limit: int = 200,
    data_dir: str = "data",
    corpus: str | None = None,
) -> list[dict[str, Any]]:
    """Return entity/topic timeline entries."""
    settings = resolve_corpus_settings(Path(data_dir), corpus)
    return entity_timeline(settings.db_path, topic=topic, episode_id=episode_id, limit=limit)


@mcp.tool()
def analytics_episode_insights(episode_id: int, limit: int = 20, data_dir: str = "data", corpus: str | None = None) -> dict[str, Any]:
    """Return granular analytics for one episode."""
    settings = resolve_corpus_settings(Path(data_dir), corpus)
    return episode_insights(settings.db_path, episode_id=episode_id, limit=limit)


@mcp.tool()
def analytics_topic_matrix(limit_entities: int = 50, data_dir: str = "data", corpus: str | None = None) -> dict[str, Any]:
    """Return entity-by-episode mention matrix data."""
    settings = resolve_corpus_settings(Path(data_dir), corpus)
    return topic_episode_matrix(settings.db_path, limit_entities=limit_entities)


@mcp.tool()
def analytics_graph(min_weight: float = 0.0, limit: int = 1000, data_dir: str = "data", corpus: str | None = None) -> dict[str, Any]:
    """Return entity graph data for visualization."""
    settings = resolve_corpus_settings(Path(data_dir), corpus)
    return graph_export(settings.db_path, min_weight=min_weight, limit=limit)


@mcp.tool()
def analytics_quality_report(data_dir: str = "data", corpus: str | None = None) -> dict[str, Any]:
    """Return quality/debug signals for transcripts, entities, and chunks."""
    settings = resolve_corpus_settings(Path(data_dir), corpus)
    return quality_report(settings.db_path)


@mcp.tool()
def analytics_system_status(data_dir: str = "data", corpus: str | None = None, qdrant_url: str | None = None) -> dict[str, Any]:
    """Return corpus/index state, chunking/profile config, and recommended next actions."""
    settings = resolve_corpus_settings(Path(data_dir), corpus, qdrant_url=qdrant_url)
    return system_status(settings.db_path, settings.data_dir, qdrant_url=settings.qdrant_url)


@mcp.tool()
def chunking_info() -> dict[str, Any]:
    """Return the current transcript chunking strategy."""
    return describe_chunking_strategy()


@mcp.tool()
def domain_profile_info(domain_profile: str = "generic_es") -> dict[str, object]:
    """Return the rules for a domain profile."""
    return profile_explanation(domain_profile)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
