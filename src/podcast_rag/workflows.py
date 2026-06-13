from __future__ import annotations

from pathlib import Path
from typing import Any

from podcast_rag.corpora import create_corpus, get_corpus, resolve_corpus_settings
from podcast_rag.db import rebuild_entities
from podcast_rag.discovery import PlaylistMode, PlaylistOrder
from podcast_rag.domain_profiles import DEFAULT_DOMAIN_PROFILE
from podcast_rag.ingest import ingest_url
from podcast_rag.retrieval_qdrant import (
    DEFAULT_QDRANT_COLLECTION,
    DEFAULT_QDRANT_DENSE_MODEL,
    DEFAULT_QDRANT_SPARSE_MODEL,
    QdrantIndexConfig,
    index_qdrant_chunks,
)


def process_url_workflow(
    url: str,
    data_dir: Path = Path("data"),
    corpus: str | None = None,
    create_missing_corpus: bool = False,
    corpus_name: str | None = None,
    playlist_mode: PlaylistMode = PlaylistMode.all,
    playlist_order: PlaylistOrder = PlaylistOrder.source,
    max_items: int | None = None,
    whisper_model: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = None,
    domain_profile: str = DEFAULT_DOMAIN_PROFILE,
    skip_existing: bool = True,
    transcribe_seconds: int | None = None,
    qdrant_url: str | None = None,
    collection: str = DEFAULT_QDRANT_COLLECTION,
    dense_model: str = DEFAULT_QDRANT_DENSE_MODEL,
    sparse_model: str = DEFAULT_QDRANT_SPARSE_MODEL,
    batch_size: int = 64,
    force_index: bool = False,
    rebuild: bool = True,
    index: bool = True,
) -> dict[str, Any]:
    resolved_corpus = corpus
    created_corpus: dict[str, Any] | None = None
    if corpus and corpus != "default":
        try:
            existing = get_corpus(data_dir, corpus)
        except LookupError:
            if not create_missing_corpus:
                raise
            existing = create_corpus(
                base_data_dir=data_dir,
                corpus_id=corpus,
                name=corpus_name,
                domain_profile=domain_profile,
                qdrant_url=qdrant_url,
            )
            created_corpus = {"id": existing.id, "name": existing.name, "data_dir": existing.data_dir}
            resolved_corpus = existing.id
        if domain_profile == DEFAULT_DOMAIN_PROFILE:
            domain_profile = existing.domain_profile or domain_profile

    settings = resolve_corpus_settings(data_dir, resolved_corpus, qdrant_url=qdrant_url)
    ingest_results = ingest_url(
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
        skip_existing=skip_existing,
        transcribe_seconds=transcribe_seconds,
    )

    entity_result: dict[str, int] | None = None
    if rebuild:
        entity_result = rebuild_entities(settings.db_path, domain_profile=domain_profile)

    indexed_chunks: int | None = None
    if index:
        config = QdrantIndexConfig(
            collection_name=collection,
            dense_model=dense_model,
            sparse_model=sparse_model,
            url=settings.qdrant_url,
        )
        indexed_chunks = index_qdrant_chunks(
            settings.db_path,
            settings.qdrant_dir,
            config=config,
            batch_size=batch_size,
            force=force_index,
        )

    return {
        "corpus": resolved_corpus or "default",
        "data_dir": str(settings.data_dir),
        "created_corpus": created_corpus,
        "source_url": url,
        "ingest": [result.__dict__ for result in ingest_results],
        "entities": entity_result,
        "index": {"enabled": index, "collection": collection, "indexed_chunks": indexed_chunks},
        "ready": all(result.status in {"imported", "skipped"} for result in ingest_results) and (indexed_chunks is not None or not index),
    }
