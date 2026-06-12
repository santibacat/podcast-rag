from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from podcast_rag.db import connect, init_db
from podcast_rag.search import get_chunk_context

DEFAULT_QDRANT_COLLECTION = "podcast_chunks"
DEFAULT_QDRANT_DENSE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_QDRANT_SPARSE_MODEL = "Qdrant/bm25"


@dataclass(frozen=True)
class QdrantIndexConfig:
    collection_name: str = DEFAULT_QDRANT_COLLECTION
    dense_model: str = DEFAULT_QDRANT_DENSE_MODEL
    sparse_model: str = DEFAULT_QDRANT_SPARSE_MODEL
    url: str | None = None


def build_qdrant_client(qdrant_dir: Path, config: QdrantIndexConfig):
    from qdrant_client import QdrantClient

    if config.url:
        client = QdrantClient(url=config.url)
    else:
        qdrant_dir.mkdir(parents=True, exist_ok=True)
        try:
            client = QdrantClient(path=str(qdrant_dir))
        except RuntimeError as exc:
            if "already accessed by another instance" in str(exc):
                raise RuntimeError(
                    "Local Qdrant storage is already in use. Run Qdrant-backed commands sequentially, "
                    "set QDRANT_URL, or pass --qdrant-url to use a Qdrant server for concurrent access."
                ) from exc
            raise
    client.set_model(config.dense_model)
    client.set_sparse_model(config.sparse_model)
    return client


def ensure_qdrant_collection(client: Any, config: QdrantIndexConfig, force: bool = False) -> None:
    if force and client.collection_exists(config.collection_name):
        client.delete_collection(config.collection_name)

    if client.collection_exists(config.collection_name):
        return

    client.create_collection(
        collection_name=config.collection_name,
        vectors_config=client.get_fastembed_vector_params(),
        sparse_vectors_config=client.get_fastembed_sparse_vector_params(),
    )


def index_qdrant_chunks(
    db_path: Path,
    qdrant_dir: Path,
    config: QdrantIndexConfig = QdrantIndexConfig(),
    batch_size: int = 64,
    force: bool = False,
) -> int:
    client = build_qdrant_client(qdrant_dir, config)
    ensure_qdrant_collection(client, config, force=force)
    rows = list_transcript_chunks(db_path)
    dense_field = client.get_vector_field_name()
    sparse_field = client.get_sparse_vector_field_name()

    indexed = 0
    for batch in batched(rows, batch_size):
        client.upload_collection(
            collection_name=config.collection_name,
            vectors=[
                build_qdrant_vectors(
                    text=str(row["text"]),
                    dense_field=dense_field,
                    sparse_field=sparse_field,
                    dense_model=config.dense_model,
                    sparse_model=config.sparse_model,
                )
                for row in batch
            ],
            payload=[build_qdrant_payload(row) for row in batch],
            ids=[int(row["chunk_id"]) for row in batch],
            batch_size=batch_size,
            wait=True,
        )
        indexed += len(batch)

    client.close()
    return indexed


def qdrant_hybrid_search(
    query: str,
    qdrant_dir: Path,
    config: QdrantIndexConfig = QdrantIndexConfig(),
    limit: int = 10,
    prefetch_limit: int = 40,
    episode_id: int | None = None,
    topic: str | None = None,
) -> list[dict[str, Any]]:
    from qdrant_client import models

    client = build_qdrant_client(qdrant_dir, config)
    if not client.collection_exists(config.collection_name):
        client.close()
        return []

    dense_field = client.get_vector_field_name()
    sparse_field = client.get_sparse_vector_field_name()
    response = client.query_points(
        collection_name=config.collection_name,
        prefetch=[
            models.Prefetch(
                query=models.Document(text=query, model=config.dense_model),
                using=dense_field,
                limit=prefetch_limit,
            ),
            models.Prefetch(
                query=models.Document(text=query, model=config.sparse_model),
                using=sparse_field,
                limit=prefetch_limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=build_qdrant_filter(episode_id=episode_id, topic=topic),
        limit=limit,
        with_payload=True,
    )
    results = [scored_point_to_result(point) for point in response.points]
    client.close()
    return results


def retrieve_evidence(
    query: str,
    db_path: Path,
    qdrant_dir: Path,
    config: QdrantIndexConfig = QdrantIndexConfig(),
    limit: int = 5,
    prefetch_limit: int = 40,
    before_segments: int = 2,
    after_segments: int = 2,
    episode_id: int | None = None,
    topic: str | None = None,
) -> list[dict[str, Any]]:
    results = qdrant_hybrid_search(
        query=query,
        qdrant_dir=qdrant_dir,
        config=config,
        limit=limit,
        prefetch_limit=prefetch_limit,
        episode_id=episode_id,
        topic=topic,
    )
    evidence: list[dict[str, Any]] = []
    for result in results:
        context = get_chunk_context(
            db_path,
            int(result["chunk_id"]),
            before_segments=before_segments,
            after_segments=after_segments,
        )
        enriched = dict(result)
        enriched["context_text"] = context["context_text"]
        enriched["context_segments"] = context["segments"]
        evidence.append(enriched)
    return evidence


def list_transcript_chunks(db_path: Path) -> list[dict[str, Any]]:
    init_db(db_path)
    with connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                transcript_chunks.id AS chunk_id,
                transcript_chunks.episode_id,
                episodes.title,
                episodes.source_url,
                episodes.author,
                episodes.language,
                transcript_chunks.start_seconds,
                transcript_chunks.end_seconds,
                transcript_chunks.text,
                GROUP_CONCAT(DISTINCT entities.name) AS entities,
                GROUP_CONCAT(DISTINCT entities.entity_type) AS entity_types
            FROM transcript_chunks
            JOIN episodes ON episodes.id = transcript_chunks.episode_id
            LEFT JOIN entity_mentions ON entity_mentions.chunk_id = transcript_chunks.id
            LEFT JOIN entities ON entities.id = entity_mentions.entity_id
            GROUP BY transcript_chunks.id
            ORDER BY transcript_chunks.id
            """
        ).fetchall()
    return [_normalize_chunk_row(dict(row)) for row in rows]


def build_qdrant_vectors(
    text: str,
    dense_field: str,
    sparse_field: str,
    dense_model: str,
    sparse_model: str,
) -> dict[str, Any]:
    from qdrant_client import models

    return {
        dense_field: models.Document(text=text, model=dense_model),
        sparse_field: models.Document(text=text, model=sparse_model),
    }


def build_qdrant_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": int(row["chunk_id"]),
        "episode_id": int(row["episode_id"]),
        "title": row["title"],
        "source_url": row["source_url"],
        "author": row["author"],
        "language": row["language"],
        "start_seconds": row["start_seconds"],
        "end_seconds": row["end_seconds"],
        "text": row["text"],
        "entities": row.get("entities", []),
        "entity_types": row.get("entity_types", []),
    }


def build_qdrant_filter(episode_id: int | None = None, topic: str | None = None) -> Any | None:
    if episode_id is None and topic is None:
        return None

    from qdrant_client import models

    must: list[Any] = []
    if episode_id is not None:
        must.append(models.FieldCondition(key="episode_id", match=models.MatchValue(value=episode_id)))
    if topic:
        must.append(models.FieldCondition(key="entities", match=models.MatchValue(value=topic)))
    return models.Filter(must=must)


def scored_point_to_result(point: Any) -> dict[str, Any]:
    payload = dict(point.payload or {})
    payload["score"] = float(point.score)
    payload["point_id"] = point.id
    return payload


def _normalize_chunk_row(row: dict[str, Any]) -> dict[str, Any]:
    entities = row.get("entities")
    if isinstance(entities, str) and entities:
        row["entities"] = sorted(item for item in entities.split(",") if item)
    else:
        row["entities"] = []
    entity_types = row.get("entity_types")
    if isinstance(entity_types, str) and entity_types:
        row["entity_types"] = sorted(item for item in entity_types.split(",") if item)
    else:
        row["entity_types"] = []
    return row


def batched(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [rows[index : index + batch_size] for index in range(0, len(rows), batch_size)]
