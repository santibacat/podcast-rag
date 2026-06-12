from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from podcast_rag.db import connect, init_db

DEFAULT_QDRANT_COLLECTION = "podcast_chunks"
DEFAULT_QDRANT_DENSE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_QDRANT_SPARSE_MODEL = "Qdrant/bm25"


@dataclass(frozen=True)
class QdrantIndexConfig:
    collection_name: str = DEFAULT_QDRANT_COLLECTION
    dense_model: str = DEFAULT_QDRANT_DENSE_MODEL
    sparse_model: str = DEFAULT_QDRANT_SPARSE_MODEL


def build_qdrant_client(qdrant_dir: Path, config: QdrantIndexConfig):
    from qdrant_client import QdrantClient

    qdrant_dir.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(qdrant_dir))
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
        limit=limit,
        with_payload=True,
    )
    results = [scored_point_to_result(point) for point in response.points]
    client.close()
    return results


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
                transcript_chunks.text
            FROM transcript_chunks
            JOIN episodes ON episodes.id = transcript_chunks.episode_id
            ORDER BY transcript_chunks.id
            """
        ).fetchall()
    return [dict(row) for row in rows]


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
    }


def scored_point_to_result(point: Any) -> dict[str, Any]:
    payload = dict(point.payload or {})
    payload["score"] = float(point.score)
    payload["point_id"] = point.id
    return payload


def batched(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [rows[index : index + batch_size] for index in range(0, len(rows), batch_size)]
