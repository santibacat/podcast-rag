from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from podcast_rag.db import connect, init_db

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class Embedder(Protocol):
    model_name: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass
class SentenceTransformerEmbedder:
    model_name: str = DEFAULT_EMBEDDING_MODEL

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "The SQLite embedding path requires the optional 'sentence-transformers' extra. "
                "Install it with: uv sync --extra sentence-transformers"
            ) from exc

        self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [list(map(float, vector)) for vector in vectors]


def build_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL) -> Embedder:
    return SentenceTransformerEmbedder(model_name=model_name)


def rebuild_chunk_embeddings(
    db_path: Path,
    embedder: Embedder,
    batch_size: int = 32,
    force: bool = False,
) -> int:
    init_db(db_path)
    indexed = 0
    with connect(db_path) as connection:
        if force:
            connection.execute("DELETE FROM chunk_embeddings WHERE model_name = ?", (embedder.model_name,))

        while True:
            rows = connection.execute(
                """
                SELECT transcript_chunks.id, transcript_chunks.text
                FROM transcript_chunks
                LEFT JOIN chunk_embeddings
                    ON chunk_embeddings.chunk_id = transcript_chunks.id
                    AND chunk_embeddings.model_name = ?
                WHERE chunk_embeddings.chunk_id IS NULL
                ORDER BY transcript_chunks.id
                LIMIT ?
                """,
                (embedder.model_name, batch_size),
            ).fetchall()
            if not rows:
                break

            vectors = embedder.embed([str(row["text"]) for row in rows])
            for row, vector in zip(rows, vectors, strict=True):
                connection.execute(
                    """
                    INSERT OR REPLACE INTO chunk_embeddings
                    (chunk_id, model_name, dimension, embedding)
                    VALUES (?, ?, ?, ?)
                    """,
                    (int(row["id"]), embedder.model_name, len(vector), pack_vector(vector)),
                )
                indexed += 1

    return indexed


def semantic_search(
    db_path: Path,
    query: str,
    embedder: Embedder,
    limit: int = 10,
) -> list[dict[str, object]]:
    init_db(db_path)
    query_vector = embedder.embed([query])[0]
    results: list[dict[str, object]] = []
    with connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                transcript_chunks.id AS chunk_id,
                transcript_chunks.episode_id,
                episodes.title,
                episodes.source_url,
                transcript_chunks.start_seconds,
                transcript_chunks.end_seconds,
                transcript_chunks.text,
                chunk_embeddings.embedding
            FROM chunk_embeddings
            JOIN transcript_chunks ON transcript_chunks.id = chunk_embeddings.chunk_id
            JOIN episodes ON episodes.id = transcript_chunks.episode_id
            WHERE chunk_embeddings.model_name = ?
            """,
            (embedder.model_name,),
        ).fetchall()

    for row in rows:
        vector = unpack_vector(row["embedding"])
        score = cosine_similarity(query_vector, vector)
        item = dict(row)
        item.pop("embedding", None)
        item["score"] = score
        results.append(item)

    results.sort(key=lambda item: float(item["score"]), reverse=True)
    return results[:limit]


def count_indexed_embeddings(db_path: Path, model_name: str) -> int:
    init_db(db_path)
    with connect(db_path) as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS count FROM chunk_embeddings WHERE model_name = ?",
            (model_name,),
        ).fetchone()
    return int(row["count"])


def pack_vector(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def unpack_vector(value: bytes) -> list[float]:
    if len(value) % 4 != 0:
        raise ValueError("Invalid float32 vector blob")
    return list(struct.unpack(f"<{len(value) // 4}f", value))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vectors must have the same dimension")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
