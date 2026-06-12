from types import SimpleNamespace

import pytest

from podcast_rag.db import add_episode
from podcast_rag.models import TranscriptSegment
from podcast_rag.retrieval_qdrant import (
    batched,
    build_qdrant_payload,
    list_transcript_chunks,
    scored_point_to_result,
)


def test_list_transcript_chunks_returns_payload_source_rows(tmp_path):
    db_path = tmp_path / "rag.sqlite3"
    add_episode(
        db_path,
        title="Historia",
        segments=[TranscriptSegment("Felipe II y El Escorial.", 1, 5)],
        source_url="https://example.com/episode",
        author="Autor",
        language="es",
    )

    rows = list_transcript_chunks(db_path)

    assert rows[0]["chunk_id"] == 1
    assert rows[0]["episode_id"] == 1
    assert rows[0]["title"] == "Historia"
    assert rows[0]["source_url"] == "https://example.com/episode"
    assert rows[0]["language"] == "es"


def test_build_qdrant_payload_preserves_retrieval_metadata():
    payload = build_qdrant_payload(
        {
            "chunk_id": 7,
            "episode_id": 3,
            "title": "Episode",
            "source_url": "https://example.com",
            "author": "Author",
            "language": "es",
            "start_seconds": 10.0,
            "end_seconds": 20.0,
            "text": "Transcript chunk",
        }
    )

    assert payload["chunk_id"] == 7
    assert payload["episode_id"] == 3
    assert payload["text"] == "Transcript chunk"
    assert payload["start_seconds"] == 10.0


def test_scored_point_to_result_flattens_payload_and_score():
    result = scored_point_to_result(SimpleNamespace(id=42, score=0.75, payload={"text": "chunk"}))

    assert result == {"text": "chunk", "score": 0.75, "point_id": 42}


def test_batched_validates_batch_size():
    assert batched([{"id": 1}, {"id": 2}, {"id": 3}], 2) == [[{"id": 1}, {"id": 2}], [{"id": 3}]]
    with pytest.raises(ValueError):
        batched([], 0)
