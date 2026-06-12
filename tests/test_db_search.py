from podcast_rag.db import add_episode
from podcast_rag.models import TranscriptSegment
from podcast_rag.search import search_chunks


def test_add_episode_indexes_chunks_for_search(tmp_path):
    db_path = tmp_path / "rag.sqlite3"

    add_episode(
        db_path,
        title="Historia",
        segments=[
            TranscriptSegment("Felipe II aparece en esta parte.", start_seconds=1, end_seconds=5),
            TranscriptSegment("El Escorial tambien aparece.", start_seconds=5, end_seconds=9),
        ],
    )

    results = search_chunks(db_path, "Felipe")

    assert len(results) == 1
    assert results[0]["title"] == "Historia"
    assert results[0]["start_seconds"] == 1


def test_add_episode_is_idempotent_by_source_url(tmp_path):
    db_path = tmp_path / "rag.sqlite3"
    segments = [TranscriptSegment("Felipe II.", start_seconds=1, end_seconds=2)]

    first_id = add_episode(db_path, title="Uno", segments=segments, source_url="https://example.com/1")
    second_id = add_episode(db_path, title="Dos", segments=segments, source_url="https://example.com/1")

    assert second_id == first_id
