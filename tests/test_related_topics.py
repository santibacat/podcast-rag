from podcast_rag.db import add_episode
from podcast_rag.models import TranscriptSegment
from podcast_rag.search import related_topics


def test_related_topics_uses_shared_chunks(tmp_path):
    db_path = tmp_path / "rag.sqlite3"
    add_episode(
        db_path,
        title="Historia",
        segments=[
            TranscriptSegment("Felipe II visita El Escorial.", 1, 5),
            TranscriptSegment("La Armada Invencible aparece despues.", 5, 10),
        ],
    )

    rows = related_topics(db_path, "Felipe", limit=10)

    assert rows
    assert rows[0]["name"] == "El Escorial"
