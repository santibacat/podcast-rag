from podcast_rag.db import add_episode
from podcast_rag.models import TranscriptSegment
from podcast_rag.search import entity_connections, list_topics, related_topics


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


def test_topics_include_entity_metadata_and_connections(tmp_path):
    db_path = tmp_path / "rag.sqlite3"
    add_episode(
        db_path,
        title="Historia",
        segments=[
            TranscriptSegment("El conquistador Francisco Pizarro viaja a Peru en 1533.", 1, 5),
            TranscriptSegment("El asesinato de Francisco Pizarro marca la guerra civil.", 5, 10),
        ],
    )

    topics = {row["name"]: row for row in list_topics(db_path, limit=20)}
    connections = entity_connections(db_path, name="Francisco Pizarro", limit=20)

    assert topics["Francisco Pizarro"]["entity_type"] == "PERSON"
    assert topics["Peru"]["entity_type"] == "PLACE"
    assert topics["1533"]["entity_type"] == "DATE"
    assert connections
