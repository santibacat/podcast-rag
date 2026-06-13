from podcast_rag.dashboard import route_api
from podcast_rag.corpora import create_corpus
from podcast_rag.db import add_episode
from podcast_rag.models import TranscriptSegment


def test_dashboard_api_status_and_topics(tmp_path):
    add_episode(
        tmp_path / "podcast_rag.sqlite3",
        title="Pizarro",
        segments=[TranscriptSegment("Francisco Pizarro viaja a Peru.", 1, 5)],
        domain_profile="history_es",
    )

    status = route_api("/api/status", {}, tmp_path, None)
    topics = route_api("/api/topics", {"limit": ["10"]}, tmp_path, None)

    assert status["stats"]["counts"]["episodes"] == 1
    assert any(topic["name"] == "Francisco Pizarro" for topic in topics)


def test_dashboard_api_entity_profile(tmp_path):
    add_episode(
        tmp_path / "podcast_rag.sqlite3",
        title="Pizarro",
        segments=[TranscriptSegment("Francisco Pizarro viaja a Peru.", 1, 5)],
        domain_profile="history_es",
    )

    profile = route_api("/api/entity-profile", {"name": ["Francisco Pizarro"]}, tmp_path, None)

    assert profile["entity"]["name"] == "Francisco Pizarro"


def test_dashboard_api_lists_and_merges_registered_corpora(tmp_path):
    first = create_corpus(tmp_path, "historia", name="Historia")
    second = create_corpus(tmp_path, "arte", name="Arte")
    add_episode(
        tmp_path / "corpora" / first.id / "podcast_rag.sqlite3",
        title="Pizarro",
        segments=[TranscriptSegment("Francisco Pizarro viaja a Peru.", 1, 5)],
        domain_profile="history_es",
    )
    add_episode(
        tmp_path / "corpora" / second.id / "podcast_rag.sqlite3",
        title="Velazquez",
        segments=[TranscriptSegment("Diego Velazquez pinta en Madrid.", 1, 5)],
        domain_profile="history_es",
    )

    corpora = route_api("/api/corpora", {}, tmp_path, None)
    stats = route_api("/api/stats", {"corpus": ["all"]}, tmp_path, None)
    topics = route_api("/api/topics", {"corpus": ["all"], "limit": ["20"]}, tmp_path, None)

    assert [item["id"] for item in corpora["corpora"]] == ["arte", "historia"]
    assert stats["counts"]["episodes"] == 2
    assert {topic["name"] for topic in topics} >= {"Francisco Pizarro", "Diego Velazquez"}
