from podcast_rag.analytics import profile_explanation, system_status
from podcast_rag.chunking import describe_chunking_strategy
from podcast_rag.db import add_episode
from podcast_rag.models import TranscriptSegment


def test_describe_chunking_strategy_is_explicit():
    info = describe_chunking_strategy()

    assert info["max_words"] == 180
    assert info["overlap_words"] == 35
    assert "timestamp" in info["timestamp_policy"]


def test_profile_explanation_returns_profile_rules():
    profile = profile_explanation("history_es")

    assert profile["name"] == "history_es"
    assert "conquistador" in profile["person_hints"]


def test_system_status_recommends_indexing_when_chunks_exist_without_qdrant(tmp_path):
    db_path = tmp_path / "podcast_rag.sqlite3"
    add_episode(
        db_path,
        title="Episode",
        segments=[TranscriptSegment("Francisco Pizarro viaja a Peru.", 1, 5)],
        domain_profile="history_es",
    )

    status = system_status(db_path, tmp_path)

    assert status["stats"]["counts"]["episodes"] == 1
    assert any("index-retrieval" in recommendation for recommendation in status["recommendations"])
