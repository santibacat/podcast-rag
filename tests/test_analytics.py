from podcast_rag.analytics import (
    corpus_stats,
    entity_profile,
    entity_timeline,
    episode_insights,
    graph_export,
    quality_report,
    topic_episode_matrix,
)
from podcast_rag.db import add_episode
from podcast_rag.models import TranscriptSegment


def _build_analytics_fixture(db_path):
    add_episode(
        db_path,
        title="Pizarro",
        segments=[
            TranscriptSegment("El conquistador Francisco Pizarro viaja a Peru en 1533.", 1, 5),
            TranscriptSegment("Francisco Pizarro y La Corona aparecen en Lima.", 5, 10),
        ],
        language="es",
        domain_profile="history_es",
    )


def test_corpus_stats_reports_counts_and_types(tmp_path):
    db_path = tmp_path / "rag.sqlite3"
    _build_analytics_fixture(db_path)

    stats = corpus_stats(db_path)

    assert stats["counts"]["episodes"] == 1
    assert stats["counts"]["entities"] >= 4
    assert any(row["entity_type"] == "PERSON" for row in stats["entity_types"])


def test_entity_profile_contains_mentions_and_connections(tmp_path):
    db_path = tmp_path / "rag.sqlite3"
    _build_analytics_fixture(db_path)

    profile = entity_profile(db_path, "Francisco Pizarro")

    assert profile["entity"]["entity_type"] == "PERSON"
    assert profile["mentions"]
    assert profile["connections"]


def test_entity_timeline_can_filter_topic(tmp_path):
    db_path = tmp_path / "rag.sqlite3"
    _build_analytics_fixture(db_path)

    rows = entity_timeline(db_path, topic="Francisco Pizarro")

    assert rows
    assert {row["name"] for row in rows} == {"Francisco Pizarro"}


def test_episode_insights_returns_density_and_top_entities(tmp_path):
    db_path = tmp_path / "rag.sqlite3"
    _build_analytics_fixture(db_path)

    insights = episode_insights(db_path, episode_id=1)

    assert insights["episode"]["title"] == "Pizarro"
    assert insights["top_entities"]
    assert insights["entity_density"]


def test_topic_episode_matrix_and_graph_export(tmp_path):
    db_path = tmp_path / "rag.sqlite3"
    _build_analytics_fixture(db_path)

    matrix = topic_episode_matrix(db_path)
    graph = graph_export(db_path)

    assert matrix["episodes"]
    assert matrix["entities"]
    assert matrix["cells"]
    assert graph["nodes"]
    assert graph["edges"]


def test_quality_report_shape(tmp_path):
    db_path = tmp_path / "rag.sqlite3"
    _build_analytics_fixture(db_path)

    report = quality_report(db_path)

    assert set(report) == {
        "low_confidence_entities",
        "missing_timestamps",
        "empty_or_short_chunks",
        "chunks_without_entities",
    }
