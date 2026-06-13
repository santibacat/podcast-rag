from __future__ import annotations

from pathlib import Path
from typing import Any

from podcast_rag.chunking import describe_chunking_strategy
from podcast_rag.db import connect, init_db
from podcast_rag.domain_profiles import DEFAULT_DOMAIN_PROFILE, describe_domain_profile, list_domain_profiles


def corpus_stats(db_path: Path) -> dict[str, Any]:
    init_db(db_path)
    with connect(db_path) as connection:
        counts = {
            "episodes": _count(connection, "episodes"),
            "segments": _count(connection, "transcript_segments"),
            "chunks": _count(connection, "transcript_chunks"),
            "entities": _count(connection, "entities"),
            "entity_mentions": _count(connection, "entity_mentions"),
            "entity_relations": _count(connection, "entity_relations"),
        }
        entity_types = [
            dict(row)
            for row in connection.execute(
                """
                SELECT entity_type, COUNT(*) AS count
                FROM entities
                GROUP BY entity_type
                ORDER BY count DESC, entity_type
                """
            ).fetchall()
        ]
        richest_episodes = [
            dict(row)
            for row in connection.execute(
                """
                SELECT
                    episodes.id AS episode_id,
                    episodes.title,
                    COUNT(DISTINCT entity_mentions.entity_id) AS unique_entities,
                    COUNT(entity_mentions.id) AS mentions,
                    COUNT(DISTINCT transcript_chunks.id) AS chunks
                FROM episodes
                LEFT JOIN transcript_chunks ON transcript_chunks.episode_id = episodes.id
                LEFT JOIN entity_mentions ON entity_mentions.episode_id = episodes.id
                GROUP BY episodes.id
                ORDER BY unique_entities DESC, mentions DESC
                LIMIT 10
                """
            ).fetchall()
        ]
        avg_entities_per_episode = (
            sum(int(row["unique_entities"]) for row in richest_episodes) / counts["episodes"] if counts["episodes"] else 0
        )

    return {
        "counts": counts,
        "entity_types": entity_types,
        "richest_episodes": richest_episodes,
        "avg_entities_per_episode": avg_entities_per_episode,
    }


def entity_profile(db_path: Path, name: str, limit: int = 10) -> dict[str, Any]:
    init_db(db_path)
    with connect(db_path) as connection:
        entity = connection.execute("SELECT * FROM entities WHERE name = ?", (name,)).fetchone()
        if entity is None:
            raise LookupError(f"Entity {name!r} does not exist")

        mentions = [
            dict(row)
            for row in connection.execute(
                """
                SELECT
                    entity_mentions.count,
                    episodes.id AS episode_id,
                    episodes.title,
                    episodes.source_url,
                    transcript_chunks.id AS chunk_id,
                    transcript_chunks.start_seconds,
                    transcript_chunks.end_seconds,
                    transcript_chunks.text
                FROM entity_mentions
                JOIN episodes ON episodes.id = entity_mentions.episode_id
                LEFT JOIN transcript_chunks ON transcript_chunks.id = entity_mentions.chunk_id
                WHERE entity_mentions.entity_id = ?
                ORDER BY episodes.id, transcript_chunks.start_seconds
                LIMIT ?
                """,
                (int(entity["id"]), limit),
            ).fetchall()
        ]
        episodes = [
            dict(row)
            for row in connection.execute(
                """
                SELECT episodes.id AS episode_id, episodes.title, SUM(entity_mentions.count) AS mentions
                FROM entity_mentions
                JOIN episodes ON episodes.id = entity_mentions.episode_id
                WHERE entity_mentions.entity_id = ?
                GROUP BY episodes.id
                ORDER BY mentions DESC, episodes.id
                """,
                (int(entity["id"]),),
            ).fetchall()
        ]
        connections = _connections_for_entity(connection, int(entity["id"]), limit=limit)

    return {
        "entity": dict(entity),
        "episodes": episodes,
        "mentions": mentions,
        "connections": connections,
    }


def entity_timeline(db_path: Path, topic: str | None = None, episode_id: int | None = None, limit: int = 200) -> list[dict[str, Any]]:
    init_db(db_path)
    params: list[Any] = []
    where_clauses = []
    if topic:
        where_clauses.append("entities.name = ?")
        params.append(topic)
    if episode_id is not None:
        where_clauses.append("episodes.id = ?")
        params.append(episode_id)
    where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    params.append(limit)

    with connect(db_path) as connection:
        rows = connection.execute(
            f"""
            SELECT
                episodes.id AS episode_id,
                episodes.title,
                entities.name,
                entities.entity_type,
                entity_mentions.count,
                transcript_chunks.id AS chunk_id,
                transcript_chunks.start_seconds,
                transcript_chunks.end_seconds,
                transcript_chunks.text
            FROM entity_mentions
            JOIN entities ON entities.id = entity_mentions.entity_id
            JOIN episodes ON episodes.id = entity_mentions.episode_id
            LEFT JOIN transcript_chunks ON transcript_chunks.id = entity_mentions.chunk_id
            {where}
            ORDER BY episodes.id, transcript_chunks.start_seconds, entities.name
            LIMIT ?
            """,
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def episode_insights(db_path: Path, episode_id: int, limit: int = 20) -> dict[str, Any]:
    init_db(db_path)
    with connect(db_path) as connection:
        episode = connection.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,)).fetchone()
        if episode is None:
            raise LookupError(f"Episode {episode_id} does not exist")

        top_entities = [
            dict(row)
            for row in connection.execute(
                """
                SELECT
                    entities.name,
                    entities.entity_type,
                    SUM(entity_mentions.count) AS mentions,
                    COUNT(DISTINCT entity_mentions.chunk_id) AS chunks
                FROM entity_mentions
                JOIN entities ON entities.id = entity_mentions.entity_id
                WHERE entity_mentions.episode_id = ?
                GROUP BY entities.id
                ORDER BY mentions DESC, chunks DESC, entities.name
                LIMIT ?
                """,
                (episode_id, limit),
            ).fetchall()
        ]
        entity_density = [
            dict(row)
            for row in connection.execute(
                """
                SELECT
                    transcript_chunks.id AS chunk_id,
                    transcript_chunks.start_seconds,
                    transcript_chunks.end_seconds,
                    COUNT(DISTINCT entity_mentions.entity_id) AS unique_entities,
                    transcript_chunks.text
                FROM transcript_chunks
                LEFT JOIN entity_mentions ON entity_mentions.chunk_id = transcript_chunks.id
                WHERE transcript_chunks.episode_id = ?
                GROUP BY transcript_chunks.id
                ORDER BY unique_entities DESC, transcript_chunks.start_seconds
                LIMIT ?
                """,
                (episode_id, limit),
            ).fetchall()
        ]
        connections = [
            dict(row)
            for row in connection.execute(
                """
                SELECT
                    source_entities.name AS source,
                    target_entities.name AS target,
                    source_entities.entity_type AS source_type,
                    target_entities.entity_type AS target_type,
                    entity_relations.weight,
                    entity_relations.shared_chunks
                FROM entity_relations
                JOIN entities AS source_entities ON source_entities.id = entity_relations.source_entity_id
                JOIN entities AS target_entities ON target_entities.id = entity_relations.target_entity_id
                WHERE entity_relations.source_entity_id IN (
                    SELECT entity_id FROM entity_mentions WHERE episode_id = ?
                )
                OR entity_relations.target_entity_id IN (
                    SELECT entity_id FROM entity_mentions WHERE episode_id = ?
                )
                ORDER BY entity_relations.weight DESC
                LIMIT ?
                """,
                (episode_id, episode_id, limit),
            ).fetchall()
        ]

    return {
        "episode": dict(episode),
        "top_entities": top_entities,
        "entity_density": entity_density,
        "connections": connections,
    }


def topic_episode_matrix(db_path: Path, limit_entities: int = 50) -> dict[str, Any]:
    init_db(db_path)
    with connect(db_path) as connection:
        episodes = [dict(row) for row in connection.execute("SELECT id, title FROM episodes ORDER BY id").fetchall()]
        entities = [
            dict(row)
            for row in connection.execute(
                """
                SELECT entities.id, entities.name, entities.entity_type, SUM(entity_mentions.count) AS mentions
                FROM entities
                JOIN entity_mentions ON entity_mentions.entity_id = entities.id
                GROUP BY entities.id
                ORDER BY mentions DESC
                LIMIT ?
                """,
                (limit_entities,),
            ).fetchall()
        ]
        rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT entity_id, episode_id, SUM(count) AS mentions
                FROM entity_mentions
                GROUP BY entity_id, episode_id
                """
            ).fetchall()
        ]
    entity_ids = {int(entity["id"]) for entity in entities}
    cells = [row for row in rows if int(row["entity_id"]) in entity_ids]
    return {"episodes": episodes, "entities": entities, "cells": cells}


def graph_export(db_path: Path, min_weight: float = 0.0, limit: int = 1000) -> dict[str, Any]:
    init_db(db_path)
    with connect(db_path) as connection:
        nodes = [
            dict(row)
            for row in connection.execute(
                """
                SELECT
                    entities.id,
                    entities.name,
                    entities.entity_type,
                    entities.confidence,
                    COALESCE(SUM(entity_mentions.count), 0) AS mentions
                FROM entities
                LEFT JOIN entity_mentions ON entity_mentions.entity_id = entities.id
                GROUP BY entities.id
                """
            ).fetchall()
        ]
        edges = [
            dict(row)
            for row in connection.execute(
                """
                SELECT
                    source_entity_id AS source,
                    target_entity_id AS target,
                    relation_type,
                    weight,
                    shared_chunks,
                    shared_episodes,
                    evidence
                FROM entity_relations
                WHERE weight >= ?
                ORDER BY weight DESC
                LIMIT ?
                """,
                (min_weight, limit),
            ).fetchall()
        ]
    return {"nodes": nodes, "edges": edges}


def quality_report(db_path: Path) -> dict[str, Any]:
    init_db(db_path)
    with connect(db_path) as connection:
        low_confidence_entities = [
            dict(row)
            for row in connection.execute(
                """
                SELECT name, entity_type, confidence, evidence
                FROM entities
                WHERE confidence < 0.6 OR entity_type = 'UNKNOWN'
                ORDER BY confidence ASC, name
                LIMIT 50
                """
            ).fetchall()
        ]
        missing_timestamps = [
            dict(row)
            for row in connection.execute(
                """
                SELECT episodes.id AS episode_id, episodes.title, COUNT(*) AS segments
                FROM transcript_segments
                JOIN episodes ON episodes.id = transcript_segments.episode_id
                WHERE transcript_segments.start_seconds IS NULL
                GROUP BY episodes.id
                ORDER BY segments DESC
                """
            ).fetchall()
        ]
        empty_or_short_chunks = [
            dict(row)
            for row in connection.execute(
                """
                SELECT id AS chunk_id, episode_id, start_seconds, text
                FROM transcript_chunks
                WHERE LENGTH(TRIM(text)) < 40
                ORDER BY episode_id, id
                LIMIT 50
                """
            ).fetchall()
        ]
        chunks_without_entities = [
            dict(row)
            for row in connection.execute(
                """
                SELECT transcript_chunks.id AS chunk_id, transcript_chunks.episode_id, transcript_chunks.start_seconds
                FROM transcript_chunks
                LEFT JOIN entity_mentions ON entity_mentions.chunk_id = transcript_chunks.id
                WHERE entity_mentions.id IS NULL
                LIMIT 50
                """
            ).fetchall()
        ]
    return {
        "low_confidence_entities": low_confidence_entities,
        "missing_timestamps": missing_timestamps,
        "empty_or_short_chunks": empty_or_short_chunks,
        "chunks_without_entities": chunks_without_entities,
    }


def system_status(db_path: Path, data_dir: Path, qdrant_url: str | None = None) -> dict[str, Any]:
    stats = corpus_stats(db_path)
    quality = quality_report(db_path)
    qdrant_local_path = data_dir / "qdrant"
    has_qdrant_local_storage = qdrant_local_path.exists()
    quality_counts = {key: len(value) for key, value in quality.items()}
    recommendations = build_recommendations(stats, quality_counts, has_qdrant_local_storage, qdrant_url)
    return {
        "data_dir": str(data_dir),
        "sqlite_db": str(db_path),
        "qdrant": {
            "mode": "server" if qdrant_url else "local",
            "url": qdrant_url,
            "local_path": str(qdrant_local_path),
            "local_storage_exists": has_qdrant_local_storage,
        },
        "chunking": describe_chunking_strategy(),
        "domain_profiles": {
            "default": DEFAULT_DOMAIN_PROFILE,
            "available": list_domain_profiles(),
        },
        "stats": stats,
        "quality_counts": quality_counts,
        "recommendations": recommendations,
    }


def build_recommendations(
    stats: dict[str, Any],
    quality_counts: dict[str, int],
    has_qdrant_local_storage: bool,
    qdrant_url: str | None,
) -> list[str]:
    recommendations: list[str] = []
    counts = stats["counts"]
    if counts["episodes"] == 0:
        recommendations.append("Ingest transcripts or URLs before building retrieval indexes.")
    if counts["chunks"] > 0 and counts["entities"] == 0:
        recommendations.append("Run rebuild-entities with an appropriate domain profile.")
    if counts["chunks"] > 0 and not has_qdrant_local_storage and not qdrant_url:
        recommendations.append("Run index-retrieval before hybrid-search, retrieve, or ask.")
    if quality_counts.get("low_confidence_entities", 0):
        recommendations.append("Inspect low-confidence entities and consider another domain profile.")
    if quality_counts.get("missing_timestamps", 0):
        recommendations.append("Some segments lack timestamps; timestamp-based navigation will be weaker.")
    if not recommendations:
        recommendations.append("Corpus looks ready for dashboard exploration and agentic retrieval.")
    return recommendations


def profile_explanation(name: str | None = None) -> dict[str, object]:
    return describe_domain_profile(name)


def _count(connection: Any, table: str) -> int:
    row = connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
    return int(row["count"])


def _connections_for_entity(connection: Any, entity_id: int, limit: int) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in connection.execute(
            """
            SELECT
                source_entities.name AS source,
                source_entities.entity_type AS source_type,
                target_entities.name AS target,
                target_entities.entity_type AS target_type,
                entity_relations.relation_type,
                entity_relations.weight,
                entity_relations.shared_chunks,
                entity_relations.shared_episodes,
                entity_relations.evidence
            FROM entity_relations
            JOIN entities AS source_entities ON source_entities.id = entity_relations.source_entity_id
            JOIN entities AS target_entities ON target_entities.id = entity_relations.target_entity_id
            WHERE entity_relations.source_entity_id = ? OR entity_relations.target_entity_id = ?
            ORDER BY entity_relations.weight DESC, entity_relations.shared_chunks DESC
            LIMIT ?
            """,
            (entity_id, entity_id, limit),
        ).fetchall()
    ]
