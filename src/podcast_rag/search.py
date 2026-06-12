from __future__ import annotations

from pathlib import Path

from podcast_rag.db import connect, init_db


def search_chunks(db_path: Path, query: str, limit: int = 10) -> list[dict[str, object]]:
    init_db(db_path)
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
                snippet(transcript_chunks_fts, 0, '[', ']', '...', 24) AS snippet,
                transcript_chunks.text
            FROM transcript_chunks_fts
            JOIN transcript_chunks ON transcript_chunks.id = transcript_chunks_fts.rowid
            JOIN episodes ON episodes.id = transcript_chunks.episode_id
            WHERE transcript_chunks_fts MATCH ?
            ORDER BY bm25(transcript_chunks_fts)
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def list_episodes(db_path: Path) -> list[dict[str, object]]:
    init_db(db_path)
    with connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                episodes.id,
                episodes.title,
                episodes.source_url,
                episodes.author,
                episodes.language,
                episodes.created_at,
                COUNT(transcript_segments.id) AS segment_count
            FROM episodes
            LEFT JOIN transcript_segments ON transcript_segments.episode_id = episodes.id
            GROUP BY episodes.id
            ORDER BY episodes.created_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_episode_segments(db_path: Path, episode_id: int) -> tuple[dict[str, object], list[dict[str, object]]]:
    init_db(db_path)
    with connect(db_path) as connection:
        episode = connection.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,)).fetchone()
        if episode is None:
            raise LookupError(f"Episode {episode_id} does not exist")
        rows = connection.execute(
            """
            SELECT position, start_seconds, end_seconds, text
            FROM transcript_segments
            WHERE episode_id = ?
            ORDER BY position
            """,
            (episode_id,),
        ).fetchall()
    return dict(episode), [dict(row) for row in rows]


def list_topics(db_path: Path, limit: int = 50) -> list[dict[str, object]]:
    init_db(db_path)
    with connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT entities.name, SUM(entity_mentions.count) AS mentions, COUNT(DISTINCT episode_id) AS episodes
            FROM entities
            JOIN entity_mentions ON entity_mentions.entity_id = entities.id
            GROUP BY entities.id
            ORDER BY mentions DESC, entities.name
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def related_topics(db_path: Path, name: str, limit: int = 25) -> list[dict[str, object]]:
    init_db(db_path)
    with connect(db_path) as connection:
        rows = connection.execute(
            """
            WITH target AS (
                SELECT id FROM entities WHERE name = ?
            ),
            target_chunks AS (
                SELECT DISTINCT chunk_id
                FROM entity_mentions
                WHERE entity_id = (SELECT id FROM target)
            )
            SELECT
                entities.name,
                SUM(entity_mentions.count) AS mentions,
                COUNT(DISTINCT entity_mentions.chunk_id) AS shared_chunks,
                COUNT(DISTINCT entity_mentions.episode_id) AS episodes
            FROM entity_mentions
            JOIN entities ON entities.id = entity_mentions.entity_id
            WHERE entity_mentions.chunk_id IN (SELECT chunk_id FROM target_chunks)
              AND entities.name != ?
            GROUP BY entities.id
            ORDER BY shared_chunks DESC, mentions DESC, entities.name
            LIMIT ?
            """,
            (name, name, limit),
        ).fetchall()
    return [dict(row) for row in rows]
