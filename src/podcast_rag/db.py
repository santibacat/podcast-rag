from __future__ import annotations

import sqlite3
from pathlib import Path

from podcast_rag.chunking import chunk_segments
from podcast_rag.entities import extract_candidate_entities
from podcast_rag.models import TranscriptSegment

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    source_url TEXT,
    author TEXT,
    language TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_episodes_source_url
ON episodes(source_url)
WHERE source_url IS NOT NULL;

CREATE TABLE IF NOT EXISTS transcript_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    start_seconds REAL,
    end_seconds REAL,
    text TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transcript_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    segment_start INTEGER NOT NULL,
    segment_end INTEGER NOT NULL,
    start_seconds REAL,
    end_seconds REAL,
    text TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS transcript_chunks_fts USING fts5(text, title);

CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    entity_type TEXT NOT NULL DEFAULT 'UNKNOWN',
    confidence REAL NOT NULL DEFAULT 0.0,
    evidence TEXT
);

CREATE TABLE IF NOT EXISTS entity_mentions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    episode_id INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    chunk_id INTEGER REFERENCES transcript_chunks(id) ON DELETE CASCADE,
    count INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id INTEGER NOT NULL REFERENCES transcript_chunks(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (chunk_id, model_name)
);

CREATE TABLE IF NOT EXISTS entity_relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL DEFAULT 'CO_OCCURS',
    weight REAL NOT NULL DEFAULT 1.0,
    shared_chunks INTEGER NOT NULL DEFAULT 0,
    shared_episodes INTEGER NOT NULL DEFAULT 0,
    evidence TEXT,
    UNIQUE(source_entity_id, target_entity_id, relation_type)
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db(db_path: Path) -> None:
    with connect(db_path) as connection:
        connection.executescript(SCHEMA)
        migrate_schema(connection)


def add_episode(
    db_path: Path,
    title: str,
    segments: list[TranscriptSegment],
    source_url: str | None = None,
    author: str | None = None,
    language: str | None = None,
    domain_profile: str | None = None,
) -> int:
    init_db(db_path)
    with connect(db_path) as connection:
        if source_url:
            existing = connection.execute(
                "SELECT id FROM episodes WHERE source_url = ?",
                (source_url,),
            ).fetchone()
            if existing is not None:
                return int(existing["id"])

        cursor = connection.execute(
            "INSERT INTO episodes (title, source_url, author, language) VALUES (?, ?, ?, ?)",
            (title, source_url, author, language),
        )
        episode_id = int(cursor.lastrowid)

        connection.executemany(
            """
            INSERT INTO transcript_segments
            (episode_id, position, start_seconds, end_seconds, text)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (episode_id, index, segment.start_seconds, segment.end_seconds, segment.text)
                for index, segment in enumerate(segments)
            ],
        )

        chunks = chunk_segments(segments)
        chunk_rows = []
        for chunk in chunks:
            cursor = connection.execute(
                """
                INSERT INTO transcript_chunks
                (episode_id, segment_start, segment_end, start_seconds, end_seconds, text)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    episode_id,
                    chunk.segment_start,
                    chunk.segment_end,
                    chunk.start_seconds,
                    chunk.end_seconds,
                    chunk.text,
                ),
            )
            chunk_id = int(cursor.lastrowid)
            chunk_rows.append((chunk_id, chunk.text))
            connection.execute(
                "INSERT INTO transcript_chunks_fts (rowid, text, title) VALUES (?, ?, ?)",
                (chunk_id, chunk.text, title),
            )

        for chunk_id, text in chunk_rows:
            for candidate in extract_candidate_entities(text, domain_profile=domain_profile):
                entity_id = upsert_entity(
                    connection,
                    candidate.name,
                    entity_type=candidate.entity_type,
                    confidence=candidate.confidence,
                    evidence=candidate.evidence,
                )
                connection.execute(
                    """
                    INSERT INTO entity_mentions (entity_id, episode_id, chunk_id, count)
                    VALUES (?, ?, ?, ?)
                    """,
                    (entity_id, episode_id, chunk_id, candidate.count),
                )
        rebuild_entity_relations_for_episode(connection, episode_id)

    return episode_id


def rebuild_entities(db_path: Path, domain_profile: str | None = None) -> dict[str, int]:
    init_db(db_path)
    with connect(db_path) as connection:
        connection.execute("DELETE FROM entity_relations")
        connection.execute("DELETE FROM entity_mentions")
        connection.execute("DELETE FROM entities")
        rows = connection.execute(
            """
            SELECT id, episode_id, text
            FROM transcript_chunks
            ORDER BY episode_id, id
            """
        ).fetchall()
        mentions = 0
        for row in rows:
            for candidate in extract_candidate_entities(str(row["text"]), domain_profile=domain_profile):
                entity_id = upsert_entity(
                    connection,
                    candidate.name,
                    entity_type=candidate.entity_type,
                    confidence=candidate.confidence,
                    evidence=candidate.evidence,
                )
                connection.execute(
                    """
                    INSERT INTO entity_mentions (entity_id, episode_id, chunk_id, count)
                    VALUES (?, ?, ?, ?)
                    """,
                    (entity_id, int(row["episode_id"]), int(row["id"]), candidate.count),
                )
                mentions += 1
        episode_ids = [
            int(row["episode_id"])
            for row in connection.execute("SELECT DISTINCT episode_id FROM transcript_chunks").fetchall()
        ]
        for episode_id in episode_ids:
            rebuild_entity_relations_for_episode(connection, episode_id)
        entity_count = int(connection.execute("SELECT COUNT(*) AS count FROM entities").fetchone()["count"])
        relation_count = int(connection.execute("SELECT COUNT(*) AS count FROM entity_relations").fetchone()["count"])

    return {"entities": entity_count, "mentions": mentions, "relations": relation_count}


def episode_exists(db_path: Path, source_url: str) -> bool:
    init_db(db_path)
    with connect(db_path) as connection:
        row = connection.execute(
            "SELECT 1 FROM episodes WHERE source_url = ? LIMIT 1",
            (source_url,),
        ).fetchone()
    return row is not None


def migrate_schema(connection: sqlite3.Connection) -> None:
    entity_columns = {row["name"] for row in connection.execute("PRAGMA table_info(entities)").fetchall()}
    if "entity_type" not in entity_columns:
        connection.execute("ALTER TABLE entities ADD COLUMN entity_type TEXT NOT NULL DEFAULT 'UNKNOWN'")
    if "confidence" not in entity_columns:
        connection.execute("ALTER TABLE entities ADD COLUMN confidence REAL NOT NULL DEFAULT 0.0")
    if "evidence" not in entity_columns:
        connection.execute("ALTER TABLE entities ADD COLUMN evidence TEXT")


def upsert_entity(
    connection: sqlite3.Connection,
    name: str,
    entity_type: str = "UNKNOWN",
    confidence: float = 0.0,
    evidence: str | None = None,
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO entities (name, entity_type, confidence, evidence)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            entity_type = CASE
                WHEN excluded.confidence >= entities.confidence THEN excluded.entity_type
                ELSE entities.entity_type
            END,
            confidence = MAX(entities.confidence, excluded.confidence),
            evidence = COALESCE(entities.evidence, excluded.evidence)
        """,
        (name, entity_type, confidence, evidence),
    )
    row = connection.execute("SELECT id FROM entities WHERE name = ?", (name,)).fetchone()
    return int(row["id"])


def rebuild_entity_relations_for_episode(connection: sqlite3.Connection, episode_id: int) -> None:
    connection.execute(
        """
        DELETE FROM entity_relations
        WHERE source_entity_id IN (
            SELECT DISTINCT entity_id FROM entity_mentions WHERE episode_id = ?
        )
        OR target_entity_id IN (
            SELECT DISTINCT entity_id FROM entity_mentions WHERE episode_id = ?
        )
        """,
        (episode_id, episode_id),
    )
    rows = connection.execute(
        """
        SELECT
            MIN(left_mentions.entity_id, right_mentions.entity_id) AS source_entity_id,
            MAX(left_mentions.entity_id, right_mentions.entity_id) AS target_entity_id,
            COUNT(DISTINCT left_mentions.chunk_id) AS shared_chunks,
            COUNT(DISTINCT left_mentions.episode_id) AS shared_episodes,
            MIN(transcript_chunks.text) AS evidence
        FROM entity_mentions AS left_mentions
        JOIN entity_mentions AS right_mentions
            ON right_mentions.chunk_id = left_mentions.chunk_id
           AND right_mentions.entity_id != left_mentions.entity_id
        JOIN transcript_chunks ON transcript_chunks.id = left_mentions.chunk_id
        WHERE left_mentions.episode_id = ?
        GROUP BY source_entity_id, target_entity_id
        """,
        (episode_id,),
    ).fetchall()
    for row in rows:
        shared_chunks = int(row["shared_chunks"])
        shared_episodes = int(row["shared_episodes"])
        weight = shared_chunks + shared_episodes * 0.25
        connection.execute(
            """
            INSERT OR REPLACE INTO entity_relations
            (source_entity_id, target_entity_id, relation_type, weight, shared_chunks, shared_episodes, evidence)
            VALUES (?, ?, 'CO_OCCURS', ?, ?, ?, ?)
            """,
            (
                int(row["source_entity_id"]),
                int(row["target_entity_id"]),
                weight,
                shared_chunks,
                shared_episodes,
                row["evidence"],
            ),
        )
