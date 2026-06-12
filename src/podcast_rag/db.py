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
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS entity_mentions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    episode_id INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    chunk_id INTEGER REFERENCES transcript_chunks(id) ON DELETE CASCADE,
    count INTEGER NOT NULL DEFAULT 1
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


def add_episode(
    db_path: Path,
    title: str,
    segments: list[TranscriptSegment],
    source_url: str | None = None,
    author: str | None = None,
    language: str | None = None,
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
            for name, count in extract_candidate_entities(text):
                entity_id = upsert_entity(connection, name)
                connection.execute(
                    """
                    INSERT INTO entity_mentions (entity_id, episode_id, chunk_id, count)
                    VALUES (?, ?, ?, ?)
                    """,
                    (entity_id, episode_id, chunk_id, count),
                )

    return episode_id


def episode_exists(db_path: Path, source_url: str) -> bool:
    init_db(db_path)
    with connect(db_path) as connection:
        row = connection.execute(
            "SELECT 1 FROM episodes WHERE source_url = ? LIMIT 1",
            (source_url,),
        ).fetchone()
    return row is not None


def upsert_entity(connection: sqlite3.Connection, name: str) -> int:
    cursor = connection.execute("INSERT OR IGNORE INTO entities (name) VALUES (?)", (name,))
    if cursor.lastrowid:
        return int(cursor.lastrowid)
    row = connection.execute("SELECT id FROM entities WHERE name = ?", (name,)).fetchone()
    return int(row["id"])
