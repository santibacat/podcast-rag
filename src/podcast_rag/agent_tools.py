from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from podcast_rag.config import build_settings
from podcast_rag.retrieval_qdrant import (
    DEFAULT_QDRANT_COLLECTION,
    DEFAULT_QDRANT_DENSE_MODEL,
    DEFAULT_QDRANT_SPARSE_MODEL,
    QdrantIndexConfig,
    retrieve_evidence,
)
from podcast_rag.search import entity_connections, get_chunk_context, list_episodes, list_topics, related_topics, search_chunks


@dataclass(frozen=True)
class AgentToolConfig:
    data_dir: Path = Path("data")
    qdrant_url: str | None = None
    collection: str = DEFAULT_QDRANT_COLLECTION
    dense_model: str = DEFAULT_QDRANT_DENSE_MODEL
    sparse_model: str = DEFAULT_QDRANT_SPARSE_MODEL

    @property
    def qdrant_config(self) -> QdrantIndexConfig:
        settings = build_settings(self.data_dir, qdrant_url=self.qdrant_url)
        return QdrantIndexConfig(
            collection_name=self.collection,
            dense_model=self.dense_model,
            sparse_model=self.sparse_model,
            url=settings.qdrant_url,
        )


def tool_list_episodes(config: AgentToolConfig | None = None) -> list[dict[str, Any]]:
    resolved = config or AgentToolConfig()
    settings = build_settings(resolved.data_dir, qdrant_url=resolved.qdrant_url)
    return [dict(row) for row in list_episodes(settings.db_path)]


def tool_topics(limit: int = 50, config: AgentToolConfig | None = None) -> list[dict[str, Any]]:
    resolved = config or AgentToolConfig()
    settings = build_settings(resolved.data_dir, qdrant_url=resolved.qdrant_url)
    return [dict(row) for row in list_topics(settings.db_path, limit=limit)]


def tool_connections(topic: str | None = None, limit: int = 50, config: AgentToolConfig | None = None) -> list[dict[str, Any]]:
    resolved = config or AgentToolConfig()
    settings = build_settings(resolved.data_dir, qdrant_url=resolved.qdrant_url)
    return [dict(row) for row in entity_connections(settings.db_path, name=topic, limit=limit)]


def tool_related(topic: str, limit: int = 25, config: AgentToolConfig | None = None) -> list[dict[str, Any]]:
    resolved = config or AgentToolConfig()
    settings = build_settings(resolved.data_dir, qdrant_url=resolved.qdrant_url)
    return [dict(row) for row in related_topics(settings.db_path, topic, limit=limit)]


def tool_context(
    chunk_id: int,
    before_segments: int = 2,
    after_segments: int = 2,
    config: AgentToolConfig | None = None,
) -> dict[str, Any]:
    resolved = config or AgentToolConfig()
    settings = build_settings(resolved.data_dir, qdrant_url=resolved.qdrant_url)
    return dict(
        get_chunk_context(
            settings.db_path,
            chunk_id=chunk_id,
            before_segments=before_segments,
            after_segments=after_segments,
        )
    )


def tool_lexical_search(query: str, limit: int = 10, config: AgentToolConfig | None = None) -> list[dict[str, Any]]:
    resolved = config or AgentToolConfig()
    settings = build_settings(resolved.data_dir, qdrant_url=resolved.qdrant_url)
    return [dict(row) for row in search_chunks(settings.db_path, query=query, limit=limit)]


def tool_retrieve(
    query: str,
    limit: int = 5,
    before_segments: int = 2,
    after_segments: int = 2,
    episode_id: int | None = None,
    topic: str | None = None,
    config: AgentToolConfig | None = None,
) -> list[dict[str, Any]]:
    resolved = config or AgentToolConfig()
    settings = build_settings(resolved.data_dir, qdrant_url=resolved.qdrant_url)
    return retrieve_evidence(
        query=query,
        db_path=settings.db_path,
        qdrant_dir=settings.qdrant_dir,
        config=resolved.qdrant_config,
        limit=limit,
        before_segments=before_segments,
        after_segments=after_segments,
        episode_id=episode_id,
        topic=topic,
    )


def agentic_research(
    question: str,
    limit: int = 5,
    config: AgentToolConfig | None = None,
) -> dict[str, Any]:
    resolved = config or AgentToolConfig()
    candidate_topics = infer_query_topics(question, resolved)
    primary_topic = candidate_topics[0]["name"] if candidate_topics else None

    tool_calls: list[dict[str, Any]] = []
    evidence = tool_retrieve(question, limit=limit, topic=primary_topic, config=resolved)
    tool_calls.append({"tool": "retrieve", "topic": primary_topic, "result_count": len(evidence)})

    if not evidence and primary_topic:
        evidence = tool_retrieve(question, limit=limit, config=resolved)
        tool_calls.append({"tool": "retrieve", "topic": None, "result_count": len(evidence)})

    connections = tool_connections(primary_topic, limit=10, config=resolved) if primary_topic else []
    if primary_topic:
        tool_calls.append({"tool": "connections", "topic": primary_topic, "result_count": len(connections)})

    lexical = []
    if not evidence:
        lexical = tool_lexical_search(question, limit=limit, config=resolved)
        tool_calls.append({"tool": "lexical_search", "result_count": len(lexical)})

    return {
        "question": question,
        "inferred_topics": candidate_topics,
        "tool_calls": tool_calls,
        "evidence": evidence,
        "connections": connections,
        "lexical_fallback": lexical,
        "brief": build_research_brief(question, evidence, connections, lexical),
    }


def infer_query_topics(question: str, config: AgentToolConfig, limit: int = 5) -> list[dict[str, Any]]:
    lowered = question.lower()
    topics = tool_topics(limit=200, config=config)
    matches = [topic for topic in topics if str(topic["name"]).lower() in lowered]
    if matches:
        return matches[:limit]
    terms = {term for term in lowered.replace("¿", " ").replace("?", " ").split() if len(term) >= 4}
    scored = []
    for topic in topics:
        name = str(topic["name"]).lower()
        score = sum(1 for term in terms if term in name)
        if score:
            item = dict(topic)
            item["match_score"] = score
            scored.append(item)
    scored.sort(key=lambda item: (int(item["match_score"]), int(item.get("mentions", 0))), reverse=True)
    return scored[:limit]


def build_research_brief(
    question: str,
    evidence: list[dict[str, Any]],
    connections: list[dict[str, Any]],
    lexical: list[dict[str, Any]],
) -> str:
    lines = [f"Question: {question}"]
    if evidence:
        lines.append("Evidence:")
        for index, item in enumerate(evidence, start=1):
            timestamp = item.get("start_seconds")
            stamp = f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}" if isinstance(timestamp, (float, int)) else "--:--"
            lines.append(f"{index}. Episode {item.get('episode_id')} [{stamp}] {item.get('title')}: {item.get('context_text')}")
    elif lexical:
        lines.append("Lexical fallback:")
        for index, item in enumerate(lexical, start=1):
            lines.append(f"{index}. Episode {item.get('episode_id')} {item.get('title')}: {item.get('snippet')}")
    else:
        lines.append("No direct evidence found.")

    if connections:
        lines.append("Entity connections:")
        for connection in connections[:5]:
            lines.append(
                f"- {connection.get('source')} -> {connection.get('target')} "
                f"({connection.get('relation_type')}, weight={connection.get('weight')})"
            )
    return "\n".join(lines)
