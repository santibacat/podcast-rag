from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from podcast_rag.agent_tools import (
    AgentToolConfig,
    agentic_research,
    tool_connections,
    tool_context,
    tool_lexical_search,
    tool_list_episodes,
    tool_related,
    tool_retrieve,
    tool_topics,
)

mcp = FastMCP("podcast-rag")


def _config(data_dir: str = "data", qdrant_url: str | None = None) -> AgentToolConfig:
    return AgentToolConfig(data_dir=Path(data_dir), qdrant_url=qdrant_url)


@mcp.tool()
def episodes(data_dir: str = "data") -> list[dict[str, Any]]:
    """List ingested podcast/video episodes."""
    return tool_list_episodes(_config(data_dir))


@mcp.tool()
def topics(limit: int = 50, data_dir: str = "data") -> list[dict[str, Any]]:
    """List detected entities/topics with type, confidence, and counts."""
    return tool_topics(limit=limit, config=_config(data_dir))


@mcp.tool()
def connections(topic: str | None = None, limit: int = 50, data_dir: str = "data") -> list[dict[str, Any]]:
    """Show semantic connections between detected entities."""
    return tool_connections(topic=topic, limit=limit, config=_config(data_dir))


@mcp.tool()
def related(topic: str, limit: int = 25, data_dir: str = "data") -> list[dict[str, Any]]:
    """Show topics that co-occur with a topic in transcript chunks."""
    return tool_related(topic=topic, limit=limit, config=_config(data_dir))


@mcp.tool()
def context(
    chunk_id: int,
    before_segments: int = 2,
    after_segments: int = 2,
    data_dir: str = "data",
) -> dict[str, Any]:
    """Return transcript context around a chunk."""
    return tool_context(
        chunk_id=chunk_id,
        before_segments=before_segments,
        after_segments=after_segments,
        config=_config(data_dir),
    )


@mcp.tool()
def lexical_search(query: str, limit: int = 10, data_dir: str = "data") -> list[dict[str, Any]]:
    """Run SQLite FTS/BM25 lexical search over transcript chunks."""
    return tool_lexical_search(query=query, limit=limit, config=_config(data_dir))


@mcp.tool()
def retrieve(
    query: str,
    limit: int = 5,
    before_segments: int = 2,
    after_segments: int = 2,
    episode_id: int | None = None,
    topic: str | None = None,
    data_dir: str = "data",
    qdrant_url: str | None = None,
) -> list[dict[str, Any]]:
    """Run Qdrant hybrid retrieval and return evidence with expanded transcript context."""
    return tool_retrieve(
        query=query,
        limit=limit,
        before_segments=before_segments,
        after_segments=after_segments,
        episode_id=episode_id,
        topic=topic,
        config=_config(data_dir, qdrant_url=qdrant_url),
    )


@mcp.tool()
def research(question: str, limit: int = 5, data_dir: str = "data", qdrant_url: str | None = None) -> dict[str, Any]:
    """Run the local agentic retrieval workflow and return evidence, connections, and a brief."""
    return agentic_research(question=question, limit=limit, config=_config(data_dir, qdrant_url=qdrant_url))


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
