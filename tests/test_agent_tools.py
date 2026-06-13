from pathlib import Path

from podcast_rag.agent_tools import AgentToolConfig, agentic_research, infer_query_topics


def test_infer_query_topics_prefers_explicit_topic_match(monkeypatch):
    config = AgentToolConfig(data_dir=Path("data"))

    monkeypatch.setattr(
        "podcast_rag.agent_tools.tool_topics",
        lambda limit, config: [
            {"name": "Francisco Pizarro", "mentions": 2},
            {"name": "Peru", "mentions": 1},
        ],
    )

    topics = infer_query_topics("Que relacion hay con Francisco Pizarro?", config)

    assert topics[0]["name"] == "Francisco Pizarro"


def test_agentic_research_builds_brief_from_retrieval_and_connections(monkeypatch):
    config = AgentToolConfig(data_dir=Path("data"))

    monkeypatch.setattr(
        "podcast_rag.agent_tools.infer_query_topics",
        lambda question, config: [{"name": "Francisco Pizarro", "mentions": 2}],
    )
    monkeypatch.setattr(
        "podcast_rag.agent_tools.tool_retrieve",
        lambda query, limit, topic, config: [
            {
                "episode_id": 1,
                "title": "Pizarro",
                "start_seconds": 1.0,
                "context_text": "Francisco Pizarro viaja a Peru.",
            }
        ],
    )
    monkeypatch.setattr(
        "podcast_rag.agent_tools.tool_connections",
        lambda topic, limit, config: [
            {
                "source": "Francisco Pizarro",
                "target": "Peru",
                "relation_type": "CO_OCCURS",
                "weight": 1.25,
            }
        ],
    )

    result = agentic_research("Que relacion hay entre Francisco Pizarro y Peru?", config=config)

    assert result["inferred_topics"][0]["name"] == "Francisco Pizarro"
    assert result["tool_calls"][0]["tool"] == "retrieve"
    assert "Francisco Pizarro viaja a Peru" in result["brief"]


def test_agentic_research_llm_mode_adds_synthesis(monkeypatch):
    config = AgentToolConfig(data_dir=Path("data"))

    monkeypatch.setattr("podcast_rag.agent_tools.infer_query_topics", lambda question, config: [])
    monkeypatch.setattr(
        "podcast_rag.agent_tools.tool_retrieve",
        lambda query, limit, topic, config: [
            {
                "episode_id": 1,
                "title": "Magallanes",
                "start_seconds": 12.0,
                "context_text": "Magallanes organiza la expedicion.",
            }
        ],
    )
    monkeypatch.setattr("podcast_rag.agent_tools.tool_connections", lambda topic, limit, config: [])
    monkeypatch.setattr("podcast_rag.agent_tools.synthesize_with_llm", lambda question, local_brief: "Respuesta sintetizada.")

    result = agentic_research("Que se dice de Magallanes?", config=config, mode="llm")

    assert result["mode"] == "llm"
    assert result["llm_answer"] == "Respuesta sintetizada."
    assert "Local evidence brief" in result["brief"]


def test_mcp_server_module_imports():
    import podcast_rag.mcp_server as mcp_server

    assert mcp_server.mcp is not None
