# AGENTS.md

Guidance for agents working on this repository.

## Project Purpose

This repo implements a local-first agentic retrieval system for podcasts and videos. It ingests media/transcripts, transcribes audio, extracts entities, builds semantic connections, indexes chunks in Qdrant, and exposes retrieval tools through CLI and MCP.

## Core Architecture

- SQLite is the source of truth for episodes, transcript segments, chunks, entities, mentions, and entity relations.
- Qdrant is the retrieval index for dense+sparse hybrid search.
- The CLI is implemented with Typer in `src/podcast_rag/cli.py`.
- Reusable agent tools live in `src/podcast_rag/agent_tools.py`.
- The MCP server lives in `src/podcast_rag/mcp_server.py`.

## Setup

```bash
uv run pytest
```

The first use of Whisper, sentence-transformers, or Qdrant FastEmbed models may download model files.

## Main CLI Flows

Import a transcript:

```bash
uv run podcast-rag ingest-transcript examples/pizarro_transcript.txt --title Pizarro --language es --domain-profile history_es
```

Download and transcribe a URL:

```bash
uv run podcast-rag ingest-url "https://example.com/podcast-page" --playlist-mode single --language es
```

Create and use named corpora:

```bash
uv run podcast-rag create-corpus memorias --name "Memorias de un Tambor" --domain-profile history_es
uv run podcast-rag ingest-url "https://example.com/podcast-page" --playlist-mode all --max-items 5 --corpus memorias
uv run podcast-rag index-retrieval --corpus memorias
uv run podcast-rag ask "Donde se habla de Pizarro?" --corpus memorias
```

Prefer `--corpus` over manually passing nested data directories when working with registered podcast collections. The dashboard supports the same registry and can display one corpus or `all` corpora together.

Build the recommended retrieval index:

```bash
uv run podcast-rag index-retrieval
```

Domain profiles:

```bash
uv run podcast-rag domain-profiles
uv run podcast-rag profile-info history_es
uv run podcast-rag rebuild-entities --domain-profile history_es
uv run podcast-rag index-retrieval --force
```

Use `generic_es` by default for Spanish material. Use `history_es` for Spanish historical/cultural podcasts. Profiles live in `src/podcast_rag/domain_profiles.py`; avoid hardcoding domain-specific NER rules directly in `entities.py`.

Chunking currently accumulates transcript segments up to roughly 180 words with roughly 35 words of overlap, preserving timestamps and source segment ranges. Retrieval should prefer Qdrant hybrid dense+sparse search through `index-retrieval`, `hybrid-search`, and `retrieve`.

Transparency commands:

```bash
uv run podcast-rag chunking-info
uv run podcast-rag system-status
```

Search and retrieve:

```bash
uv run podcast-rag hybrid-search "Francisco Pizarro Peru"
uv run podcast-rag retrieve "donde se conecta Pizarro con Peru" --before 2 --after 2
uv run podcast-rag ask "Que relacion hay entre Francisco Pizarro y Peru?"
```

Explore entities:

```bash
uv run podcast-rag topics
uv run podcast-rag connections --topic "Francisco Pizarro"
uv run podcast-rag related "Francisco Pizarro"
```

Use analytics before building or changing dashboard surfaces:

```bash
uv run podcast-rag corpus-stats
uv run podcast-rag entity-profile "Francisco Pizarro"
uv run podcast-rag timeline --topic "Francisco Pizarro"
uv run podcast-rag episode-insights 1
uv run podcast-rag topic-matrix --json
uv run podcast-rag graph-export graph.json
uv run podcast-rag quality-report
```

Run the dashboard:

```bash
cd dashboard
npm install
npm run build
cd ..
uv run podcast-rag-dashboard --data-dir data --port 8765
```

The dashboard frontend lives in `dashboard/` as a React/Vite app. The Python dashboard server serves `dashboard/dist` and exposes `/api/*`. For frontend development, run `uv run podcast-rag-dashboard --data-dir data --port 8765` plus `cd dashboard && npm run dev`; Vite proxies API calls to the Python server.

The dashboard is intentionally powered by the same CLI/MCP-ready APIs. When adding dashboard data, prefer adding backend queries in `analytics.py` or reusable tools in `agent_tools.py` first, then consume them from the React API client. Keep `?corpus=...` support for new API routes; `corpus=all` should return a sensible merged view when practical.

## Qdrant

By default, Qdrant uses embedded local storage under `data/qdrant`. Embedded local Qdrant is single-process; do not run multiple Qdrant-backed commands at the same time for the same `--data-dir`.

For concurrent use, run Qdrant server and pass `QDRANT_URL`:

```bash
QDRANT_URL="http://localhost:6333" uv run podcast-rag index-retrieval
uv run podcast-rag hybrid-search "Pizarro Peru" --qdrant-url "http://localhost:6333"
```

## MCP Server

Start the MCP server:

```bash
uv run podcast-rag-mcp
```

MCP tools can now create and populate corpora end to end:

- `list_corpora`
- `create_corpus`
- `discover_url`
- `ingest_url`
- `rebuild_corpus_entities`
- `index_retrieval`
- `research_across_corpora`

Query and analytics MCP tools also accept `corpus` where practical. Use `research_across_corpora` for comparisons/synergies across registered podcast collections.

Exposed MCP tools:

- `episodes`
- `topics`
- `connections`
- `related`
- `context`
- `lexical_search`
- `retrieve`
- `research`
- `analytics_corpus_stats`
- `analytics_entity_profile`
- `analytics_timeline`
- `analytics_episode_insights`
- `analytics_topic_matrix`
- `analytics_graph`
- `analytics_quality_report`
- `analytics_system_status`
- `chunking_info`
- `domain_profile_info`

Use `retrieve` for evidence with context. Use `research` for the local agentic workflow that selects topics, retrieves evidence, and includes entity connections.
Use the analytics tools for dashboard cards, graph views, timelines, matrices, and quality/debug panels.

## Testing

Run all tests:

```bash
uv run pytest
```

Compile check:

```bash
uv run python -m compileall src tests
```

Tests should avoid network/model downloads. Use mocks/fakes for Qdrant, Whisper, and embedding behavior when possible.

## Development Notes

- Keep ingestion idempotent by `source_url`.
- Preserve timestamps through all processing.
- Prefer adding retrieval capabilities in `agent_tools.py` first, then expose them through CLI/MCP.
- Prefer adding dashboard data queries in `analytics.py` first, then expose them through CLI/MCP.
- Keep SQLite migrations backward-compatible in `db.py`.
- Do not put agent-only logic directly in the MCP layer; MCP should wrap reusable tools.
- Keep domain-specific entity hints in domain profiles, not in generic extraction logic.
