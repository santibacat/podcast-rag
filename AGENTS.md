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
uv run podcast-rag ingest-transcript examples/pizarro_transcript.txt --title Pizarro --language es
```

Download and transcribe a URL:

```bash
uv run podcast-rag ingest-url "https://example.com/podcast-page" --playlist-mode single --language es
```

Build the recommended retrieval index:

```bash
uv run podcast-rag index-retrieval
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

Exposed MCP tools:

- `episodes`
- `topics`
- `connections`
- `related`
- `context`
- `lexical_search`
- `retrieve`
- `research`

Use `retrieve` for evidence with context. Use `research` for the local agentic workflow that selects topics, retrieves evidence, and includes entity connections.

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
- Keep SQLite migrations backward-compatible in `db.py`.
- Do not put agent-only logic directly in the MCP layer; MCP should wrap reusable tools.
