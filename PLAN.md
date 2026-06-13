# Podcast RAG Implementation Plan

## Goal

Build a local-first RAG system for podcasts and videos, optimized for collections that share an author, theme, or historical domain. The system should ingest media links, transcribe audio, preserve timestamps, build searchable indexes, extract entities and concepts, and support agentic workflows for exploration.

## Product Capabilities

1. Ingest podcast or video URLs.
2. Download and normalize audio.
3. Detect language and transcribe locally.
4. Store timestamped transcripts.
5. Search by phrase, topic, person, place, event, or concept.
6. Show where something is discussed, including timestamps and nearby context.
7. Build a dynamic entity/topic index.
8. Explore correlations and connections between entities.
9. Expose tools that an agent can call to answer higher-level questions.

## Initial Architecture

The first implementation will be a Python CLI managed with `uv`.

Core components:

- `podcast_rag.cli`: Typer-based command line interface.
- `podcast_rag.db`: SQLite persistence and schema migrations.
- `podcast_rag.ingest`: source ingestion and transcript import.
- `podcast_rag.chunking`: transcript chunking with timestamp preservation.
- `podcast_rag.search`: lexical search over transcript chunks.
- `podcast_rag.entities`: early entity/topic extraction.
- `podcast_rag.domain_profiles`: configurable domain-specific entity extraction profiles.
- `podcast_rag.models`: shared dataclasses and domain objects.

Storage:

- SQLite for episodes, transcript segments, chunks, entities, and mentions.
- Qdrant local storage for dense+sparse retrieval indexes.
- Local filesystem for downloaded media, extracted audio, and raw transcripts.

## CLI Milestones

### Milestone 1: Local Transcript RAG Foundation

Commands:

```bash
uv run podcast-rag init-db
uv run podcast-rag ingest-transcript path/to/transcript.txt --title "Episode title" --source-url "https://..."
uv run podcast-rag episodes
uv run podcast-rag search "Felipe II"
uv run podcast-rag show EPISODE_ID
uv run podcast-rag topics
```

Features:

- Create SQLite schema.
- Import plain text or timestamped transcript files.
- Split transcripts into searchable chunks.
- Store timestamps when available.
- Run lexical search with context.
- Extract simple candidate topics and named terms.

### Milestone 2: Media URL Ingestion

Commands:

```bash
uv run podcast-rag ingest-url "https://youtube.com/..."
uv run podcast-rag discover-url "https://example.com/podcast-page"
uv run podcast-rag ingest-url "https://youtube.com/playlist?list=..." --playlist-mode all --max-items 5
```

Features:

- Use `yt-dlp` to download metadata and audio.
- Use `ffmpeg` to normalize audio.
- Store media assets under `data/media`.
- Avoid re-downloading sources already ingested.
- Expand playlists and web pages that contain multiple media links.
- Allow ingesting one item, all items, or a limited number of items.

Status: implemented for `yt-dlp`-supported media, YouTube playlists, direct media links found in web pages, and embedded YouTube links found in web pages.

### Milestone 3: Local Transcription

Features:

- Use `faster-whisper` for local transcription.
- Detect language automatically.
- Store timestamped transcript segments.
- Allow model selection through config:
  - `tiny`
  - `base`
  - `small`
  - `medium`
  - `large-v3`

Status: implemented through `ingest-url` and `transcribe-audio`.

### Milestone 4: Semantic Search

Features:

- Chunk transcripts semantically.
- Generate dense embeddings with FastEmbed/Qdrant.
- Generate sparse BM25 vectors with FastEmbed/Qdrant.
- Store retrieval vectors in local Qdrant.
- Add hybrid retrieval:
  - lexical score
  - semantic score
  - RRF fusion
  - later entity/topic boost

Commands:

```bash
uv run podcast-rag semantic-search "reyes catolicos y unidad dinastica"
uv run podcast-rag index-retrieval
uv run podcast-rag hybrid-search "reyes catolicos y unidad dinastica"
```

Status: implemented with local Qdrant through `index-retrieval` and `hybrid-search`. The previous SQLite-only semantic index remains as a fallback/debug path.

Additional retrieval features implemented:

- Qdrant payloads include episode metadata, timestamps, text, and candidate entities.
- `hybrid-search` supports filtering by `--episode-id` and `--topic`.
- `retrieve` returns hybrid results with expanded transcript context.
- `context` shows neighboring transcript segments around a chunk.
- Qdrant server mode is supported with `QDRANT_URL` or `--qdrant-url`.
- Contextual entity extraction stores inferred type, confidence, and evidence.
- Domain profiles make entity extraction configurable across corpora.
- Entity co-occurrence relations are populated automatically into `entity_relations`.
- Registered corpora allow separate podcast collections plus combined dashboard/API views.

Remaining retrieval work:

- Add reranking for top-k evidence.
- Add entity-aware scoring boosts.

Domain profile status:

- Implemented profiles: `generic_es`, `history_es`, `generic_en`, `custom`.
- Ingestion supports `--domain-profile`.
- Entity extraction can be rebuilt with `rebuild-entities`.
- Domain-specific hints should live in `domain_profiles.py`, not in core extraction logic.

### Milestone 5: Entity and Knowledge Graph

Features:

- Extract people, places, organizations, events, dates, battles, dynasties, and concepts.
- Store entity mentions with episode and timestamps.
- Build co-occurrence relationships.
- Support graph export.

Commands:

```bash
uv run podcast-rag entity "Felipe II"
uv run podcast-rag related "Guerra de Sucesion"
uv run podcast-rag export-graph graph.json
```

Status: candidate topic extraction and `related` co-occurrence lookup are implemented. Richer entity classification and graph export are pending.

### Milestone 6: Agentic Workflow

Features:

- Add tool abstractions for:
  - hybrid transcript search
  - entity lookup
  - timestamp context expansion
  - topic summarization
  - relationship exploration
- Add an `ask` command that lets an LLM reason over retrieved evidence.

Commands:

```bash
uv run podcast-rag ask "Where does the podcast connect Carlos V with the Comuneros?"
```

Status: first local agentic retrieval layer is implemented without external LLM dependency:

- `agent_tools.py` exposes reusable tools for episodes, topics, connections, related topics, lexical search, hybrid retrieval, and context.
- `ask` runs a local planning workflow: infer topic, retrieve evidence, inspect connections, and produce a research brief.
- `mcp_server.py` exposes query, analytics, corpus management, URL discovery, ingestion, entity rebuild, Qdrant indexing, and cross-corpus research tools for external agents.
- External agents can now call MCP tools to create a corpus, discover a podcast page/playlist, ingest/transcribe it, build retrieval indexes, and research it.

Remaining agentic work:

- Add LLM synthesis on top of the evidence brief.
- Add tool-call logging and run traces.
- Add configurable policies for when to call `retrieve`, `connections`, `context`, and lexical fallback.
- Add answer citations with timestamps and source URLs.

### Milestone 6.5: Multi-Corpus Workspaces

Features:

- Register named podcast/video corpora under one dashboard root.
- Keep corpus state isolated: SQLite DB, media, transcripts, and local Qdrant storage.
- Reuse the same dashboard for one corpus or a combined `all` view.
- Let MCP agents operate by corpus id instead of manual data paths.

Commands:

```bash
uv run podcast-rag create-corpus memorias --name "Memorias de un Tambor" --domain-profile history_es
uv run podcast-rag ingest-url "https://example.com/podcast-page" --playlist-mode all --corpus memorias
uv run podcast-rag index-retrieval --corpus memorias
uv run podcast-rag-dashboard --data-dir data --port 8765
```

Status: implemented through `corpora.json`, CLI `--corpus` support, dashboard `?corpus=...`, MCP corpus tools, and merged dashboard views for `corpus=all`.

### Milestone 7: Analytics Layer Before Dashboard

Purpose:

Build granular, dashboard-ready analytics so the UI can reveal structure in the corpus, not only provide chat/search.

Commands:

```bash
uv run podcast-rag corpus-stats
uv run podcast-rag entity-profile "Francisco Pizarro"
uv run podcast-rag timeline --topic "Francisco Pizarro"
uv run podcast-rag episode-insights 1
uv run podcast-rag graph-export graph.json
uv run podcast-rag quality-report
```

Features:

- Corpus-wide counts and density metrics.
- Entity profiles with mentions, timestamps, representative chunks, and connections.
- Entity/topic timelines across episodes and timestamps.
- Episode-level granular insights.
- Graph JSON export for future visualization.
- Quality/debug report for uncertain entities, missing timestamps, empty chunks, and indexing gaps.

Status: implemented as a React/Vite dashboard served by the Python dashboard API. Continue improving graph analysis, corpus metrics, and agent-facing drilldowns before adding heavier frontend dependencies.

## Development Priorities

1. Preserve timestamps throughout the pipeline.
2. Make ingestion idempotent.
3. Keep raw artifacts and derived indexes separate.
4. Prefer local/offline models where practical.
5. Keep the CLI clean and scriptable before adding a dashboard.
6. Design the search layer so a future agent can use it as tools.

## First Implementation Scope

This first coding pass implements Milestone 1:

- `uv` project scaffold.
- SQLite database schema.
- CLI commands for initialization, transcript ingestion, episode listing, display, basic search, and topic listing.
- Plain text transcript support.
- Basic timestamp parsing for common transcript formats.
- Simple chunking and keyword/topic extraction.

The media download, Whisper transcription, semantic vector store, and agentic `ask` command will be added after the local transcript foundation is working.
