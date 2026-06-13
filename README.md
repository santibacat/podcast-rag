# Podcast RAG

Local-first RAG tooling for podcasts and videos.

Current status: CLI foundation for importing transcripts, preserving timestamps, chunking text, and running basic lexical search.

## Quick Start

One command can process a podcast page, media file, YouTube video, or playlist into a ready-to-query corpus:

```bash
uv run podcast-rag process-url "https://example.com/podcast-page" \
  --corpus memorias \
  --create-corpus \
  --corpus-name "Memorias de un Tambor" \
  --domain-profile history_es \
  --language es \
  --whisper-model tiny
```

Then query it:

```bash
uv run podcast-rag query "Que se dice de Magallanes y Elcano?" --corpus memorias
```

For a quick smoke test, add `--transcribe-seconds 180`. For production quality, use `--whisper-model small` or larger and let CPU transcription run longer.

```bash
uv run podcast-rag init-db
uv run podcast-rag ingest-transcript transcript.txt --title "Episode title" --source-url "https://example.com"
uv run podcast-rag transcribe-audio episode.mp3 --title "Episode title" --language es
uv run podcast-rag discover-url "https://example.com/podcast-page"
uv run podcast-rag ingest-url "https://youtube.com/..." --playlist-mode single
uv run podcast-rag ingest-url "https://youtube.com/playlist?list=..." --playlist-mode all --playlist-order newest --max-items 5
uv run podcast-rag search "Felipe II"
uv run podcast-rag index-retrieval
uv run podcast-rag hybrid-search "Felipe Escorial simbolo religioso"
uv run podcast-rag query "Felipe Escorial simbolo religioso"
uv run podcast-rag connections --topic "Francisco Pizarro"
uv run podcast-rag index-embeddings
uv run podcast-rag semantic-search "monarquia y simbolos religiosos"
uv run podcast-rag related "Felipe"
uv run podcast-rag episodes
```

By default, data is stored under `./data`.

Media ingestion requires `ffmpeg` to be available on your system path. Transcription uses `faster-whisper` locally on CPU by default (`--device cpu --compute-type int8`). GPU use is opt-in through CLI/MCP flags.

The base install avoids optional Torch/CUDA dependencies. The recommended retrieval path is Qdrant + FastEmbed, which runs on CPU by default. The older SQLite-only `index-embeddings` path requires an optional extra:

```bash
uv sync --extra sentence-transformers
```

## Multiple Corpora

You can keep separate podcast collections and inspect them with the same dashboard. A registered corpus stores its own SQLite DB, media, transcripts, and local Qdrant index.

```bash
uv run podcast-rag create-corpus memorias --name "Memorias de un Tambor" --domain-profile history_es
uv run podcast-rag create-corpus arte --name "Arte e Historia" --domain-profile history_es

uv run podcast-rag process-url "https://example.com/podcast-page" --playlist-mode all --max-items 5 --corpus memorias

uv run podcast-rag topics --corpus memorias
uv run podcast-rag query "Donde se habla de Pizarro?" --corpus memorias
```

Run the dashboard with the registry root and use the corpus selector for `default`, any registered corpus, or `all` for a combined view:

```bash
uv run podcast-rag-dashboard --data-dir data --port 8765
```

Useful ingestion options:

- `--playlist-mode single`: process only the given URL.
- `--playlist-mode all`: expand playlists or pages with many media links.
- `--max-items N`: limit expanded playlists/pages to N items.
- `--playlist-order newest`: prefer newest items when source metadata includes dates.
- `--whisper-model small`: choose the local Whisper model size.
- `--device cpu`: default Whisper device; pass `--device cuda` only when you intentionally want GPU.
- `--compute-type int8`: default CPU-friendly Whisper compute type.
- `--language es`: force Spanish transcription, or omit it for auto-detection.
- `--domain-profile generic_es`: choose entity extraction heuristics.

Available domain profiles:

```bash
uv run podcast-rag domain-profiles
uv run podcast-rag profile-info history_es
```

Current profiles:

- `generic_es`: default Spanish profile with general-purpose hints.
- `history_es`: Spanish history profile with historical event/person/place/concept hints.
- `generic_en`: general-purpose English profile.
- `custom`: minimal low-assumption profile.

Rebuild entities and relations without reingesting:

```bash
uv run podcast-rag rebuild-entities --domain-profile history_es
uv run podcast-rag index-retrieval --force
```

## Chunking And Retrieval

Transcript chunks are built from timestamped transcript segments. The current strategy accumulates neighboring transcript segments up to about 180 words, with about 35 words of overlap between chunks. Each chunk preserves the first available start timestamp, final end timestamp, and source segment range.

Inspect the active chunking strategy:

```bash
uv run podcast-rag chunking-info
```

Recommended retrieval uses Qdrant hybrid search:

- dense multilingual embeddings for semantic similarity
- sparse BM25 vectors for keyword/fuzzy-ish lexical matching
- RRF fusion to combine both result sets

The older SQLite-only embedding path remains available through `index-embeddings` and `semantic-search`, mostly for debugging and comparison.

Inspect corpus/index state and recommended next actions:

```bash
uv run podcast-rag system-status
```

## Testing

Run the automated suite:

```bash
uv run pytest
```

Current coverage includes:

- transcript timestamp parsing
- chunking with timestamp preservation
- SQLite indexing and FTS search
- idempotent episode ingestion by `source_url`
- direct media URL detection
- mocked URL ingestion from discovery to searchable transcript
- embedding vector storage and semantic search with a deterministic test embedder

Preview real URLs without downloading or transcribing:

```bash
uv run podcast-rag discover-url "https://www.youtube.com/watch?v=_-Tuvu6CIQA&t=1196s" --playlist-mode single --json
uv run podcast-rag discover-url "https://memoriasdeuntambor.com/96-el-asesinato-de-pizarro" --playlist-mode single --json
```

Full ingestion downloads audio and runs local CPU Whisper, so it can take a while:

```bash
uv run podcast-rag ingest-url "https://memoriasdeuntambor.com/96-el-asesinato-de-pizarro" --playlist-mode single --language es --whisper-model small
```

For intentional GPU transcription, pass the device explicitly:

```bash
uv run podcast-rag ingest-url "https://example.com/podcast-page" --playlist-mode single --device cuda --compute-type float16
```

Build semantic search after transcripts are imported:

```bash
uv run podcast-rag index-retrieval
uv run podcast-rag hybrid-search "donde se habla de poder religioso y monarquia"
```

`index-retrieval` is the recommended retrieval path. It stores transcript chunks in local Qdrant under `data/qdrant` with both dense multilingual embeddings and sparse BM25 vectors, then `hybrid-search` queries both and fuses results with RRF.

Local Qdrant storage is single-process. Run Qdrant-backed CLI commands sequentially for the same `--data-dir`; use a Qdrant server later if you need concurrent indexing/search.

Use a Qdrant server instead of local embedded storage:

```bash
docker compose -f docker-compose.qdrant.yml up -d
uv run podcast-rag qdrant-health --qdrant-url "http://localhost:6333"
QDRANT_URL="http://localhost:6333" uv run podcast-rag index-retrieval
uv run podcast-rag hybrid-search "Pizarro Peru" --qdrant-url "http://localhost:6333"
```

Qdrant server is optional, free, and open source. The local embedded mode is convenient for one command at a time, but the server mode is recommended when the dashboard, CLI, MCP tools, or multiple agents may query/index concurrently.

Retrieve evidence with expanded transcript context:

```bash
uv run podcast-rag retrieve "donde aparece El Escorial como simbolo politico" --topic "El Escorial" --before 2 --after 2
uv run podcast-rag ask "Que relacion hay entre Francisco Pizarro y Peru?"
```

Explore automatically detected entity connections:

```bash
uv run podcast-rag topics
uv run podcast-rag connections --topic "Francisco Pizarro"
```

Dashboard-ready analytics:

```bash
uv run podcast-rag corpus-stats
uv run podcast-rag entity-profile "Francisco Pizarro"
uv run podcast-rag timeline --topic "Francisco Pizarro"
uv run podcast-rag episode-insights 1
uv run podcast-rag topic-matrix --json
uv run podcast-rag graph-export graph.json
uv run podcast-rag quality-report
```

Run the MCP server for external agents:

```bash
uv run podcast-rag-mcp
```

The MCP server exposes ingestion and corpus-management tools: `list_corpora`, `create_corpus`, `discover_url`, `process_url`, `ingest_url`, `rebuild_corpus_entities`, `index_retrieval`, and `research_across_corpora`. Prefer `process_url` for one-shot ingestion/indexing.

It also exposes query tools: `episodes`, `topics`, `connections`, `related`, `context`, `lexical_search`, `retrieve`, `query`, `research`, and analytics tools such as `analytics_corpus_stats`, `analytics_entity_profile`, `analytics_timeline`, `analytics_episode_insights`, `analytics_topic_matrix`, `analytics_graph`, `analytics_quality_report`, `analytics_system_status`, `chunking_info`, and `domain_profile_info`. Most tools accept `corpus` plus `data_dir`, so an agent can create a corpus, ingest a URL into it, index it, and then research it without manually managing paths.

Run the local dashboard:

```bash
cd dashboard
npm install
npm run build
cd ..
uv run podcast-rag-dashboard --data-dir data --port 8765
```

Then open `http://<server-ip-or-domain>:8765`. The dashboard binds to `0.0.0.0` by default so it can receive traffic on remote servers.

For frontend development with hot reload, run the Python API server on port `8765` and Vite in another terminal:

```bash
uv run podcast-rag-dashboard --data-dir data --host 0.0.0.0 --port 8765
cd dashboard
npm run dev
```

Then open `http://<server-ip-or-domain>:5173`. Vite binds to `0.0.0.0` and proxies `/api/*` to the Python dashboard server.

The older SQLite-only semantic index is still available for comparison and simple debugging, but it requires the optional `sentence-transformers` extra. On some Linux setups that extra may install Torch wheels with CUDA libraries, so keep it opt-in:

```bash
uv sync --extra sentence-transformers
uv run podcast-rag index-embeddings
uv run podcast-rag semantic-search "donde se habla de poder religioso y monarquia"
```

## Ask Modes

`ask` defaults to local agentic retrieval. It does not call an LLM unless you opt into LLM mode.

```bash
uv run podcast-rag ask "Que se dice de Magallanes?" --mode local
```

LLM mode first runs the same local tools, then sends the retrieved evidence to an OpenAI-compatible chat endpoint for synthesis:

```bash
export PODCAST_RAG_LLM_MODEL="your-chat-model"
export PODCAST_RAG_LLM_BASE_URL="https://api.openai.com/v1"
export PODCAST_RAG_LLM_API_KEY="..."
uv run podcast-rag ask "Que se dice de Magallanes?" --mode llm
```

For local LLM servers, point `PODCAST_RAG_LLM_BASE_URL` at their OpenAI-compatible `/v1` endpoint and set the model name they expose.
