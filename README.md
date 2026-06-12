# Podcast RAG

Local-first RAG tooling for podcasts and videos.

Current status: CLI foundation for importing transcripts, preserving timestamps, chunking text, and running basic lexical search.

## Quick Start

```bash
uv run podcast-rag init-db
uv run podcast-rag ingest-transcript transcript.txt --title "Episode title" --source-url "https://example.com"
uv run podcast-rag search "Felipe II"
uv run podcast-rag episodes
```

By default, data is stored under `./data`.
