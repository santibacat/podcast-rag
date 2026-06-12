# Podcast RAG

Local-first RAG tooling for podcasts and videos.

Current status: CLI foundation for importing transcripts, preserving timestamps, chunking text, and running basic lexical search.

## Quick Start

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
uv run podcast-rag connections --topic "Francisco Pizarro"
uv run podcast-rag index-embeddings
uv run podcast-rag semantic-search "monarquia y simbolos religiosos"
uv run podcast-rag related "Felipe"
uv run podcast-rag episodes
```

By default, data is stored under `./data`.

Media ingestion requires `ffmpeg` to be available on your system path. Transcription uses `faster-whisper` locally.

Useful ingestion options:

- `--playlist-mode single`: process only the given URL.
- `--playlist-mode all`: expand playlists or pages with many media links.
- `--max-items N`: limit expanded playlists/pages to N items.
- `--playlist-order newest`: prefer newest items when source metadata includes dates.
- `--whisper-model small`: choose the local Whisper model size.
- `--language es`: force Spanish transcription, or omit it for auto-detection.

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

Full ingestion downloads audio and runs local Whisper, so it can take a while:

```bash
uv run podcast-rag ingest-url "https://memoriasdeuntambor.com/96-el-asesinato-de-pizarro" --playlist-mode single --language es --whisper-model small
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
QDRANT_URL="http://localhost:6333" uv run podcast-rag index-retrieval
uv run podcast-rag hybrid-search "Pizarro Peru" --qdrant-url "http://localhost:6333"
```

Retrieve evidence with expanded transcript context:

```bash
uv run podcast-rag retrieve "donde aparece El Escorial como simbolo politico" --topic "El Escorial" --before 2 --after 2
```

Explore automatically detected entity connections:

```bash
uv run podcast-rag topics
uv run podcast-rag connections --topic "Francisco Pizarro"
```

The older SQLite-only semantic index is still available for comparison and simple debugging:

```bash
uv run podcast-rag index-embeddings
uv run podcast-rag semantic-search "donde se habla de poder religioso y monarquia"
```
