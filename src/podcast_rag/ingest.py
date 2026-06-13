from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from podcast_rag.config import Settings
from podcast_rag.db import add_episode, episode_exists
from podcast_rag.discovery import PlaylistMode, PlaylistOrder, discover_sources
from podcast_rag.media import download_audio
from podcast_rag.transcription import transcribe_audio


@dataclass(frozen=True)
class IngestResult:
    source_url: str
    title: str | None
    status: str
    episode_id: int | None = None
    message: str | None = None


def ingest_url(
    url: str,
    settings: Settings,
    playlist_mode: PlaylistMode,
    playlist_order: PlaylistOrder,
    max_items: int | None,
    whisper_model: str,
    device: str,
    compute_type: str,
    language: str | None,
    domain_profile: str | None = None,
    skip_existing: bool = True,
    transcribe_seconds: int | None = None,
) -> list[IngestResult]:
    sources = discover_sources(
        url,
        playlist_mode=playlist_mode,
        max_items=max_items,
        playlist_order=playlist_order,
    )
    results: list[IngestResult] = []

    for source in sources:
        check_url = source.webpage_url or source.url
        if skip_existing and episode_exists(settings.db_path, check_url):
            results.append(IngestResult(source_url=check_url, title=source.title, status="skipped", message="already ingested"))
            continue

        try:
            downloaded = download_audio(source, settings.media_dir)
            if skip_existing and episode_exists(settings.db_path, downloaded.source_url):
                results.append(
                    IngestResult(
                        source_url=downloaded.source_url,
                        title=downloaded.title,
                        status="skipped",
                        message="already ingested",
                    )
                )
                continue

            segments, detected_language = transcribe_audio(
                Path(downloaded.audio_path),
                model_size=whisper_model,
                device=device,
                compute_type=compute_type,
                language=language,
                transcript_dir=settings.transcript_dir,
                transcribe_seconds=transcribe_seconds,
            )
            episode_id = add_episode(
                settings.db_path,
                title=downloaded.title,
                segments=segments,
                source_url=downloaded.source_url,
                author=downloaded.author,
                language=language or detected_language,
                domain_profile=domain_profile,
            )
            results.append(
                IngestResult(
                    source_url=downloaded.source_url,
                    title=downloaded.title,
                    status="imported",
                    episode_id=episode_id,
                    message=f"{len(segments)} segments",
                )
            )
        except Exception as exc:
            results.append(IngestResult(source_url=source.url, title=source.title, status="failed", message=str(exc)))

    return results
