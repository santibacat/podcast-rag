from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TranscriptSegment:
    text: str
    start_seconds: float | None = None
    end_seconds: float | None = None


@dataclass(frozen=True)
class TranscriptChunk:
    text: str
    start_seconds: float | None
    end_seconds: float | None
    segment_start: int
    segment_end: int


@dataclass(frozen=True)
class MediaSource:
    url: str
    title: str | None = None
    webpage_url: str | None = None
    source_type: str = "media"


@dataclass(frozen=True)
class DownloadedMedia:
    source_url: str
    title: str
    author: str | None
    audio_path: str
    webpage_url: str | None = None
