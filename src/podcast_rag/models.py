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
