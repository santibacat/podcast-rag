from __future__ import annotations

from podcast_rag.models import TranscriptChunk, TranscriptSegment

DEFAULT_MAX_WORDS = 180
DEFAULT_OVERLAP_WORDS = 35


def chunk_segments(
    segments: list[TranscriptSegment],
    max_words: int = DEFAULT_MAX_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[TranscriptChunk]:
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative")
    if overlap_words >= max_words:
        overlap_words = max(0, max_words // 4)

    chunks: list[TranscriptChunk] = []
    current: list[tuple[int, TranscriptSegment]] = []
    current_words = 0

    for index, segment in enumerate(segments):
        word_count = len(segment.text.split())
        if current and current_words + word_count > max_words:
            chunks.append(_build_chunk(current))
            current, current_words = _overlap_tail(current, overlap_words)

        current.append((index, segment))
        current_words += word_count

    if current:
        chunks.append(_build_chunk(current))

    return chunks


def _build_chunk(items: list[tuple[int, TranscriptSegment]]) -> TranscriptChunk:
    first_index = items[0][0]
    last_index = items[-1][0]
    segments = [item[1] for item in items]
    start = next((segment.start_seconds for segment in segments if segment.start_seconds is not None), None)
    end = next((segment.end_seconds for segment in reversed(segments) if segment.end_seconds is not None), None)
    return TranscriptChunk(
        text=" ".join(segment.text for segment in segments),
        start_seconds=start,
        end_seconds=end,
        segment_start=first_index,
        segment_end=last_index,
    )


def _overlap_tail(
    items: list[tuple[int, TranscriptSegment]],
    overlap_words: int,
) -> tuple[list[tuple[int, TranscriptSegment]], int]:
    if overlap_words == 0:
        return [], 0

    selected: list[tuple[int, TranscriptSegment]] = []
    total = 0
    for item in reversed(items):
        words = len(item[1].text.split())
        if selected and total + words > overlap_words:
            break
        selected.append(item)
        total += words
        if total >= overlap_words:
            break

    selected.reverse()
    return selected, total


def describe_chunking_strategy() -> dict[str, object]:
    return {
        "strategy": "timestamp-preserving segment accumulation",
        "max_words": DEFAULT_MAX_WORDS,
        "overlap_words": DEFAULT_OVERLAP_WORDS,
        "timestamp_policy": "timestamp start is the first available segment start; timestamp end is the last available segment end",
        "segment_policy": "chunks keep the source segment_start and segment_end positions",
        "why": "simple, deterministic, cheap to rebuild, and transparent before adding semantic boundary detection",
    }
