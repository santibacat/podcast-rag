from __future__ import annotations

from pathlib import Path

from podcast_rag.models import TranscriptSegment
from podcast_rag.timecode import TIMECODE_RE, parse_timecode


def parse_transcript_file(path: Path) -> list[TranscriptSegment]:
    raw = path.read_text(encoding="utf-8")
    return parse_transcript_text(raw)


def parse_transcript_text(raw: str) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    untimed_lines: list[str] = []

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        match = TIMECODE_RE.match(stripped)
        if match:
            if untimed_lines:
                segments.append(TranscriptSegment(text=" ".join(untimed_lines)))
                untimed_lines = []
            start = parse_timecode(match.group(1))
            text = match.group(2).strip()
            if text:
                segments.append(TranscriptSegment(text=text, start_seconds=start))
            continue

        untimed_lines.append(stripped)

    if untimed_lines:
        segments.append(TranscriptSegment(text=" ".join(untimed_lines)))

    return infer_segment_ends(segments)


def infer_segment_ends(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    inferred: list[TranscriptSegment] = []
    for index, segment in enumerate(segments):
        end = segment.end_seconds
        if end is None and segment.start_seconds is not None:
            for next_segment in segments[index + 1 :]:
                if next_segment.start_seconds is not None:
                    end = next_segment.start_seconds
                    break
        inferred.append(
            TranscriptSegment(
                text=segment.text,
                start_seconds=segment.start_seconds,
                end_seconds=end,
            )
        )
    return inferred
