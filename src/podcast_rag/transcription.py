from __future__ import annotations

import json
from pathlib import Path

from podcast_rag.models import TranscriptSegment


def transcribe_audio(
    audio_path: Path,
    model_size: str = "small",
    device: str = "auto",
    compute_type: str = "auto",
    language: str | None = None,
    transcript_dir: Path | None = None,
) -> tuple[list[TranscriptSegment], str | None]:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments_iter, info = model.transcribe(str(audio_path), language=language, vad_filter=True)

    segments: list[TranscriptSegment] = []
    raw_segments: list[dict[str, object]] = []
    for segment in segments_iter:
        text = segment.text.strip()
        if not text:
            continue
        segments.append(
            TranscriptSegment(
                text=text,
                start_seconds=float(segment.start),
                end_seconds=float(segment.end),
            )
        )
        raw_segments.append(
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": text,
            }
        )

    detected_language = getattr(info, "language", None)
    if transcript_dir is not None:
        transcript_dir.mkdir(parents=True, exist_ok=True)
        output_path = transcript_dir / f"{audio_path.stem}.json"
        output_path.write_text(
            json.dumps(
                {
                    "audio_path": str(audio_path),
                    "language": detected_language,
                    "duration": getattr(info, "duration", None),
                    "segments": raw_segments,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return segments, detected_language
