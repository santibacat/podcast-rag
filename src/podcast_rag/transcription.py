from __future__ import annotations

import json
from pathlib import Path

from podcast_rag.models import TranscriptSegment


def transcribe_audio(
    audio_path: Path,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = None,
    transcript_dir: Path | None = None,
    transcribe_seconds: int | None = None,
) -> tuple[list[TranscriptSegment], str | None]:
    from faster_whisper import WhisperModel

    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except RuntimeError as exc:
        if not _should_retry_on_cpu(exc, device):
            raise
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
    transcribe_options: dict[str, object] = {"language": language, "vad_filter": True}
    if transcribe_seconds is not None:
        transcribe_options["clip_timestamps"] = f"0,{transcribe_seconds}"
    segments_iter, info = model.transcribe(str(audio_path), **transcribe_options)

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
                    "transcribe_seconds": transcribe_seconds,
                    "segments": raw_segments,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return segments, detected_language


def _should_retry_on_cpu(exc: RuntimeError, device: str) -> bool:
    if device not in {"auto", "cuda"}:
        return False
    message = str(exc).lower()
    cuda_markers = [
        "libcublas",
        "libcudnn",
        "cuda",
        "cublas",
        "cudnn",
    ]
    return any(marker in message for marker in cuda_markers)
