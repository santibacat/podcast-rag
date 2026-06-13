from pathlib import Path
import sys
import types

from podcast_rag.transcription import transcribe_audio


def test_transcribe_audio_falls_back_to_cpu_when_cuda_libraries_are_missing(monkeypatch, tmp_path):
    calls = []

    class FakeSegment:
        start = 1.0
        end = 2.0
        text = " Hola mundo "

    class FakeInfo:
        language = "es"
        duration = 2.0

    class FakeWhisperModel:
        def __init__(self, model_size, device, compute_type):
            calls.append((model_size, device, compute_type))
            if device == "auto":
                raise RuntimeError("Library libcublas.so.12 is not found or cannot be loaded")

        def transcribe(self, audio_path, language, vad_filter):
            return iter([FakeSegment()]), FakeInfo()

    fake_module = types.SimpleNamespace(WhisperModel=FakeWhisperModel)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    segments, language = transcribe_audio(Path(tmp_path / "episode.mp3"), model_size="small", device="auto", compute_type="auto")

    assert calls == [("small", "auto", "auto"), ("small", "cpu", "int8")]
    assert language == "es"
    assert segments[0].text == "Hola mundo"


def test_transcribe_audio_defaults_to_cpu_int8(monkeypatch, tmp_path):
    calls = []

    class FakeInfo:
        language = "es"
        duration = 0.0

    class FakeWhisperModel:
        def __init__(self, model_size, device, compute_type):
            calls.append((model_size, device, compute_type))

        def transcribe(self, audio_path, **kwargs):
            return iter([]), FakeInfo()

    fake_module = types.SimpleNamespace(WhisperModel=FakeWhisperModel)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    transcribe_audio(Path(tmp_path / "episode.mp3"))

    assert calls == [("small", "cpu", "int8")]


def test_transcribe_audio_can_limit_transcription_seconds(monkeypatch, tmp_path):
    options = {}

    class FakeInfo:
        language = "es"
        duration = 120.0

    class FakeWhisperModel:
        def __init__(self, model_size, device, compute_type):
            pass

        def transcribe(self, audio_path, **kwargs):
            options.update(kwargs)
            return iter([]), FakeInfo()

    fake_module = types.SimpleNamespace(WhisperModel=FakeWhisperModel)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    segments, language = transcribe_audio(
        Path(tmp_path / "episode.mp3"),
        model_size="tiny",
        device="cpu",
        compute_type="int8",
        language="es",
        transcribe_seconds=45,
    )

    assert segments == []
    assert language == "es"
    assert options["clip_timestamps"] == "0,45"
