from pathlib import Path

from podcast_rag.config import build_settings
from podcast_rag.discovery import PlaylistMode, PlaylistOrder
from podcast_rag.ingest import ingest_url
from podcast_rag.models import DownloadedMedia, MediaSource, TranscriptSegment
from podcast_rag.search import search_chunks


def test_ingest_url_orchestrates_download_transcription_and_indexing(tmp_path, monkeypatch):
    settings = build_settings(tmp_path / "data")

    def fake_discover_sources(url, playlist_mode, max_items, playlist_order):
        assert url == "https://example.com/episode"
        assert playlist_mode == PlaylistMode.single
        assert playlist_order == PlaylistOrder.source
        assert max_items is None
        return [MediaSource(url=url, title="Episode")]

    def fake_download_audio(source, media_dir):
        assert source.url == "https://example.com/episode"
        assert media_dir == settings.media_dir
        return DownloadedMedia(
            source_url=source.url,
            title="Episode",
            author="Author",
            audio_path=str(tmp_path / "episode.mp3"),
        )

    def fake_transcribe_audio(audio_path, model_size, device, compute_type, language, transcript_dir):
        assert audio_path == Path(tmp_path / "episode.mp3")
        assert model_size == "tiny"
        assert device == "cpu"
        assert compute_type == "int8"
        assert language == "es"
        assert transcript_dir == settings.transcript_dir
        return [TranscriptSegment("Felipe II y El Escorial.", 1, 5)], "es"

    monkeypatch.setattr("podcast_rag.ingest.discover_sources", fake_discover_sources)
    monkeypatch.setattr("podcast_rag.ingest.download_audio", fake_download_audio)
    monkeypatch.setattr("podcast_rag.ingest.transcribe_audio", fake_transcribe_audio)

    results = ingest_url(
        url="https://example.com/episode",
        settings=settings,
        playlist_mode=PlaylistMode.single,
        playlist_order=PlaylistOrder.source,
        max_items=None,
        whisper_model="tiny",
        device="cpu",
        compute_type="int8",
        language="es",
        domain_profile="generic_es",
    )

    assert results[0].status == "imported"
    assert results[0].episode_id == 1
    assert search_chunks(settings.db_path, "Felipe")[0]["title"] == "Episode"


def test_ingest_url_skips_existing_sources(tmp_path, monkeypatch):
    settings = build_settings(tmp_path / "data")
    calls = {"download": 0}

    def fake_discover_sources(url, playlist_mode, max_items, playlist_order):
        return [MediaSource(url="https://example.com/episode", title="Episode")]

    def fake_download_audio(source, media_dir):
        calls["download"] += 1
        return DownloadedMedia(
            source_url=source.url,
            title="Episode",
            author=None,
            audio_path=str(tmp_path / "episode.mp3"),
        )

    def fake_transcribe_audio(audio_path, model_size, device, compute_type, language, transcript_dir):
        return [TranscriptSegment("Felipe II.", 1, 2)], "es"

    monkeypatch.setattr("podcast_rag.ingest.discover_sources", fake_discover_sources)
    monkeypatch.setattr("podcast_rag.ingest.download_audio", fake_download_audio)
    monkeypatch.setattr("podcast_rag.ingest.transcribe_audio", fake_transcribe_audio)

    first = ingest_url(
        "https://example.com/episode",
        settings,
        PlaylistMode.single,
        PlaylistOrder.source,
        None,
        "tiny",
        "cpu",
        "int8",
        "es",
        "generic_es",
    )
    second = ingest_url(
        "https://example.com/episode",
        settings,
        PlaylistMode.single,
        PlaylistOrder.source,
        None,
        "tiny",
        "cpu",
        "int8",
        "es",
        "generic_es",
    )

    assert first[0].status == "imported"
    assert second[0].status == "skipped"
    assert calls["download"] == 1
