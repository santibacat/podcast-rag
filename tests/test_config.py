from pathlib import Path

from podcast_rag.config import build_settings


def test_build_settings_reads_qdrant_url_from_argument():
    settings = build_settings(Path("data"), qdrant_url="http://localhost:6333")

    assert settings.qdrant_url == "http://localhost:6333"


def test_build_settings_reads_qdrant_url_from_environment(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")

    settings = build_settings(Path("data"))

    assert settings.qdrant_url == "http://qdrant:6333"
