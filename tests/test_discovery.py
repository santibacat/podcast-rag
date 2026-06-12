from podcast_rag.discovery import PlaylistMode, discover_sources, _looks_like_media_url
from podcast_rag.models import MediaSource


def test_detects_direct_media_urls_with_querystrings():
    assert _looks_like_media_url("https://example.com/audio/episode.mp3?download=1")
    assert _looks_like_media_url("https://example.com/video/item.webm")
    assert not _looks_like_media_url("https://example.com/post/felipe-ii")


def test_single_source_discovery_uses_metadata_when_available(monkeypatch):
    monkeypatch.setattr(
        "podcast_rag.discovery.discover_single_source",
        lambda url: MediaSource(url=url, title="Episode", source_type="media"),
    )

    sources = discover_sources("https://example.com/episode", PlaylistMode.single)

    assert sources == [MediaSource(url="https://example.com/episode", title="Episode", source_type="media")]
