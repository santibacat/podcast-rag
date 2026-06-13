from podcast_rag.discovery import (
    PlaylistMode,
    PlaylistOrder,
    YOUTUBE_RE,
    _extract_feed_enclosures,
    _extract_media_urls,
    _looks_like_media_url,
    discover_sources,
)
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


def test_all_mode_falls_back_when_ytdlp_echoes_page(monkeypatch):
    monkeypatch.setattr(
        "podcast_rag.discovery.discover_with_ytdlp",
        lambda url, max_items, playlist_order: [MediaSource(url=url, title="HTML page", source_type="media")],
    )
    monkeypatch.setattr(
        "podcast_rag.discovery.discover_from_web_page",
        lambda url, max_items: [MediaSource(url="https://cdn.example.com/episode.mp3", title="Episode", source_type="web-media")],
    )

    sources = discover_sources("https://example.com/episode", PlaylistMode.all, playlist_order=PlaylistOrder.source)

    assert sources == [MediaSource(url="https://cdn.example.com/episode.mp3", title="Episode", source_type="web-media")]


def test_extracts_media_urls_from_embedded_text():
    html = '{"audio":"https:\\/\\/traffic.libsyn.com\\/secure\\/show\\/episode.mp3?download=1"}'

    assert _extract_media_urls(html) == ["https://traffic.libsyn.com/secure/show/episode.mp3?download=1"]


def test_extracts_matching_feed_enclosure():
    feed = """
    <rss><channel><item>
      <title>La Primera Vuelta al Mundo</title>
      <link>https://memoriasdeuntambor.com/la-primera-vuelta-al-mundo</link>
      <enclosure url="https://traffic.libsyn.com/show/episode.mp3" type="audio/mpeg" />
    </item></channel></rss>
    """

    sources = _extract_feed_enclosures(
        feed,
        page_url="https://memoriasdeuntambor.com/la-primera-vuelta-al-mundo",
        page_title="La Primera Vuelta al Mundo - Memorias de un tambor.",
    )

    assert sources[0].url == "https://traffic.libsyn.com/show/episode.mp3"
    assert sources[0].source_type == "feed-enclosure"


def test_youtube_regex_ignores_channel_links():
    html = """
    <a href="http://www.youtube.com/c/Memoriasdeuntamborpodcast">channel</a>
    <a href="https://www.youtube.com/watch?v=_-Tuvu6CIQA&t=1196s">episode</a>
    """

    assert [match.group(0) for match in YOUTUBE_RE.finditer(html)] == [
        "https://www.youtube.com/watch?v=_-Tuvu6CIQA&t=1196s"
    ]
