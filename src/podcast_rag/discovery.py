from __future__ import annotations

import re
from enum import Enum
from pathlib import PurePosixPath
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree

from podcast_rag.models import MediaSource

DIRECT_MEDIA_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}

YOUTUBE_RE = re.compile(
    r"https?://(?:www\.)?(?:youtube\.com/(?:watch\?[^\s\"'<>]+|playlist\?[^\s\"'<>]+|shorts/[^\s\"'<>]+)|youtu\.be/[^\s\"'<>]+)"
)
MEDIA_URL_RE = re.compile(r"https?://[^\s\"'<>]+?\.(?:aac|flac|m4a|mp3|mp4|ogg|opus|wav|webm)(?:\?[^\s\"'<>]*)?", re.I)
FEED_URL_RE = re.compile(r"https?://[^\s\"'<>]*(?:/feed|/rss|feeds\.libsyn\.com/[^\s\"'<>]+)", re.I)


class PlaylistMode(str, Enum):
    single = "single"
    all = "all"


class PlaylistOrder(str, Enum):
    source = "source"
    newest = "newest"


def discover_sources(
    url: str,
    playlist_mode: PlaylistMode,
    max_items: int | None = None,
    playlist_order: PlaylistOrder = PlaylistOrder.source,
) -> list[MediaSource]:
    if playlist_mode == PlaylistMode.single:
        return [discover_single_source(url)]

    sources = discover_with_ytdlp(url, max_items=max_items, playlist_order=playlist_order)
    if sources and not _only_echoed_web_page(url, sources):
        return sources

    return discover_from_web_page(url, max_items=max_items)


def discover_single_source(url: str) -> MediaSource:
    from yt_dlp import YoutubeDL

    options = {
        "quiet": True,
        "skip_download": True,
        "ignoreerrors": True,
        "noplaylist": True,
    }
    try:
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception:
        return MediaSource(url=url, source_type="media")

    if not info:
        return MediaSource(url=url, source_type="media")

    return MediaSource(
        url=info.get("webpage_url") or url,
        title=info.get("title"),
        webpage_url=info.get("webpage_url"),
        source_type="media",
    )


def discover_with_ytdlp(
    url: str,
    max_items: int | None = None,
    playlist_order: PlaylistOrder = PlaylistOrder.source,
) -> list[MediaSource]:
    from yt_dlp import YoutubeDL

    options = {
        "extract_flat": "in_playlist",
        "quiet": True,
        "skip_download": True,
        "ignoreerrors": True,
        "noplaylist": False,
    }
    try:
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception:
        return []

    if not info:
        return []

    entries = info.get("entries") or []
    if not entries:
        return [MediaSource(url=info.get("webpage_url") or url, title=info.get("title"), source_type="media")]

    sources: list[MediaSource] = []
    ordered_entries = list(entries)
    if playlist_order == PlaylistOrder.newest:
        ordered_entries = sorted(ordered_entries, key=_entry_recency_key, reverse=True)

    for entry in ordered_entries:
        if not entry:
            continue
        entry_url = entry.get("url") or entry.get("webpage_url")
        if not entry_url:
            continue
        if not entry_url.startswith("http"):
            ie_key = entry.get("ie_key")
            if ie_key and ie_key.lower() == "youtube":
                entry_url = f"https://www.youtube.com/watch?v={entry_url}"
            else:
                entry_url = urljoin(url, entry_url)
        sources.append(
            MediaSource(
                url=entry_url,
                title=entry.get("title"),
                webpage_url=entry.get("webpage_url"),
                source_type="playlist-entry",
            )
        )
        if max_items is not None and len(sources) >= max_items:
            break

    return sources


def _entry_recency_key(entry: dict[str, object] | None) -> tuple[int, str]:
    if not entry:
        return (0, "")
    timestamp = entry.get("timestamp") or entry.get("release_timestamp") or 0
    upload_date = str(entry.get("upload_date") or "")
    try:
        return (int(timestamp), upload_date)
    except (TypeError, ValueError):
        return (0, upload_date)


def discover_from_web_page(url: str, max_items: int | None = None) -> list[MediaSource]:
    import httpx
    from bs4 import BeautifulSoup

    response = httpx.get(url, follow_redirects=True, timeout=20)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    found: dict[str, MediaSource] = {}

    for tag in soup.find_all(["audio", "video", "source", "a"]):
        candidate = tag.get("src") or tag.get("href")
        if not candidate:
            continue
        absolute = urljoin(str(response.url), candidate)
        if _looks_like_media_url(absolute):
            found[absolute] = MediaSource(
                url=absolute,
                title=_clean_title(tag.get_text(" ", strip=True)) or None,
                webpage_url=str(response.url),
                source_type="web-media",
            )

    for candidate in _extract_media_urls(response.text):
        found[candidate] = MediaSource(
            url=candidate,
            title=_page_title(soup),
            webpage_url=str(response.url),
            source_type="web-media",
        )

    for source in discover_from_feeds(
        soup,
        page_url=str(response.url),
        page_title=_page_title(soup),
        page_html=response.text,
        max_items=max_items,
    ):
        found[source.url] = source

    for match in YOUTUBE_RE.finditer(response.text):
        youtube_url = match.group(0)
        found[youtube_url] = MediaSource(url=youtube_url, webpage_url=str(response.url), source_type="web-youtube")

    sources = list(found.values())
    if max_items is not None:
        return sources[:max_items]
    return sources


def _only_echoed_web_page(url: str, sources: list[MediaSource]) -> bool:
    if len(sources) != 1:
        return False
    source = sources[0]
    if _looks_like_media_url(source.url):
        return False
    return _normalized_url(source.url) == _normalized_url(url) and source.source_type == "media"


def _extract_media_urls(text: str) -> list[str]:
    normalized_text = _html_unescape(text)
    candidates = {match.group(0) for match in MEDIA_URL_RE.finditer(normalized_text)}
    return sorted(_html_unescape(candidate).rstrip(").,;") for candidate in candidates)


def discover_from_feeds(
    soup: object,
    page_url: str,
    page_title: str | None,
    page_html: str = "",
    max_items: int | None = None,
) -> list[MediaSource]:
    import httpx

    feed_urls: list[str] = []
    for tag in soup.find_all("link"):
        rel = " ".join(tag.get("rel") or [])
        type_ = str(tag.get("type") or "").lower()
        href = tag.get("href")
        if href and "alternate" in rel and ("rss" in type_ or "atom" in type_):
            feed_urls.append(urljoin(page_url, href))
    feed_urls.extend(_html_unescape(match.group(0)).rstrip(").,;") for match in FEED_URL_RE.finditer(_html_unescape(page_html)))

    sources: list[MediaSource] = []
    for feed_url in dict.fromkeys(feed_urls):
        try:
            response = httpx.get(feed_url, follow_redirects=True, timeout=20)
            response.raise_for_status()
        except Exception:
            continue
        sources.extend(_extract_feed_enclosures(response.text, page_url=page_url, page_title=page_title))
        if max_items is not None and len(sources) >= max_items:
            return sources[:max_items]
    return sources


def _extract_feed_enclosures(feed_xml: str, page_url: str, page_title: str | None) -> list[MediaSource]:
    try:
        root = ElementTree.fromstring(feed_xml)
    except ElementTree.ParseError:
        return []

    page_slug = PurePosixPath(urlparse(page_url).path).name
    normalized_title = _normalize_text(page_title or "")
    sources: list[MediaSource] = []
    for item in root.findall(".//item"):
        title = _child_text(item, "title")
        link = _child_text(item, "link")
        guid = _child_text(item, "guid")
        if not _feed_item_matches_page(title=title, link=link, guid=guid, page_slug=page_slug, normalized_title=normalized_title):
            continue
        for enclosure in item.findall("enclosure"):
            url = enclosure.get("url")
            if url and _looks_like_media_url(url):
                sources.append(MediaSource(url=url, title=title or page_title, webpage_url=page_url, source_type="feed-enclosure"))
    return sources


def _feed_item_matches_page(title: str | None, link: str | None, guid: str | None, page_slug: str, normalized_title: str) -> bool:
    for value in [link, guid]:
        if value and page_slug and page_slug in urlparse(value).path:
            return True
    if title and normalized_title:
        title_norm = _normalize_text(title)
        return normalized_title in title_norm or title_norm in normalized_title
    return False


def _child_text(item: ElementTree.Element, tag_name: str) -> str | None:
    child = item.find(tag_name)
    if child is None or child.text is None:
        return None
    return _clean_title(child.text)


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _html_unescape(value: str) -> str:
    import html

    return html.unescape(value.replace("\\/", "/"))


def _normalized_url(url: str) -> str:
    parsed = urlparse(url)
    return parsed._replace(fragment="", query="").geturl().rstrip("/")


def _page_title(soup: object) -> str | None:
    title = getattr(soup, "title", None)
    if title and getattr(title, "string", None):
        return _clean_title(str(title.string))
    return None


def _looks_like_media_url(url: str) -> bool:
    parsed = urlparse(url)
    suffix = PurePosixPath(parsed.path).suffix.lower()
    return suffix in DIRECT_MEDIA_EXTENSIONS


def _clean_title(value: str) -> str:
    return " ".join(value.split())
