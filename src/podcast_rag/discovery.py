from __future__ import annotations

import re
from enum import Enum
from pathlib import PurePosixPath
from urllib.parse import urljoin, urlparse

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

YOUTUBE_RE = re.compile(r"https?://(?:www\.)?(?:youtube\.com|youtu\.be)/[^\s\"'<>]+")


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
    if sources:
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

    for match in YOUTUBE_RE.finditer(response.text):
        youtube_url = match.group(0)
        found[youtube_url] = MediaSource(url=youtube_url, webpage_url=str(response.url), source_type="web-youtube")

    sources = list(found.values())
    if max_items is not None:
        return sources[:max_items]
    return sources


def _looks_like_media_url(url: str) -> bool:
    parsed = urlparse(url)
    suffix = PurePosixPath(parsed.path).suffix.lower()
    return suffix in DIRECT_MEDIA_EXTENSIONS


def _clean_title(value: str) -> str:
    return " ".join(value.split())
