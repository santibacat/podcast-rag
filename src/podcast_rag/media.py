from __future__ import annotations

from pathlib import Path

from podcast_rag.models import DownloadedMedia, MediaSource


def download_audio(source: MediaSource, media_dir: Path) -> DownloadedMedia:
    from yt_dlp import YoutubeDL

    media_dir.mkdir(parents=True, exist_ok=True)
    options = {
        "format": "bestaudio/best",
        "outtmpl": str(media_dir / "%(extractor)s-%(id)s.%(ext)s"),
        "quiet": True,
        "noplaylist": True,
        "ignoreerrors": False,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }

    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(source.url, download=True)
        prepared = Path(ydl.prepare_filename(info))

    audio_path = prepared.with_suffix(".mp3")
    if not audio_path.exists() and prepared.exists():
        audio_path = prepared

    title = info.get("title") or source.title or audio_path.stem
    author = info.get("uploader") or info.get("channel") or info.get("creator")
    source_url = info.get("webpage_url") or source.webpage_url or source.url

    return DownloadedMedia(
        source_url=source_url,
        title=title,
        author=author,
        audio_path=str(audio_path),
        webpage_url=info.get("webpage_url") or source.webpage_url,
    )
