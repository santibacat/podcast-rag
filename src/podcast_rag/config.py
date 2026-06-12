from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    qdrant_url: str | None = None

    @property
    def db_path(self) -> Path:
        return self.data_dir / "podcast_rag.sqlite3"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def media_dir(self) -> Path:
        return self.data_dir / "media"

    @property
    def transcript_dir(self) -> Path:
        return self.data_dir / "transcripts"

    @property
    def qdrant_dir(self) -> Path:
        return self.data_dir / "qdrant"


def build_settings(data_dir: Path | None = None, qdrant_url: str | None = None) -> Settings:
    return Settings(data_dir=data_dir or Path("data"), qdrant_url=qdrant_url or os.getenv("QDRANT_URL"))
