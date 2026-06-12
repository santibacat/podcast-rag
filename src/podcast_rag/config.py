from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "podcast_rag.sqlite3"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"


def build_settings(data_dir: Path | None = None) -> Settings:
    return Settings(data_dir=data_dir or Path("data"))
