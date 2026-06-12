from __future__ import annotations

import re

TIMECODE_RE = re.compile(
    r"^\s*(?:\[|\()?((?:(?:\d{1,2}:)?\d{1,2}:\d{2})(?:[,.]\d{1,3})?)(?:\]|\))?\s*(.*)$"
)


def parse_timecode(value: str) -> float:
    normalized = value.replace(",", ".")
    parts = normalized.split(":")
    seconds = float(parts[-1])
    minutes = int(parts[-2]) if len(parts) >= 2 else 0
    hours = int(parts[-3]) if len(parts) >= 3 else 0
    return hours * 3600 + minutes * 60 + seconds


def format_timestamp(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
