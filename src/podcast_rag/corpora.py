from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from pathlib import Path
from typing import Any

from podcast_rag.config import Settings, build_settings

CORPORA_FILE = "corpora.json"


@dataclass(frozen=True)
class CorpusConfig:
    id: str
    name: str
    data_dir: str
    description: str | None = None
    domain_profile: str | None = None
    qdrant_url: str | None = None
    tags: list[str] = field(default_factory=list)


def registry_path(base_data_dir: Path) -> Path:
    return base_data_dir / CORPORA_FILE


def load_corpora(base_data_dir: Path) -> list[CorpusConfig]:
    path = registry_path(base_data_dir)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [CorpusConfig(**item) for item in payload.get("corpora", [])]


def save_corpora(base_data_dir: Path, corpora: list[CorpusConfig]) -> None:
    base_data_dir.mkdir(parents=True, exist_ok=True)
    payload = {"corpora": [asdict(corpus) for corpus in sorted(corpora, key=lambda item: item.id)]}
    registry_path(base_data_dir).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def create_corpus(
    base_data_dir: Path,
    corpus_id: str,
    name: str | None = None,
    data_dir: Path | None = None,
    description: str | None = None,
    domain_profile: str | None = None,
    qdrant_url: str | None = None,
    tags: list[str] | None = None,
) -> CorpusConfig:
    normalized_id = normalize_corpus_id(corpus_id)
    corpora = load_corpora(base_data_dir)
    if any(corpus.id == normalized_id for corpus in corpora):
        raise ValueError(f"Corpus {normalized_id!r} already exists")

    resolved_data_dir = data_dir or (base_data_dir / "corpora" / normalized_id)
    resolved_data_dir.mkdir(parents=True, exist_ok=True)
    corpus = CorpusConfig(
        id=normalized_id,
        name=name or normalized_id.replace("-", " ").title(),
        data_dir=str(resolved_data_dir),
        description=description,
        domain_profile=domain_profile,
        qdrant_url=qdrant_url,
        tags=tags or [],
    )
    save_corpora(base_data_dir, [*corpora, corpus])
    return corpus


def get_corpus(base_data_dir: Path, corpus_id: str) -> CorpusConfig:
    for corpus in load_corpora(base_data_dir):
        if corpus.id == corpus_id:
            return corpus
    raise LookupError(f"Corpus {corpus_id!r} does not exist")


def resolve_corpus_settings(base_data_dir: Path, corpus_id: str | None = None, qdrant_url: str | None = None) -> Settings:
    if not corpus_id or corpus_id == "default":
        return build_settings(base_data_dir, qdrant_url=qdrant_url)
    corpus = get_corpus(base_data_dir, corpus_id)
    return build_settings(Path(corpus.data_dir), qdrant_url=qdrant_url or corpus.qdrant_url)


def resolve_corpus_set(base_data_dir: Path, corpus_selector: str | None = None) -> list[CorpusConfig]:
    corpora = load_corpora(base_data_dir)
    if not corpus_selector or corpus_selector == "default":
        return [default_corpus(base_data_dir)]
    if corpus_selector == "all":
        return corpora or [default_corpus(base_data_dir)]

    requested = [item.strip() for item in corpus_selector.split(",") if item.strip()]
    resolved: list[CorpusConfig] = []
    for corpus_id in requested:
        if corpus_id == "default":
            resolved.append(default_corpus(base_data_dir))
        else:
            resolved.append(get_corpus(base_data_dir, corpus_id))
    return resolved


def default_corpus(base_data_dir: Path) -> CorpusConfig:
    return CorpusConfig(id="default", name="Default corpus", data_dir=str(base_data_dir))


def corpus_to_dict(corpus: CorpusConfig) -> dict[str, Any]:
    return asdict(corpus)


def normalize_corpus_id(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    if not normalized:
        raise ValueError("Corpus id cannot be empty")
    return normalized
