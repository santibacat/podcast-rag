from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from podcast_rag.domain_profiles import DomainProfile, get_domain_profile

ENTITY_RE = re.compile(
    r"\b(?:[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)(?:\s+(?:de|del|la|las|los|y|el|[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)){0,4}"
)

DATE_RE = re.compile(r"\b(?:\d{3,4}|siglo\s+[XVI]+|siglo\s+\d{1,2})\b", re.IGNORECASE)


@dataclass(frozen=True)
class EntityCandidate:
    name: str
    count: int
    entity_type: str
    confidence: float
    evidence: str


def extract_candidate_entities(
    text: str,
    limit: int = 30,
    domain_profile: str | DomainProfile | None = None,
) -> list[EntityCandidate]:
    profile = resolve_profile(domain_profile)
    counter: Counter[str] = Counter()
    contexts: dict[str, str] = {}
    for match in ENTITY_RE.finditer(text):
        value = " ".join(match.group(0).split())
        if value in profile.stopwords:
            continue
        if len(value) < 3:
            continue
        context = _context_window(text, match.start(), match.end())
        for entity_value in split_coordinated_entity(value, profile):
            counter[entity_value] += 1
            contexts.setdefault(entity_value, context)

    for match in DATE_RE.finditer(text):
        value = match.group(0)
        counter[value] += 1
        contexts.setdefault(value, _context_window(text, match.start(), match.end()))

    candidates = []
    for name, count in counter.most_common(limit):
        entity_type, confidence = infer_entity_type(name, contexts.get(name, ""), profile)
        candidates.append(
            EntityCandidate(
                name=name,
                count=count,
                entity_type=entity_type,
                confidence=confidence,
                evidence=contexts.get(name, ""),
            )
        )
    return candidates


def infer_entity_type(
    name: str,
    context: str,
    domain_profile: str | DomainProfile | None = None,
) -> tuple[str, float]:
    profile = resolve_profile(domain_profile)
    lowered = f"{name} {context}".lower()
    scores = {
        "PERSON": _score_hints(lowered, profile.person_hints),
        "PLACE": _score_hints(lowered, profile.place_hints),
        "EVENT": _score_hints(lowered, profile.event_hints),
        "CONCEPT": _score_hints(lowered, profile.concept_hints),
        "DATE": 4.0 if DATE_RE.fullmatch(name) else 0.0,
    }

    if name.lower() in profile.known_places:
        scores["PLACE"] += 4.0
    if name.lower() in profile.known_concepts:
        scores["CONCEPT"] += 4.0
    if name.startswith(profile.event_prefixes):
        scores["EVENT"] += 1.6
    if name.startswith(profile.concept_prefixes):
        scores["PLACE"] += 0.4
        scores["CONCEPT"] += 0.3
    if _looks_like_person_name(name, profile):
        scores["PERSON"] += profile.person_name_bias
    if any(hint in context.lower() for hint in profile.person_hints):
        scores["PERSON"] += 0.8

    entity_type, score = max(scores.items(), key=lambda item: item[1])
    if score <= 0:
        return "UNKNOWN", 0.35
    return entity_type, min(0.95, 0.45 + score * 0.2)


def _score_hints(text: str, hints: set[str]) -> float:
    return sum(1.0 for hint in hints if hint in text)


def _looks_like_person_name(name: str, profile: DomainProfile) -> bool:
    words = name.split()
    if len(words) < 2:
        return False
    if words[0] in {"El", "La", "Los", "Las"}:
        return False
    if name.lower() in profile.known_places:
        return False
    if name.startswith((*profile.event_prefixes, *profile.concept_prefixes)):
        return False
    return all(word[:1].isupper() or word in {"de", "del", "la", "las", "los", "y", "el"} for word in words)


def split_coordinated_entity(value: str, domain_profile: str | DomainProfile | None = None) -> list[str]:
    profile = resolve_profile(domain_profile)
    for separator in profile.coordination_separators:
        if separator in value:
            parts = [part.strip() for part in value.split(separator) if part.strip()]
            if len(parts) > 1:
                normalized = []
                article = separator.strip().split()[-1]
                for index, part in enumerate(parts):
                    if index > 0 and separator != " y ":
                        part = f"{article.capitalize()} {part}"
                    normalized.append(part)
                return normalized
    return [value]


def resolve_profile(domain_profile: str | DomainProfile | None = None) -> DomainProfile:
    if isinstance(domain_profile, DomainProfile):
        return domain_profile
    return get_domain_profile(domain_profile)


def _context_window(text: str, start: int, end: int, radius: int = 90) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return " ".join(text[left:right].split())
