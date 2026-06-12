from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

STOPWORDS = {
    "A",
    "Al",
    "Ante",
    "Aunque",
    "Como",
    "Con",
    "Contra",
    "Cuando",
    "De",
    "Del",
    "Desde",
    "Despues",
    "Durante",
    "El",
    "Ella",
    "En",
    "Entre",
    "Era",
    "Es",
    "Este",
    "Esto",
    "Fue",
    "Hay",
    "La",
    "Las",
    "Lo",
    "Los",
    "Mas",
    "No",
    "Para",
    "Pero",
    "Por",
    "Que",
    "Se",
    "Sin",
    "Sobre",
    "Su",
    "Tambien",
    "También",
    "Tras",
    "Un",
    "Una",
    "Y",
}

ENTITY_RE = re.compile(
    r"\b(?:[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)(?:\s+(?:de|del|la|las|los|y|el|[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)){0,4}"
)

DATE_RE = re.compile(r"\b(?:\d{3,4}|siglo\s+[XVI]+|siglo\s+\d{1,2})\b", re.IGNORECASE)

PERSON_HINTS = {
    "rey",
    "reina",
    "emperador",
    "emperatriz",
    "monarca",
    "conde",
    "duque",
    "marques",
    "marqués",
    "virrey",
    "capitan",
    "capitán",
    "general",
    "cronista",
    "conquistador",
    "papa",
    "obispo",
}

PLACE_HINTS = {
    "ciudad",
    "villa",
    "reino",
    "imperio",
    "provincia",
    "region",
    "región",
    "territorio",
    "palacio",
    "monasterio",
    "iglesia",
    "catedral",
    "rio",
    "río",
    "mar",
    "castilla",
    "aragon",
    "aragón",
    "españa",
    "peru",
    "perú",
    "america",
    "américa",
}

EVENT_HINTS = {
    "batalla",
    "guerra",
    "revuelta",
    "rebelion",
    "rebelión",
    "asedio",
    "conquista",
    "asesinato",
    "muerte",
    "expedicion",
    "expedición",
    "armada",
    "tratado",
}

CONCEPT_HINTS = {
    "monarquia",
    "monarquía",
    "dinastia",
    "dinastía",
    "sucesion",
    "sucesión",
    "religion",
    "religión",
    "politica",
    "política",
    "imperio",
    "corona",
    "poder",
}

KNOWN_PLACES = {
    "el escorial",
    "españa",
    "castilla",
    "aragon",
    "aragón",
    "peru",
    "perú",
    "lima",
    "inglaterra",
    "america",
    "américa",
}

KNOWN_CONCEPTS = {
    "corona",
    "la corona",
    "monarquia",
    "monarquía",
    "imperio",
}


@dataclass(frozen=True)
class EntityCandidate:
    name: str
    count: int
    entity_type: str
    confidence: float
    evidence: str


def extract_candidate_entities(text: str, limit: int = 30) -> list[EntityCandidate]:
    counter: Counter[str] = Counter()
    contexts: dict[str, str] = {}
    for match in ENTITY_RE.finditer(text):
        value = " ".join(match.group(0).split())
        if value in STOPWORDS:
            continue
        if len(value) < 3:
            continue
        context = _context_window(text, match.start(), match.end())
        for entity_value in split_coordinated_entity(value):
            counter[entity_value] += 1
            contexts.setdefault(entity_value, context)

    for match in DATE_RE.finditer(text):
        value = match.group(0)
        counter[value] += 1
        contexts.setdefault(value, _context_window(text, match.start(), match.end()))

    candidates = []
    for name, count in counter.most_common(limit):
        entity_type, confidence = infer_entity_type(name, contexts.get(name, ""))
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


def infer_entity_type(name: str, context: str) -> tuple[str, float]:
    lowered = f"{name} {context}".lower()
    scores = {
        "PERSON": _score_hints(lowered, PERSON_HINTS),
        "PLACE": _score_hints(lowered, PLACE_HINTS),
        "EVENT": _score_hints(lowered, EVENT_HINTS),
        "CONCEPT": _score_hints(lowered, CONCEPT_HINTS),
        "DATE": 4.0 if DATE_RE.fullmatch(name) else 0.0,
    }

    if name.lower() in KNOWN_PLACES:
        scores["PLACE"] += 4.0
    if name.lower() in KNOWN_CONCEPTS:
        scores["CONCEPT"] += 4.0
    if name.startswith(("Guerra", "Batalla", "Conquista", "Armada", "Tratado")):
        scores["EVENT"] += 1.6
    if name.startswith(("Reino", "Imperio", "Corona")):
        scores["PLACE"] += 0.4
        scores["CONCEPT"] += 0.3
    if _looks_like_person_name(name):
        scores["PERSON"] += 2.2
    if any(hint in context.lower() for hint in PERSON_HINTS):
        scores["PERSON"] += 0.8

    entity_type, score = max(scores.items(), key=lambda item: item[1])
    if score <= 0:
        return "UNKNOWN", 0.35
    return entity_type, min(0.95, 0.45 + score * 0.2)


def _score_hints(text: str, hints: set[str]) -> float:
    return sum(1.0 for hint in hints if hint in text)


def _looks_like_person_name(name: str) -> bool:
    words = name.split()
    if len(words) < 2:
        return False
    if words[0] in {"El", "La", "Los", "Las"}:
        return False
    if name.lower() in KNOWN_PLACES:
        return False
    if name.startswith(("Guerra", "Batalla", "Conquista", "Armada", "Tratado", "Reino", "Imperio", "Corona")):
        return False
    return all(word[:1].isupper() or word in {"de", "del", "la", "las", "los", "y", "el"} for word in words)


def split_coordinated_entity(value: str) -> list[str]:
    for separator in (" y la ", " y el ", " y los ", " y las ", " y "):
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


def _context_window(text: str, start: int, end: int, radius: int = 90) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return " ".join(text[left:right].split())
