from __future__ import annotations

import re
from collections import Counter

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


def extract_candidate_entities(text: str, limit: int = 30) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for match in ENTITY_RE.finditer(text):
        value = " ".join(match.group(0).split())
        if value in STOPWORDS:
            continue
        if len(value) < 3:
            continue
        counter[value] += 1
    return counter.most_common(limit)
