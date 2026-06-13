from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DomainProfile:
    name: str
    stopwords: set[str] = field(default_factory=set)
    person_hints: set[str] = field(default_factory=set)
    place_hints: set[str] = field(default_factory=set)
    event_hints: set[str] = field(default_factory=set)
    concept_hints: set[str] = field(default_factory=set)
    known_places: set[str] = field(default_factory=set)
    known_concepts: set[str] = field(default_factory=set)
    event_prefixes: tuple[str, ...] = ()
    concept_prefixes: tuple[str, ...] = ()
    coordination_separators: tuple[str, ...] = ()
    person_name_bias: float = 1.2


BASE_SPANISH_STOPWORDS = {
    "A",
    "Ahora",
    "Al",
    "Algo",
    "Algunos",
    "Ante",
    "Aqui",
    "Aquí",
    "Aunque",
    "Bien",
    "Bueno",
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
    "Ellos",
    "En",
    "Entre",
    "Entonces",
    "Era",
    "Es",
    "Eso",
    "Este",
    "Esto",
    "Estos",
    "Evidentemente",
    "Fue",
    "Hay",
    "Hola",
    "La",
    "Las",
    "Lo",
    "Los",
    "Mas",
    "Mientras",
    "Mucho",
    "Nada",
    "No",
    "Para",
    "Parece",
    "Pero",
    "Por",
    "Pues",
    "Que",
    "Realmente",
    "Se",
    "Si",
    "Sí",
    "Sin",
    "Sobre",
    "Su",
    "Tambien",
    "Tambien",
    "También",
    "Tras",
    "Un",
    "Una",
    "Y",
    "Ya",
}

BASE_ENGLISH_STOPWORDS = {
    "A",
    "An",
    "And",
    "Are",
    "As",
    "At",
    "But",
    "By",
    "For",
    "From",
    "In",
    "Into",
    "Is",
    "It",
    "Of",
    "On",
    "Or",
    "That",
    "The",
    "This",
    "To",
    "Was",
    "With",
}

GENERIC_ES = DomainProfile(
    name="generic_es",
    stopwords=BASE_SPANISH_STOPWORDS,
    person_hints={"persona", "autor", "autora", "entrevistado", "entrevistada", "doctor", "doctora", "profesor", "profesora"},
    place_hints={"ciudad", "pais", "país", "region", "región", "provincia", "territorio", "zona", "lugar"},
    event_hints={"evento", "caso", "crisis", "accidente", "conferencia", "debate", "entrevista"},
    concept_hints={"concepto", "idea", "teoria", "teoría", "metodo", "método", "modelo", "sistema", "problema"},
    coordination_separators=(" y la ", " y el ", " y los ", " y las ", " y "),
    person_name_bias=1.2,
)

HISTORY_ES = DomainProfile(
    name="history_es",
    stopwords=BASE_SPANISH_STOPWORDS,
    person_hints={
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
    },
    place_hints={
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
    },
    event_hints={
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
    },
    concept_hints={
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
    },
    known_places={"el escorial", "españa", "castilla", "aragon", "aragón", "peru", "perú", "lima", "inglaterra", "america", "américa"},
    known_concepts={"corona", "la corona", "monarquia", "monarquía", "imperio"},
    event_prefixes=("Guerra", "Batalla", "Conquista", "Armada", "Tratado"),
    concept_prefixes=("Reino", "Imperio", "Corona"),
    coordination_separators=(" y la ", " y el ", " y los ", " y las ", " y "),
    person_name_bias=2.2,
)

GENERIC_EN = DomainProfile(
    name="generic_en",
    stopwords=BASE_ENGLISH_STOPWORDS,
    person_hints={"person", "author", "guest", "host", "doctor", "professor", "researcher", "engineer"},
    place_hints={"city", "country", "region", "state", "province", "territory", "place"},
    event_hints={"event", "case", "crisis", "accident", "conference", "debate", "interview"},
    concept_hints={"concept", "idea", "theory", "method", "model", "system", "problem"},
    coordination_separators=(" and the ", " and "),
    person_name_bias=1.2,
)

CUSTOM = DomainProfile(name="custom", coordination_separators=(" y ", " and "), person_name_bias=0.8)

DOMAIN_PROFILES = {
    profile.name: profile
    for profile in (
        GENERIC_ES,
        HISTORY_ES,
        GENERIC_EN,
        CUSTOM,
    )
}

DEFAULT_DOMAIN_PROFILE = "generic_es"


def get_domain_profile(name: str | None = None) -> DomainProfile:
    profile_name = name or DEFAULT_DOMAIN_PROFILE
    try:
        return DOMAIN_PROFILES[profile_name]
    except KeyError as exc:
        valid = ", ".join(sorted(DOMAIN_PROFILES))
        raise ValueError(f"Unknown domain profile {profile_name!r}. Valid profiles: {valid}") from exc


def list_domain_profiles() -> list[str]:
    return sorted(DOMAIN_PROFILES)


def describe_domain_profile(name: str | None = None) -> dict[str, object]:
    profile = get_domain_profile(name)
    return {
        "name": profile.name,
        "stopwords": sorted(profile.stopwords),
        "person_hints": sorted(profile.person_hints),
        "place_hints": sorted(profile.place_hints),
        "event_hints": sorted(profile.event_hints),
        "concept_hints": sorted(profile.concept_hints),
        "known_places": sorted(profile.known_places),
        "known_concepts": sorted(profile.known_concepts),
        "event_prefixes": list(profile.event_prefixes),
        "concept_prefixes": list(profile.concept_prefixes),
        "coordination_separators": list(profile.coordination_separators),
        "person_name_bias": profile.person_name_bias,
    }
