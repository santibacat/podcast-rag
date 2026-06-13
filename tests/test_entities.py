from podcast_rag.entities import clean_entity_span, extract_candidate_entities, infer_entity_type, split_coordinated_entity


def test_contextual_entity_type_inference_for_person_place_event_and_date():
    assert infer_entity_type("Francisco Pizarro", "el conquistador fue asesinado en Lima", "history_es")[0] == "PERSON"
    assert infer_entity_type("El Escorial", "el monasterio fue simbolo religioso", "history_es")[0] == "PLACE"
    assert infer_entity_type("Lima", "asesinato guerra civil centros de poder", "history_es")[0] == "PLACE"
    assert infer_entity_type("La Corona", "guerra civil centros de poder politico", "history_es")[0] == "CONCEPT"
    assert infer_entity_type("Guerra de Sucesion", "la guerra cambio la corona", "history_es")[0] == "EVENT"
    assert infer_entity_type("1533", "en 1533 ocurrio la conquista", "history_es")[0] == "DATE"


def test_extract_candidate_entities_returns_contextual_metadata():
    candidates = extract_candidate_entities(
        "El conquistador Francisco Pizarro fue asesinado durante la guerra civil en Peru en 1541.",
        domain_profile="history_es",
    )
    by_name = {candidate.name: candidate for candidate in candidates}

    assert by_name["Francisco Pizarro"].entity_type == "PERSON"
    assert by_name["Peru"].entity_type == "PLACE"
    assert by_name["1541"].entity_type == "DATE"
    assert by_name["Francisco Pizarro"].evidence


def test_split_coordinated_entity():
    assert split_coordinated_entity("Lima y la Corona", "history_es") == ["Lima", "La Corona"]


def test_generic_profile_uses_context_without_history_known_places():
    assert infer_entity_type("Lima", "ciudad importante", "generic_es")[0] == "PLACE"


def test_discourse_connectors_are_not_extracted_as_entities():
    candidates = extract_candidate_entities(
        "Bueno, entonces Magallanes sale hacia América y Asia. De hecho, de nada sirve esa muletilla.",
        domain_profile="history_es",
    )
    names = {candidate.name for candidate in candidates}

    assert "Magallanes" in names
    assert "América" in names
    assert "Asia" in names
    assert "Bueno" not in names
    assert "Entonces" not in names
    assert "De" not in names


def test_entity_span_trims_dangling_connectors():
    assert clean_entity_span("Magallanes de") == "Magallanes"
    assert clean_entity_span("De Magallanes") == "Magallanes"
    assert clean_entity_span("El Escorial") == "El Escorial"
    assert clean_entity_span("Juan de Cartagena") == "Juan de Cartagena"
