from podcast_rag.entities import extract_candidate_entities, infer_entity_type, split_coordinated_entity


def test_contextual_entity_type_inference_for_person_place_event_and_date():
    assert infer_entity_type("Francisco Pizarro", "el conquistador fue asesinado en Lima")[0] == "PERSON"
    assert infer_entity_type("El Escorial", "el monasterio fue simbolo religioso")[0] == "PLACE"
    assert infer_entity_type("Lima", "asesinato guerra civil centros de poder")[0] == "PLACE"
    assert infer_entity_type("La Corona", "guerra civil centros de poder politico")[0] == "CONCEPT"
    assert infer_entity_type("Guerra de Sucesion", "la guerra cambio la corona")[0] == "EVENT"
    assert infer_entity_type("1533", "en 1533 ocurrio la conquista")[0] == "DATE"


def test_extract_candidate_entities_returns_contextual_metadata():
    candidates = extract_candidate_entities(
        "El conquistador Francisco Pizarro fue asesinado durante la guerra civil en Peru en 1541."
    )
    by_name = {candidate.name: candidate for candidate in candidates}

    assert by_name["Francisco Pizarro"].entity_type == "PERSON"
    assert by_name["Peru"].entity_type == "PLACE"
    assert by_name["1541"].entity_type == "DATE"
    assert by_name["Francisco Pizarro"].evidence


def test_split_coordinated_entity():
    assert split_coordinated_entity("Lima y la Corona") == ["Lima", "La Corona"]
