import pytest

from podcast_rag.domain_profiles import get_domain_profile, list_domain_profiles


def test_list_domain_profiles_includes_generic_and_history():
    profiles = list_domain_profiles()

    assert "generic_es" in profiles
    assert "history_es" in profiles
    assert "generic_en" in profiles


def test_get_domain_profile_rejects_unknown_profile():
    with pytest.raises(ValueError):
        get_domain_profile("nope")
