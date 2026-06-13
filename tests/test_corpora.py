from pathlib import Path

import pytest

from podcast_rag.corpora import create_corpus, get_corpus, load_corpora, resolve_corpus_set, resolve_corpus_settings


def test_create_and_resolve_corpus(tmp_path):
    corpus = create_corpus(
        tmp_path,
        "Memorias de un Tambor",
        name="Memorias",
        domain_profile="history_es",
        tags=["history", "spain"],
    )

    assert corpus.id == "memorias-de-un-tambor"
    assert Path(corpus.data_dir).exists()
    assert load_corpora(tmp_path)[0].name == "Memorias"
    assert get_corpus(tmp_path, "memorias-de-un-tambor").domain_profile == "history_es"

    settings = resolve_corpus_settings(tmp_path, "memorias-de-un-tambor")
    assert settings.data_dir == Path(corpus.data_dir)


def test_duplicate_corpus_is_rejected(tmp_path):
    create_corpus(tmp_path, "one")

    with pytest.raises(ValueError):
        create_corpus(tmp_path, "one")


def test_resolve_all_corpora(tmp_path):
    first = create_corpus(tmp_path, "first")
    second = create_corpus(tmp_path, "second")

    corpora = resolve_corpus_set(tmp_path, "all")

    assert [corpus.id for corpus in corpora] == [first.id, second.id]
