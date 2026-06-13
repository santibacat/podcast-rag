from podcast_rag.config import build_settings
from podcast_rag.ingest import IngestResult
from podcast_rag.workflows import process_url_workflow


def test_process_url_workflow_runs_ingest_rebuild_and_index(tmp_path, monkeypatch):
    calls = {}

    def fake_ingest_url(**kwargs):
        calls["ingest"] = kwargs
        return [IngestResult(source_url=kwargs["url"], title="Episode", status="imported", episode_id=1, message="3 segments")]

    def fake_rebuild_entities(db_path, domain_profile):
        calls["rebuild"] = {"db_path": db_path, "domain_profile": domain_profile}
        return {"entities": 4, "mentions": 5, "relations": 6}

    def fake_index_qdrant_chunks(db_path, qdrant_dir, config, batch_size, force):
        calls["index"] = {
            "db_path": db_path,
            "qdrant_dir": qdrant_dir,
            "collection": config.collection_name,
            "batch_size": batch_size,
            "force": force,
        }
        return 7

    monkeypatch.setattr("podcast_rag.workflows.ingest_url", fake_ingest_url)
    monkeypatch.setattr("podcast_rag.workflows.rebuild_entities", fake_rebuild_entities)
    monkeypatch.setattr("podcast_rag.workflows.index_qdrant_chunks", fake_index_qdrant_chunks)

    result = process_url_workflow(
        url="https://example.com/feed",
        data_dir=tmp_path,
        language="es",
        transcribe_seconds=30,
        collection="test_chunks",
        force_index=True,
    )
    settings = build_settings(tmp_path)

    assert result["ready"] is True
    assert result["ingest"][0]["status"] == "imported"
    assert result["entities"]["entities"] == 4
    assert result["index"]["indexed_chunks"] == 7
    assert calls["ingest"]["settings"] == settings
    assert calls["ingest"]["language"] == "es"
    assert calls["ingest"]["transcribe_seconds"] == 30
    assert calls["index"]["collection"] == "test_chunks"
    assert calls["index"]["force"] is True


def test_process_url_workflow_can_create_missing_corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "podcast_rag.workflows.ingest_url",
        lambda **kwargs: [IngestResult(source_url=kwargs["url"], title="Episode", status="skipped", message="already ingested")],
    )
    monkeypatch.setattr("podcast_rag.workflows.rebuild_entities", lambda db_path, domain_profile: {"entities": 0, "mentions": 0, "relations": 0})
    monkeypatch.setattr("podcast_rag.workflows.index_qdrant_chunks", lambda *args, **kwargs: 0)

    result = process_url_workflow(
        url="https://example.com/show",
        data_dir=tmp_path,
        corpus="memorias",
        corpus_name="Memorias",
        create_missing_corpus=True,
        domain_profile="history_es",
    )

    assert result["corpus"] == "memorias"
    assert result["created_corpus"]["name"] == "Memorias"
