import pytest

from podcast_rag.llm import synthesize_with_llm


def test_synthesize_with_llm_requires_configuration(monkeypatch):
    monkeypatch.delenv("PODCAST_RAG_LLM_MODEL", raising=False)
    monkeypatch.delenv("PODCAST_RAG_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("PODCAST_RAG_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="LLM mode is not configured"):
        synthesize_with_llm("Question", "Evidence")
