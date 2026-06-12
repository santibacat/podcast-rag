import pytest

from podcast_rag.db import add_episode
from podcast_rag.embeddings import cosine_similarity, pack_vector, rebuild_chunk_embeddings, semantic_search, unpack_vector
from podcast_rag.models import TranscriptSegment


class FakeEmbedder:
    model_name = "fake-model"

    def embed(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    1.0 if "felipe" in lowered else 0.0,
                    1.0 if "escorial" in lowered else 0.0,
                    1.0 if "armada" in lowered else 0.0,
                ]
            )
        return vectors


def test_pack_vector_roundtrip():
    vector = [0.1, 0.2, 0.3]

    unpacked = unpack_vector(pack_vector(vector))

    assert unpacked == pytest.approx(vector)


def test_cosine_similarity():
    assert cosine_similarity([1, 0], [1, 0]) == 1
    assert cosine_similarity([1, 0], [0, 1]) == 0


def test_rebuild_embeddings_and_semantic_search(tmp_path):
    db_path = tmp_path / "rag.sqlite3"
    add_episode(
        db_path,
        title="Historia",
        segments=[
            TranscriptSegment("Felipe II y El Escorial.", 1, 5),
            TranscriptSegment("La Armada Invencible contra Inglaterra.", 5, 10),
        ],
    )

    indexed = rebuild_chunk_embeddings(db_path, FakeEmbedder(), batch_size=1)
    results = semantic_search(db_path, "Felipe", FakeEmbedder(), limit=1)

    assert indexed == 1
    assert results[0]["title"] == "Historia"
    assert "Felipe" in results[0]["text"]
