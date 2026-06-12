from podcast_rag.chunking import chunk_segments
from podcast_rag.models import TranscriptSegment


def test_chunk_segments_preserves_timestamp_bounds():
    chunks = chunk_segments(
        [
            TranscriptSegment("uno dos tres", start_seconds=1, end_seconds=2),
            TranscriptSegment("cuatro cinco seis", start_seconds=2, end_seconds=3),
        ],
        max_words=10,
    )

    assert len(chunks) == 1
    assert chunks[0].start_seconds == 1
    assert chunks[0].end_seconds == 3
