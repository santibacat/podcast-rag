from podcast_rag.transcripts import parse_transcript_text


def test_parse_timestamped_transcript_infers_end_times():
    segments = parse_transcript_text(
        """
        [00:01] Intro sobre Felipe II
        [00:05] Se menciona El Escorial
        """
    )

    assert len(segments) == 2
    assert segments[0].start_seconds == 1
    assert segments[0].end_seconds == 5
    assert segments[1].start_seconds == 5


def test_parse_plain_transcript_combines_lines():
    segments = parse_transcript_text(
        """
        Primera linea.
        Segunda linea.
        """
    )

    assert len(segments) == 1
    assert segments[0].text == "Primera linea. Segunda linea."
