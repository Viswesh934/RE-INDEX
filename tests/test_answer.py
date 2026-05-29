from app.answer import answer_from_hits
from app.index import SearchHit
from app.storage import ChunkRecord


def test_answer_refuses_when_unsupported():
    chunk = ChunkRecord(
        document_id="d",
        file_name="a.pdf",
        page_number=1,
        chunk_id="c1",
        chunk_text="This document is about apples.",
        source_type="pdf_text",
        timestamp=1.0,
    )
    hit = SearchHit(chunk=chunk, semantic_score=0.1, keyword_score=0.0, combined_score=0.1)
    result = answer_from_hits("What is the refund policy?", [hit])
    assert result.refused is True
    assert "I don't know" in result.answer


def test_answer_uses_relevant_sentence():
    chunk = ChunkRecord(
        document_id="d",
        file_name="a.pdf",
        page_number=3,
        chunk_id="c1",
        chunk_text="The refund policy allows refunds within 30 days. Requests must be submitted by email.",
        source_type="pdf_text",
        timestamp=1.0,
    )
    hit = SearchHit(chunk=chunk, semantic_score=0.8, keyword_score=0.5, combined_score=0.7)
    result = answer_from_hits("What is the refund policy?", [hit])
    assert result.refused is False
    assert "30 days" in result.answer
    assert result.citations == ["Source: a.pdf, p. 3"]
