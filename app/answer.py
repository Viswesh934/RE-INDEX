from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from .index import SearchHit
from .storage import ChunkRecord


STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "what", "when", "where", "who", "whom",
    "which", "why", "how", "are", "was", "were", "been", "has", "have", "had", "does", "did", "can",
    "could", "would", "should", "will", "shall", "may", "might", "to", "of", "in", "on", "at", "by",
    "an", "a", "or", "is", "it", "as", "be", "if", "do", "not", "we", "you", "they", "i", "me",
}


@dataclass
class AnswerResult:
    answer: str
    citations: list[str]
    supporting_chunks: list[ChunkRecord]
    confidence: float
    refused: bool = False


def _tokens(text: str) -> set[str]:
    toks = re.findall(r"[A-Za-z0-9]+", text.lower())
    return {t for t in toks if len(t) > 2 and t not in STOPWORDS}


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _sentence_score(question_terms: set[str], sentence: str) -> float:
    terms = _tokens(sentence)
    if not terms or not question_terms:
        return 0.0
    return len(question_terms & terms) / max(1, len(question_terms))


def answer_from_hits(question: str, hits: list[SearchHit], min_support: float = 0.12) -> AnswerResult:
    if not hits:
        return AnswerResult(
            answer="I don't know based on the uploaded documents.",
            citations=[],
            supporting_chunks=[],
            confidence=0.0,
            refused=True,
        )

    q_terms = _tokens(question)
    scored_sentences: list[tuple[float, str, ChunkRecord]] = []

    for hit in hits:
        chunk = hit.chunk
        sentences = _sentences(chunk.chunk_text) or [chunk.chunk_text]
        for sent in sentences:
            score = _sentence_score(q_terms, sent) * 0.7 + float(hit.combined_score) * 0.3
            if score > 0:
                scored_sentences.append((score, sent, chunk))

    scored_sentences.sort(key=lambda x: x[0], reverse=True)

    if not scored_sentences:
        return AnswerResult(
            answer="I don't know based on the uploaded documents.",
            citations=[],
            supporting_chunks=[],
            confidence=0.0,
            refused=True,
        )

    best_score = scored_sentences[0][0]
    if best_score < min_support:
        return AnswerResult(
            answer="I don't know based on the uploaded documents.",
            citations=[],
            supporting_chunks=[],
            confidence=float(best_score),
            refused=True,
        )

    selected: list[tuple[float, str, ChunkRecord]] = []
    used_sources: set[tuple[str, int]] = set()
    for score, sent, chunk in scored_sentences:
        source_key = (chunk.file_name, chunk.page_number)
        if len(selected) >= 3:
            break
        if source_key in used_sources and score < best_score * 0.7:
            continue
        selected.append((score, sent, chunk))
        used_sources.add(source_key)

    answer_parts = [sent for _, sent, _ in selected]
    citations = []
    for _, _, chunk in selected:
        citation = chunk.citation()
        if citation not in citations:
            citations.append(citation)

    answer = " ".join(answer_parts).strip()
    if not answer.endswith((".", "!", "?")):
        answer += "."
    return AnswerResult(
        answer=answer,
        citations=citations,
        supporting_chunks=[chunk for _, _, chunk in selected],
        confidence=float(best_score),
        refused=False,
    )
