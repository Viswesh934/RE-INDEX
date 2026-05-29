from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from .answer import answer_from_hits
from .retrieve import Retriever
from .storage import LocalStore


DEFAULT_EVAL_SET = [
    {
        "question": "What is the main purpose of the document?",
        "expected_sources": []
    },
    {
        "question": "Which page mentions the refund process?",
        "expected_sources": []
    }
]


@dataclass
class EvalCase:
    question: str
    expected_sources: list[str]


@dataclass
class EvalResult:
    question: str
    answer: str
    citations: list[str]
    hit: bool
    refused: bool


def load_eval_set(path: str | Path | None = None) -> list[EvalCase]:
    if path is None:
        return [EvalCase(**item) for item in DEFAULT_EVAL_SET]
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvalCase(question=item["question"], expected_sources=item.get("expected_sources", [])) for item in data]


def run_eval(retriever: Retriever, eval_cases: list[EvalCase]) -> list[EvalResult]:
    results: list[EvalResult] = []
    for case in eval_cases:
        retrieval = retriever.retrieve(case.question, top_k=5, use_keyword_search=True)
        answer = answer_from_hits(case.question, retrieval.hits)
        hit = False
        expected = {s.lower() for s in case.expected_sources}
        if expected:
            hit = any(c.lower() in expected for c in answer.citations)
        results.append(
            EvalResult(
                question=case.question,
                answer=answer.answer,
                citations=answer.citations,
                hit=hit,
                refused=answer.refused,
            )
        )
    return results
