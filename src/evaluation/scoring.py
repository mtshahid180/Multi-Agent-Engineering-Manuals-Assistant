"""Scoring: deterministic (bina LLM-judge ke), isliye reproducible aur muft."""
import re
from dataclasses import dataclass, field

from src.evaluation.dataset import EvalQuestion, norm

CITATION_RE = re.compile(r"\[([^\[\]]+?),\s*pages?\s*(\d+)\s*\]", re.IGNORECASE)
NOT_FOUND_KEY = norm("Not found in the provided documents")


@dataclass
class SystemOutput:
    answer_text: str
    citations: list[tuple[str, int]] = field(default_factory=list)  # jawab mein di gayi (doc, page)
    retrieved: list[tuple[str, int]] = field(default_factory=list)  # asal mein retrieve hui (doc, page)
    abstained: bool = False
    extra: dict = field(default_factory=dict)


def strip_citations(text: str) -> str:
    """[doc, page 13] hata do, warna 'page13' se "13" wala fact ghalat match ho jata hai."""
    return re.sub(r"\[[^\[\]]*\]", " ", text)


def extract_citations(text: str) -> list[tuple[str, int]]:
    return [(doc.strip(), int(page)) for doc, page in CITATION_RE.findall(text)]


def is_abstention_text(answer: str) -> bool:
    return NOT_FOUND_KEY in norm(answer)


def fact_hits(answer: str, required_facts: list[list[str]]) -> tuple[int, int]:
    text = norm(strip_citations(answer))
    hits = sum(any(norm(alt) in text for alt in group) for group in required_facts)
    return hits, len(required_facts)


def score_output(q: EvalQuestion, out: SystemOutput) -> dict:
    cites = list(dict.fromkeys(out.citations))  # duplicates hatao, tarteeb rakho
    valid = None
    if cites:
        retrieved = set(out.retrieved)
        valid = sum(c in retrieved for c in cites) / len(cites)

    row = {
        "abstained": out.abstained,
        "n_citations": len(cites),
        "citation_valid_rate": valid,   # kitni citations asal mein retrieve hui thi (hallucination check)
        "fact_hits": None, "fact_total": None, "fact_score": None,
        "cites_gold": None,             # sahi page cite hua? (None = source_pages abhi bhare nahi)
    }
    if not q.answerable:
        row["correct"] = out.abstained  # unanswerable mein sahi = "pata nahi" kehna
        return row

    hits, total = fact_hits(out.answer_text, q.required_facts)
    row.update(fact_hits=hits, fact_total=total, fact_score=hits / total)
    row["correct"] = hits == total and not out.abstained
    if q.source_pages:
        gold = {(q.source_document, p) for p in q.source_pages}
        row["cites_gold"] = any(c in gold for c in cites)
    return row