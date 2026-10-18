"""Evaluation dataset: sawal load/save karna aur PDFs mein source pages dhoondna."""
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator

from src.ingestion.pdf_loader import PageText, load_pdf

QUESTIONS_PATH = Path("data/evaluation_questions.json")


def norm(text: str) -> str:
    """Lowercase + sirf letters/digits. Spaces, hyphens, degree signs, line breaks sab hat jate hain."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


class EvalQuestion(BaseModel):
    id: str
    type: Literal["direct", "multi", "conflict", "unanswerable"]
    question: str
    expected_answer: str
    # AND of ORs: har inner list ke koi bhi ek alternative ka jawab mein hona zaroori (norm() ke baad)
    required_facts: list[list[str]] = []
    source_document: str | None = None
    anchors: list[str] = []          # source text ke chhote tukde, page dhoondne ke liye
    source_pages: list[int] = []     # PDF viewer ke page numbers (locate command bharta hai)
    printed_pages: list[str] = []    # manual ke andar likhe page numbers (sirf note)
    answerable: bool

    @model_validator(mode="after")
    def _check(self):
        if self.answerable != (self.type != "unanswerable"):
            raise ValueError(f"{self.id}: answerable aur type match nahi karte")
        if self.answerable:
            if not self.required_facts or not all(self.required_facts):
                raise ValueError(f"{self.id}: required_facts chahiye (har group khali nahi)")
            if not self.source_document or not self.anchors:
                raise ValueError(f"{self.id}: source_document aur anchors chahiye")
        elif self.required_facts or self.source_document or self.anchors:
            raise ValueError(f"{self.id}: unanswerable sawal mein facts/source nahi hone chahiye")
        return self


def load_questions(path: str | Path = QUESTIONS_PATH) -> list[EvalQuestion]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    questions = [EvalQuestion(**item) for item in data]
    ids = [q.id for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate question ids")
    return questions


def save_questions(questions: list[EvalQuestion], path: str | Path = QUESTIONS_PATH) -> None:
    """Ek sawal = ek line (diff aur editing aasan)."""
    lines = ",\n".join("  " + json.dumps(q.model_dump(), ensure_ascii=False) for q in questions)
    Path(path).write_text("[\n" + lines + "\n]\n", encoding="utf-8")


def find_pages(pages: list[PageText], text: str) -> list[int]:
    target = norm(text)
    return [p.page for p in pages if target and target in norm(p.text)]


def locate(questions: list[EvalQuestion], pdf_dir: str | Path, overwrite: bool = False) -> list[dict]:
    """Anchors se source_pages bharta hai aur check karta hai ke required facts document mein hain.

    Yeh sirf madad hai: README ka rule hai ke har page aur jawab khud verify karo.
    """
    cache: dict[str, list[PageText]] = {}
    report: list[dict] = []
    for q in questions:
        if not q.answerable:
            continue
        path = Path(pdf_dir) / q.source_document
        if not path.exists():
            report.append({"id": q.id, "item": q.source_document, "pages": [], "status": "PDF MISSING"})
            continue
        if q.source_document not in cache:
            cache[q.source_document] = load_pdf(path)
        pages = cache[q.source_document]

        found: set[int] = set()
        for anchor in q.anchors:
            hit = find_pages(pages, anchor)
            report.append({"id": q.id, "item": anchor, "pages": hit, "status": "ok" if hit else "ANCHOR NOT FOUND"})
            found.update(hit)

        full_text = norm(" ".join(p.text for p in pages))
        for group in q.required_facts:
            if not any(norm(alt) in full_text for alt in group):
                report.append({"id": q.id, "item": f"fact {group}", "pages": [], "status": "FACT NOT IN DOCUMENT"})

        if found and (overwrite or not q.source_pages):
            q.source_pages = sorted(found)
    return report