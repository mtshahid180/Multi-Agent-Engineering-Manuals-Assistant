"""PDF se page-by-page text nikalna, page numbers ke saath."""
from pathlib import Path

import pymupdf  # PyMuPDF
from pydantic import BaseModel


class PageText(BaseModel):
    document: str  # e.g. "motor_manual.pdf"
    page: int      # 1-based page number
    text: str

def load_pdf(path: str | Path) -> list[PageText]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    pages: list[PageText] = []
    with pymupdf.open(path) as doc:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append(PageText(document=path.name, page=page_number, text=text))
    return pages