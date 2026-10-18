"""Sawal se milte-julte chunks dhoondna, source metadata ke saath."""
from langchain_chroma import Chroma
from pydantic import BaseModel

from src.ingestion.indexer import get_vectorstore


class RetrievedChunk(BaseModel):
    document: str
    page: int
    text: str
    distance: float  # cosine distance: 0 = bilkul milta-julta, bada = kam milta


def search(query: str, k: int = 4, vectorstore: Chroma | None = None) -> list[RetrievedChunk]:
    if not query.strip():
        raise ValueError("Query khali nahi ho sakti.")
    store = vectorstore or get_vectorstore()
    results = store.similarity_search_with_score(query, k=k)
    return [
        RetrievedChunk(
            document=doc.metadata["document"],
            page=doc.metadata["page"],
            text=doc.page_content,
            distance=float(score),
        )
        for doc, score in results
    ]