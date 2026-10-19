import os
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from src.ingestion.chunker import chunk_pages
from src.ingestion.pdf_loader import load_pdf
from src.services.llm import get_embeddings

COLLECTION = "manuals"


def get_vectorstore(embeddings: Embeddings | None = None, persist_dir: str | None = None) -> Chroma:
    """Persistent ChromaDB kholta hai (data/chroma_db). Distance = cosine."""
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=persist_dir or os.getenv("CHROMA_DIR", "data/chroma_db"),
        collection_metadata={"hnsw:space": "cosine"},
    )


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg


def _add_with_retry(vectorstore: Chroma, batch, retries: int = 5) -> None:
    for attempt in range(1, retries + 1):
        try:
            vectorstore.add_documents(batch, ids=[d.metadata["chunk_id"] for d in batch])
            return
        except Exception as exc:
            if attempt == retries:
                raise
            # Rate limit minute-wise reset hoti hai, isliye pura 60s rukte hain
            wait = 60 if _is_rate_limit(exc) else 5 * attempt
            print(f"  API error ({exc.__class__.__name__}), {wait}s baad dobara try ({attempt}/{retries})...")
            time.sleep(wait)


def index_pdf(
    pdf_path: str | Path,
    vectorstore: Chroma | None = None,
    batch_size: int = 20,
    max_per_minute: int | None = 80,
    force: bool = False,
) -> int:
    """PDF -> pages -> chunks -> embeddings -> ChromaDB. Document ke total chunks return karta hai.

    - max_per_minute: free tier ki limit (100/min) se bachne ke liye. None = koi throttle nahi.
    - Resume: jo chunks pehle se saved hain wo dobara embed nahi hote.
    - force=True: us document ke purane chunks hata kar sab dobara banata hai (PDF badli ho to).
    """
    pdf_path = Path(pdf_path)
    chunks = chunk_pages(load_pdf(pdf_path))
    if not chunks:
        raise ValueError(f"{pdf_path.name} mein text nahi mila (scanned PDF?).")

    store = vectorstore or get_vectorstore()

    existing = set(store.get(where={"document": pdf_path.name})["ids"])
    if force and existing:
        store.delete(ids=list(existing))
        existing = set()

    todo = [c for c in chunks if c.metadata["chunk_id"] not in existing]
    if existing:
        print(f"  {len(existing)} chunks pehle se saved, {len(todo)} baaki.")

    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        t0 = time.monotonic()
        _add_with_retry(store, batch)
        print(f"  {min(start + batch_size, len(todo))}/{len(todo)} chunks done")

        is_last = start + batch_size >= len(todo)
        if max_per_minute and not is_last:
            min_duration = len(batch) / max_per_minute * 60
            wait = min_duration - (time.monotonic() - t0)
            if wait > 0:
                time.sleep(wait)
    return len(chunks)
