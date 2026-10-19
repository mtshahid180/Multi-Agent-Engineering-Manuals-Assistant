from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.pdf_loader import PageText

CHUNK_SIZE = 1000     # characters per chunk
CHUNK_OVERLAP = 150   # agle chunk mein pichhle ka thora hissa (context na toote)


def chunk_pages(
    pages: list[PageText],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Har page alag split hota hai, isliye har chunk ka page number sahi rehta hai."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],  # pehle paragraph, phir line, phir sentence
    )
    chunks: list[Document] = []
    for page in pages:
        for i, piece in enumerate(splitter.split_text(page.text)):
            chunks.append(
                Document(
                    page_content=piece,
                    metadata={
                        "document": page.document,
                        "page": page.page,
                        "chunk_id": f"{page.document}:p{page.page}:c{i}",
                    },
                )
            )
    return chunks
