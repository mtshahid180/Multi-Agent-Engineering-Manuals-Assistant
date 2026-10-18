"""Baseline RAG: search -> context -> LLM -> cited answer.

Yeh multi-agent system ka muqabla karne ke liye "basic" version hai (Days 17-19).
"""
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from src.retrieval.search import RetrievedChunk, search
from src.services.llm import get_llm

SYSTEM_PROMPT = """You are an engineering documentation assistant.
Answer ONLY from the context passages below. Never use outside knowledge.

Rules:
- After every claim, cite its source as [document, page N], exactly as shown in the passage header.
- If the context does not contain the answer, reply exactly: "Not found in the provided documents."
- If the context only partly answers, say what is missing and mark it as uncertain.
- You are decision support only. Never present a procedure as verified safe; remind the reader to
  follow the manual's warnings and qualified-personnel requirements when giving procedures."""

HUMAN_PROMPT = "Context passages:\n\n{context}\n\nQuestion: {question}"

NOT_FOUND = "Not found in the provided documents."


def format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{c.document}, page {c.page}]\n{c.text}" for c in chunks)

def answer_question(question: str, k: int = 4, vectorstore: Chroma | None = None, llm=None, config=None) -> dict:
    chunks = search(question, k=k, vectorstore=vectorstore)
    if not chunks:  # kuch mila hi nahi -> LLM ko bulane ki zarurat nahi
        return {"answer": NOT_FOUND, "sources": []}

    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])
    chain = prompt | (llm or get_llm())  # LCEL: prompt ka output seedha LLM mein
    reply = chain.invoke({"context": format_context(chunks), "question": question}, config=config)
    return {"answer": reply.content, "sources": chunks}