import os

from pydantic import BaseModel, Field

from src.retrieval.search import RetrievedChunk, search
from src.services.basic_rag import format_context
from src.services.llm import get_llm, structured_chain
from src.workflow.state import EngiState, Finding

MAX_CHUNKS = 8      # LLM ko is se zyada passages nahi dete (token limit)
K_PER_QUERY = 4

QUERY_SYSTEM = """You help search equipment manuals. Given a problem description, write 1 to 3 short
search queries (3-8 words each) likely to find troubleshooting, causes, inspection steps or limits
in an equipment manual. Use technical terms that would appear in a manual. Do not answer the problem."""

ASSESS_SYSTEM = """You are the investigator in an engineering diagnostic team.
Using ONLY the context passages, list the possible causes of the reported problem that the passages support.

Rules:
- Each finding needs: cause, evidence (brief paraphrase of the passage), citations (document and page
  exactly as in the passage headers) and confidence.
- confidence is high only if a passage explicitly links the cause to this problem, medium if indirect, low if weak.
- Never invent causes, numbers or pages. Do not use outside knowledge.
- If the passages support no cause, return an empty findings list.
- At most 5 findings."""


class QueryPlan(BaseModel):
    queries: list[str] = Field(description="1 to 3 short search queries")


class InvestigatorOutput(BaseModel):
    findings: list[Finding]


def retrieve(queries: list[str], vectorstore=None) -> list[RetrievedChunk]:
    """Retrieval tool: kayi queries chalata hai, duplicates hata kar behtareen chunks deta hai."""
    best: dict[tuple, RetrievedChunk] = {}
    for q in queries:
        for c in search(q, k=K_PER_QUERY, vectorstore=vectorstore):
            key = (c.document, c.page, c.text[:80])
            if key not in best or c.distance < best[key].distance:
                best[key] = c
    chunks = sorted(best.values(), key=lambda c: c.distance)
    max_distance = os.getenv("MAX_DISTANCE")  # optional filter, .env se
    if max_distance:
        chunks = [c for c in chunks if c.distance <= float(max_distance)]
    return chunks[:MAX_CHUNKS]


def investigator(state: EngiState, llm=None, vectorstore=None) -> dict:
    llm = llm or get_llm()
    problem = state["user_query"]

    plan = structured_chain(llm, QueryPlan, QUERY_SYSTEM, "Problem: {problem}").invoke({"problem": problem})
    queries = [problem] + [q for q in plan.queries if q.strip() and q.strip() != problem][:3]

    chunks = retrieve(queries, vectorstore)
    if not chunks:
        return {"retrieved_chunks": [], "findings": []}

    out = structured_chain(
        llm, InvestigatorOutput, ASSESS_SYSTEM, "Problem: {problem}\n\nContext passages:\n{context}"
    ).invoke({"problem": problem, "context": format_context(chunks)})

    # Code LLM par bharosa nahi karta: sirf wahi citations rakho jo asal mein retrieve hui hain.
    valid_pages = {(c.document, c.page) for c in chunks}
    findings: list[Finding] = []
    for f in out.findings:
        f.citations = [c for c in f.citations if (c.document, c.page) in valid_pages]
        if f.citations:  # bina valid saboot ke wajah report mein nahi jati
            f.id = f"F{len(findings) + 1}"
            findings.append(f)
    return {"retrieved_chunks": chunks, "findings": findings}
