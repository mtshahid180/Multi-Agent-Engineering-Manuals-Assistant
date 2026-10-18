"""Verifier: har claim ko saboot se check karta hai aur revision ka faisla deta hai."""
from pydantic import BaseModel

from src.services.basic_rag import format_context
from src.services.llm import get_llm, structured_chain
from src.workflow.state import ClaimCheck, EngiState, VerificationResult

SYSTEM = """You are a strict verifier in an engineering diagnostic team.
For each claim, decide whether the cited passages support it.

Rules:
- supported = true only if the passage text states the claim or directly implies it.
- Numbers, limits and units must match the passage exactly.
- If the claim adds details that are not in the passages, supported = false.
- Give one short reason per claim.
- Return exactly one check for every claim id."""

HUMAN = "Claims:\n{claims}\n\nContext passages:\n{context}"


class VerifierOutput(BaseModel):
    checks: list[ClaimCheck]


def _cites(citations) -> str:
    return "; ".join(f"{c.document} p.{c.page}" for c in citations)


def verifier(state: EngiState, llm=None) -> dict:
    chunks = state["retrieved_chunks"]
    valid_pages = {(c.document, c.page) for c in chunks}

    claims: dict[str, str] = {}
    for f in state.get("findings", []):
        claims[f.id] = f"{f.id}: {f.cause}. Evidence: {f.evidence} (cites: {_cites(f.citations)})"
    plan = state.get("diagnostic_plan")
    for s in (plan.steps if plan else []):
        claims[s.id] = f"{s.id}: {s.action}. Rationale: {s.rationale} (cites: {_cites(s.citations)})"

    # Step 1 (code): citation hi nahi / galat hai -> LLM ke bina unsupported
    checks: dict[str, ClaimCheck] = {}
    cited: dict[str, list] = {f.id: f.citations for f in state.get("findings", [])}
    cited.update({s.id: s.citations for s in (plan.steps if plan else [])})
    for cid, cites in cited.items():
        if not cites or any((c.document, c.page) not in valid_pages for c in cites):
            checks[cid] = ClaimCheck(id=cid, supported=False, reason="No valid citation to retrieved evidence.")

    # Step 2 (LLM): baaki claims ka matlab saboot se milta hai ya nahi
    to_check = {cid: text for cid, text in claims.items() if cid not in checks}
    if to_check:
        llm = llm or get_llm()
        out = structured_chain(llm, VerifierOutput, SYSTEM, HUMAN).invoke(
            {"claims": "\n".join(to_check.values()), "context": format_context(chunks)}
        )
        for c in out.checks:
            if c.id in to_check:
                checks[c.id] = c
        for cid in to_check:  # LLM ne kisi claim ko skip kiya -> unsupported maano
            checks.setdefault(cid, ClaimCheck(id=cid, supported=False, reason="Not evaluated by verifier."))

    ordered = [checks[cid] for cid in claims]
    return {
        "verification_result": VerificationResult(passed=all(c.supported for c in ordered), checks=ordered)
    }