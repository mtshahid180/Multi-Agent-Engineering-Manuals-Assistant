from pydantic import BaseModel

from src.services.basic_rag import format_context
from src.services.llm import get_llm, structured_chain
from src.workflow.state import DiagnosticPlan, DiagnosticStep, EngiState

SYSTEM = """You are the planner in an engineering diagnostic team.
Turn the findings into an ordered list of diagnostic checks for a qualified technician.

Rules:
- Use ONLY the findings and context passages. Never invent values, limits, tools or procedures.
- Every step needs citations (document and page exactly as in the passage headers).
- Order steps from least invasive and safest to most invasive.
- Steps are checks and inspections only. Never tell anyone to start, stop, or command equipment remotely.
- Fill safety_note for any step involving electrical parts, rotating parts, hot surfaces or lubrication; otherwise leave it empty.
- If verifier feedback is given, fix it: remove or rewrite every step listed as unsupported, and do not rely on
  unsupported findings. Do not add new unsupported claims.
- At most 6 steps."""

HUMAN = """Problem: {problem}

Findings:
{findings}

Context passages:
{context}

Verifier feedback (empty on the first attempt):
{feedback}"""


class PlannerOutput(BaseModel):
    steps: list[DiagnosticStep]


def _findings_text(findings) -> str:
    return "\n".join(
        f"{f.id}: {f.cause} | evidence: {f.evidence} | confidence: {f.confidence} | "
        + ", ".join(f"[{c.document}, page {c.page}]" for c in f.citations)
        for f in findings
    )


def _feedback_text(verification) -> str:
    if not verification or verification.passed:
        return ""
    return "\n".join(f"{c.id}: UNSUPPORTED - {c.reason}" for c in verification.unsupported())


def planner(state: EngiState, llm=None) -> dict:
    llm = llm or get_llm()
    feedback = _feedback_text(state.get("verification_result"))
    previous = state.get("diagnostic_plan")
    revision = (previous.revision + 1) if (previous and feedback) else 0

    out = structured_chain(llm, PlannerOutput, SYSTEM, HUMAN).invoke(
        {
            "problem": state["user_query"],
            "findings": _findings_text(state["findings"]),
            "context": format_context(state["retrieved_chunks"]),
            "feedback": feedback or "(none)",
        }
    )

    valid_pages = {(c.document, c.page) for c in state["retrieved_chunks"]}
    steps = []
    for i, s in enumerate(out.steps, start=1):
        s.id, s.order = f"S{i}", i
        s.citations = [c for c in s.citations if (c.document, c.page) in valid_pages]
        steps.append(s)  # bina citation wala step bhi rehta hai; verifier use flag karega
    return {"diagnostic_plan": DiagnosticPlan(steps=steps, revision=revision)}
