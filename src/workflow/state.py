"""Shared state: wo data jo agents ke beech guzarta hai.

README ka rule: state mein sirf yeh 6 keys hongi.
"""
from typing import Literal, TypedDict

from pydantic import BaseModel, Field

from src.retrieval.search import RetrievedChunk


class Citation(BaseModel):
    document: str
    page: int


class Finding(BaseModel):
    """Investigator ka nateeja: ek mumkina wajah + uska saboot."""
    id: str = ""  # code assign karta hai (F1, F2...), LLM nahi
    cause: str = Field(description="Possible cause of the reported problem")
    evidence: str = Field(description="Brief paraphrase of what the passage says")
    citations: list[Citation] = Field(description="Document and page exactly as in the passage headers")
    confidence: Literal["high", "medium", "low"]


class DiagnosticStep(BaseModel):
    id: str = ""  # S1, S2...
    order: int = 0
    action: str = Field(description="One concrete check or inspection to perform")
    rationale: str = Field(description="Why this step, tied to a finding")
    citations: list[Citation]
    safety_note: str = Field(default="", description="Safety warning for this step, or empty")


class DiagnosticPlan(BaseModel):
    steps: list[DiagnosticStep] = []
    revision: int = 0  # 0 = pehla draft, 1 = verifier ke baad ek revision


class ClaimCheck(BaseModel):
    id: str = Field(description="Claim id such as F1 or S2")
    supported: bool
    reason: str = Field(description="One short sentence")


class VerificationResult(BaseModel):
    passed: bool
    checks: list[ClaimCheck] = []

    def unsupported(self) -> list[ClaimCheck]:
        return [c for c in self.checks if not c.supported]


class EngiState(TypedDict, total=False):
    user_query: str
    retrieved_chunks: list[RetrievedChunk]
    findings: list[Finding]
    diagnostic_plan: DiagnosticPlan | None
    verification_result: VerificationResult | None
    final_report: str