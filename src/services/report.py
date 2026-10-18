"""Final report (Markdown). Koi LLM call nahi: sirf formatting, isliye mehngi/ghalat nahi hoti."""
from src.workflow.state import Citation, EngiState

SAFETY_NOTES = [
    "EngiGraph is decision support only. It does not control equipment and its output is not a verified procedure.",
    "Only qualified personnel should inspect or service this equipment, following the manufacturer's warnings and your site's lockout/tagout rules.",
    "Confirm every limit and value in the original manual page cited before acting on it.",
]


def _cite(citations: list[Citation]) -> str:
    return ", ".join(f"[{c.document}, page {c.page}]" for c in citations) or "(no citation)"


def build_report(state: EngiState) -> str:
    query = state.get("user_query", "")
    findings = state.get("findings", [])
    plan = state.get("diagnostic_plan")
    ver = state.get("verification_result")
    bad = {c.id: c.reason for c in ver.unsupported()} if ver else {}

    lines = ["# EngiGraph Diagnostic Report", "", f"**Reported problem:** {query}", ""]

    if not findings:
        lines += [
            "## Result",
            "Insufficient evidence: the indexed documents did not contain passages that support a cause "
            "for this problem. No diagnostic steps are suggested. Try rephrasing the problem, or add the "
            "relevant manual.",
            "",
        ]
    else:
        if ver and ver.passed:
            status = "All claims were checked against the retrieved passages and are supported."
        else:
            status = f"PARTIALLY VERIFIED: {len(bad)} claim(s) are not supported by the evidence and are marked below."
        revised = " One revision was performed after verification." if plan and plan.revision else ""
        lines += ["## Verification", status + revised, "", "## Findings (possible causes)"]
        for f in findings:
            flag = f" ⚠️ UNCERTAIN: {bad[f.id]}" if f.id in bad else ""
            lines += [
                f"- **{f.id}. {f.cause}** (confidence: {f.confidence}){flag}",
                f"  - Evidence: {f.evidence}",
                f"  - Source: {_cite(f.citations)}",
            ]

        lines += ["", "## Suggested diagnostic steps"]
        steps = plan.steps if plan else []
        if not steps:
            lines.append("No supported steps could be produced.")
        for s in steps:
            flag = f" ⚠️ UNCERTAIN: {bad[s.id]}" if s.id in bad else ""
            lines += [
                f"{s.order}. **{s.action}**{flag}",
                f"   - Why: {s.rationale}",
                f"   - Source: {_cite(s.citations)}",
            ]
            if s.safety_note:
                lines.append(f"   - ⚠️ Safety: {s.safety_note}")

        lines += ["", "## Uncertainty and limits"]
        lines.append("- Based only on the retrieved passages from the provided documents; your exact equipment may differ.")
        low = [f.id for f in findings if f.confidence == "low"]
        if low:
            lines.append(f"- Low-confidence findings: {', '.join(low)}.")
        for cid, reason in bad.items():
            lines.append(f"- {cid} unsupported: {reason}")

    lines += ["", "## Safety notes"] + [f"- {n}" for n in SAFETY_NOTES]
    return "\n".join(lines) + "\n"