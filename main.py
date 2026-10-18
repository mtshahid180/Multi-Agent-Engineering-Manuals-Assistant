"""EngiGraph command-line.

    python main.py llm
    python main.py pdf  data/raw_pdfs/file.pdf
    python main.py index [path] [--force]
    python main.py search "bearing temperature"
    python main.py ask "What is the maximum bearing temperature?"
    python main.py node investigator "motor bearing is overheating"   # ek node ka test
    python main.py node planner "motor bearing is overheating"        # investigator + planner
    python main.py node verifier "motor bearing is overheating"       # investigator + planner + verifier
    python main.py diagnose "motor bearing is overheating"            # poora LangGraph workflow
"""
import argparse
from pathlib import Path

from src.agents.investigator import investigator
from src.agents.planner import planner
from src.agents.verifier import verifier
from src.ingestion.indexer import index_pdf
from src.ingestion.pdf_loader import load_pdf
from src.retrieval.search import search
from src.services.basic_rag import answer_question
from src.services.llm import get_llm
from src.workflow.graph import run_workflow

RAW_DIR = Path("data/raw_pdfs")
REPORT_PATH = Path("reports/last_report.md")


def cmd_llm(_args) -> None:
    reply = get_llm().invoke("In one sentence, what is a bearing in a motor?")
    print("LLM reply:\n", reply.content)


def cmd_pdf(args) -> None:
    pages = load_pdf(args.path)
    print(f"Extracted {len(pages)} non-empty pages from {args.path}")
    for p in pages[:3]:
        print(f"\n--- {p.document} | page {p.page} ---\n{p.text[:400]}")


def cmd_index(args) -> None:
    pdfs = [Path(args.path)] if args.path else sorted(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        print("Koi PDF nahi mili. data/raw_pdfs/ mein PDFs rakho.")
        return
    for pdf in pdfs:
        print(f"Indexing {pdf.name} ...")
        print(f"  -> {index_pdf(pdf, force=args.force)} chunks total")


def cmd_search(args) -> None:
    results = search(args.query, k=args.k)
    if not results:
        print("Kuch nahi mila. Pehle `python main.py index` chalao.")
    for i, r in enumerate(results, 1):
        print(f"\n#{i}  {r.document} | page {r.page} | distance {r.distance:.3f}")
        print(r.text[:300].replace("\n", " "))


def cmd_ask(args) -> None:
    result = answer_question(args.question, k=args.k)
    print("\nANSWER:\n", result["answer"])
    print("\nSOURCES (retrieved):")
    for c in result["sources"]:
        print(f"  - {c.document}, page {c.page} (distance {c.distance:.3f})")


def _show(title: str, items) -> None:
    print(f"\n===== {title} =====")
    if not items:
        print("(empty)")
    for it in items if isinstance(items, list) else [items]:
        print(it.model_dump_json(indent=2))


def cmd_node(args) -> None:
    """Har node ko alag chalao (README: pehle node-by-node test, phir poora graph)."""
    llm = get_llm()
    state = {"user_query": args.query}

    state.update(investigator(state, llm=llm))
    print(f"Retrieved {len(state['retrieved_chunks'])} chunks:")
    for c in state["retrieved_chunks"]:
        print(f"  - {c.document}, page {c.page} (distance {c.distance:.3f})")
    _show("FINDINGS", state["findings"])
    if args.node == "investigator" or not state["findings"]:
        return

    state.update(planner(state, llm=llm))
    _show("DIAGNOSTIC PLAN", state["diagnostic_plan"])
    if args.node == "planner":
        return

    state.update(verifier(state, llm=llm))
    _show("VERIFICATION", state["verification_result"])


def cmd_diagnose(args) -> None:
    state = run_workflow(args.query)
    print(state["final_report"])
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(state["final_report"], encoding="utf-8")
    print(f"(Report saved: {REPORT_PATH})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("llm").set_defaults(func=cmd_llm)

    p = sub.add_parser("pdf"); p.add_argument("path"); p.set_defaults(func=cmd_pdf)

    p = sub.add_parser("index"); p.add_argument("path", nargs="?")
    p.add_argument("--force", action="store_true", help="purane chunks hata kar dobara index karo")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search"); p.add_argument("query"); p.add_argument("-k", type=int, default=4)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("ask"); p.add_argument("question"); p.add_argument("-k", type=int, default=4)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("node")
    p.add_argument("node", choices=["investigator", "planner", "verifier"])
    p.add_argument("query"); p.set_defaults(func=cmd_node)

    p = sub.add_parser("diagnose"); p.add_argument("query"); p.set_defaults(func=cmd_diagnose)

    args = parser.parse_args()
    args.func(args)