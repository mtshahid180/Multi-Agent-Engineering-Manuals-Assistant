"""LangGraph: nodes (agents), edges (tarteeb), conditional routing (faislay)."""
from langgraph.graph import END, START, StateGraph

from src.agents.investigator import investigator
from src.agents.planner import planner
from src.agents.verifier import verifier
from src.services.llm import get_llm
from src.services.report import build_report
from src.workflow.state import EngiState

MAX_REVISIONS = 1  # README: sirf ek revision


def route_after_investigator(state: EngiState) -> str:
    """Saboot hi nahi mila to planner/verifier par paise zaya na karo."""
    return "planner" if state.get("findings") else "report"


def route_after_verifier(state: EngiState) -> str:
    result, plan = state["verification_result"], state["diagnostic_plan"]
    if not result.passed and plan.revision < MAX_REVISIONS:
        return "planner"
    return "report"


def build_graph(llm=None, vectorstore=None):
    llm = llm or get_llm()

    def investigator_node(state: EngiState) -> dict:
        return investigator(state, llm=llm, vectorstore=vectorstore)

    def planner_node(state: EngiState) -> dict:
        return planner(state, llm=llm)

    def verifier_node(state: EngiState) -> dict:
        return verifier(state, llm=llm)

    def report_node(state: EngiState) -> dict:
        return {"final_report": build_report(state)}

    g = StateGraph(EngiState)
    g.add_node("investigator", investigator_node)
    g.add_node("planner", planner_node)
    g.add_node("verifier", verifier_node)
    g.add_node("report", report_node)

    g.add_edge(START, "investigator")
    g.add_conditional_edges("investigator", route_after_investigator, {"planner": "planner", "report": "report"})
    g.add_edge("planner", "verifier")
    g.add_conditional_edges("verifier", route_after_verifier, {"planner": "planner", "report": "report"})
    g.add_edge("report", END)
    return g.compile()


def run_workflow(query: str, llm=None, vectorstore=None) -> EngiState:
    return build_graph(llm=llm, vectorstore=vectorstore).invoke({"user_query": query})