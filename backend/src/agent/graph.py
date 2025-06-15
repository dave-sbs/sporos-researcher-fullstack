"""LangGraph assembly for agent2.

This mirrors the control-flow in *sporos-researcher.py* but pulls node
functions from :pymod:`agent2.nodes` and data models from
:pymod:`agent2.models`.
"""
from __future__ import annotations

from langgraph.graph import StateGraph
from langgraph.constants import Send

from agent.state import ResearchGraphState, FilterResult, ReconstructedBill
from agent.nodes import (
    preprocess_input,
    compile_final_research,
    extract_filters,
    grade_documents,
    reconstruct_full_text,
    retrieve_documents,
    summarize_bills,
)
from typing import List


def initiate_parallel_summaries(state: ResearchGraphState) -> List[Send]:
    """Prepare and dispatch bills for parallel summarization."""
    # This node itself doesn't have a "loading" state in node_status,
    # but it triggers the start of the "summarize_bills" phase.
    # The frontend can infer "summarize_bills" is loading when it sees the first Send event.
    
    reconstructed_bills = state.get("reconstructed_bills", [])
    enhanced_query = state["enhanced_query"] # Pass original query for summarization context
    
    sends = []
    if not reconstructed_bills:
        # If no bills, we might need to send a signal to skip summarization or directly to compilation.
        # For now, LangGraph will simply not call "summarize_bills" if sends is empty.
        # The "compile_final_research" node needs to handle an empty "bill_summaries" list.
        print("No reconstructed bills to summarize.")
        # To ensure the graph progresses if this is a valid terminal state for summarization:
        # return [Send("compile_final_research", {})] # Or whatever the next node is if summarization is skippable.
        # For now, assume compile_final_research handles empty summaries.
        pass # No sends, graph will proceed based on edges from reconstruct_full_text

    for bill in reconstructed_bills:
        sends.append(Send("summarize_bills", {
            # Pass only necessary parts of the bill to avoid large state objects per branch
            "bill_to_summarize": {
                "bill_id": bill["id"],
                "title": bill["title"],
                "full_text": bill["full_text"]
                # Potentially other metadata if useful for summary prompt
            },
            "enhanced_query": enhanced_query 
        }))
    return sends

def initiate_parallel_grading(state: ResearchGraphState) -> List[Send]:
    """Prepare and dispatch bills for parallel grading."""
    graded_docs = state.get("graded_docs", [])
    enhanced_query = state["enhanced_query"]
    sends = []
    if not graded_docs:
        print("No graded documents to grade.")
        pass # No sends, graph will proceed based on edges from grade_documents

    for doc in graded_docs:
        sends.append(Send("grade_documents", {
            "graded_doc": doc,
            "enhanced_query": enhanced_query 
        }))
    return sends

# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_graph():
    g = StateGraph(ResearchGraphState)

    g.add_node("preprocess_input", preprocess_input)
    g.add_node("extract_filters", extract_filters)
    g.add_node("retrieve_documents", retrieve_documents)
    g.add_node("grade_documents", grade_documents)
    g.add_node("reconstruct_full_text", reconstruct_full_text)
    g.add_node("summarize_bills", summarize_bills)
    g.add_node("compile_final_research", compile_final_research)

    # Linear edges
    g.set_entry_point("preprocess_input")
    g.add_edge("preprocess_input", "extract_filters")
    g.add_edge("extract_filters", "retrieve_documents")
    g.add_edge("retrieve_documents", "grade_documents")
    g.add_edge("grade_documents", "reconstruct_full_text")

    # g.add_conditional_edges(
    #     "retrieve_documents", initiate_parallel_grading, ["reconstruct_full_text"]
    # )

    # Conditional parallel fan-out
    g.add_conditional_edges(
        "reconstruct_full_text", initiate_parallel_summaries, ["summarize_bills"]
    )

    g.add_edge("summarize_bills", "compile_final_research")
    g.set_finish_point("compile_final_research")

    return g.compile(name="agent2-research-graph")


# Singleton compiled graph
graph = _build_graph()
