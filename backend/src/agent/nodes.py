"""All LangGraph node functions for agent2.

They operate purely on and return `ResearchGraphState` patches. Heavy lifting
(like DB calls, LLM selection) is delegated to helper modules so these remain
thin and testable.
"""
from __future__ import annotations

import operator
from typing import Any, Dict, List, Tuple

from langchain_core.messages import SystemMessage, AIMessage
from langgraph.constants import Send
from langchain_core.runnables import RunnableConfig

from agent.configuration import get_llm, get_supabase_client
from agent.retrieval import retriever
from agent.prompts import (
    enhance_query_instructions,
    extract_filters_instructions,
    grade_documents_instructions,
    summarize_bills_instructions,
    compile_final_report_instructions,
    get_current_date,
)
from agent.state import ResearchGraphState, FilterResult, ReconstructedBill, BillSummary
from agent.tools_and_schemas import DocumentGrades, BillSummaryLLM
from agent.utils import get_research_topic
 

def preprocess_input(state: ResearchGraphState, config: RunnableConfig) -> ResearchGraphState:
    """Transform the original query into an enhanced query"""
    original_query = get_research_topic(state["messages"])
    print("original_query", original_query)
    enhanced_query = get_llm("gpt-4o-mini").invoke([SystemMessage(content=enhance_query_instructions.format(user_query=original_query, current_date=get_current_date))]).content
    return {"enhanced_query": enhanced_query}
    
# ---------------------------------------------------------------------------
# 1. Extract filters
# ---------------------------------------------------------------------------

def extract_filters(state: ResearchGraphState, config: RunnableConfig) -> ResearchGraphState:
    enhanced_query = state["enhanced_query"]
    llm = get_llm("gpt-4o-mini").with_structured_output(FilterResult)
    instructions = extract_filters_instructions.format(user_query=enhanced_query, current_date=get_current_date)
    try: 
        result = llm.invoke([SystemMessage(content=instructions)])
        return {"filters": result}
    except Exception as e:
        print("extract_filters error", e)
        return {"filters": None}


# ---------------------------------------------------------------------------
# 2. Retrieve documents
# ---------------------------------------------------------------------------

def retrieve_documents(state: ResearchGraphState, config: RunnableConfig) -> ResearchGraphState:
    enhanced_query = state["enhanced_query"]
    filters = state.get("filters")

    filter_kwargs: Dict[str, Any] = {}
    if filters and filters.state:
        filter_kwargs["state"] = filters.state

    docs = retriever.invoke({"query": enhanced_query, "k": 8, "filters": filter_kwargs or None})
    return {"retrieved_docs": docs}


# ---------------------------------------------------------------------------
# 3. Grade documents
# ---------------------------------------------------------------------------
def grade_documents(state: ResearchGraphState, config: RunnableConfig) -> ResearchGraphState:
    enhanced_query = state["enhanced_query"]
    retrieved_docs = state.get("retrieved_docs", [])
    # print(f"state: {state}")
    if not retrieved_docs:
        return {"graded_docs": []}

    snippets = []
    for idx, (doc, score) in enumerate(retrieved_docs):
        snippets.append(
            f"Index: {idx}\nTitle: {doc.metadata.get('title')}\nSnippet: {doc.page_content[:500]}\nScore: {score:.3f}\n---"
        )
    context = "\n".join(snippets)
    # print(f"context: {context}")

    prompt = grade_documents_instructions.format(
        user_query=enhanced_query,
        doc_context=context,
    )
    grades = get_llm("gpt-4o-mini").with_structured_output(DocumentGrades).invoke([SystemMessage(content=prompt)])
    # print(f"grades: {grades}")

    # Create a list of graded documents with full metadata
    graded_docs = []
    # print(f"grades: {grades}")
    # print(f"grades.grades: {grades.grades}")
    for grade in grades.grades:
        print(f"grade: {grade}")
        print(f"grade.is_relevant: {grade.is_relevant}")
        if grade.is_relevant:
            doc, score = retrieved_docs[grade.doc_index]
            graded_docs.append({
                "doc": doc,
                "score": score,
                "is_relevant": grade.is_relevant,
                "reasoning": grade.reasoning,
                "doc_index": grade.doc_index,
                "title": doc.metadata.get("title", "N/A")
            })
    
    print(f"graded_docs: {graded_docs}")

    return {"graded_docs": graded_docs}


# ---------------------------------------------------------------------------
# 4. Reconstruct full bill text
# ---------------------------------------------------------------------------

def reconstruct_full_text(state: ResearchGraphState, config: RunnableConfig) -> ResearchGraphState:
    graded_docs = state.get("graded_docs", [])
    if not graded_docs:
        return {"reconstructed_bills": []}

    sb = get_supabase_client()
    bills: List[ReconstructedBill] = []
    for gd in graded_docs:
        doc = gd["doc"]
        bill_id = doc.metadata.get("bill_id")
        if not bill_id:
            continue
        res = sb.table("chunks_test2").select("*").eq("bill_id", bill_id).order("chunk_idx").execute()
        full_text = "".join(chunk.get("chunk_text", "") for chunk in res.data)
        bills.append(
            {
                "id": bill_id,
                "bill_identifier": doc.metadata.get("bill_identifier", "N/A"),
                "year": doc.metadata.get("year", 0),
                "state": doc.metadata.get("state", "N/A"),
                "title": doc.metadata.get("title", "N/A"),
                "similarity_score": gd["score"],
                "status": doc.metadata.get("status", []),
                "full_text": full_text,
            }
        )
    return {"reconstructed_bills": bills}


# ---------------------------------------------------------------------------
# 6. Summarize a bill (runs in parallel)
# ---------------------------------------------------------------------------

def summarize_bills(state: ResearchGraphState, config: RunnableConfig) -> ResearchGraphState:
    bill = state["bill_to_summarize"] # This comes from the Send payload
    prompt = summarize_bills_instructions.format(
        user_query=state["enhanced_query"],
        title=bill["title"],
        truncated_text=bill["full_text"][:10000],
    )
    llm = get_llm("gpt-4o-mini").with_structured_output(BillSummaryLLM)
    summary = llm.invoke([SystemMessage(content=prompt)])
    try:
        summary_text = summary
        print(summary_text)
        
        bill_summary_output: BillSummary = {
            "bill_id": bill["bill_id"],
            "title": bill["title"],
            "summary_text": summary_text,
            "one_line_summary": ""
        }
        return {"bill_summaries": [bill_summary_output]} # operator.add appends this list
    except Exception as e:
        print(f"Error summarizing bill {bill['bill_id']}: {e}")
        error_summary: BillSummary = {
            "bill_id": bill["bill_id"],
            "title": bill["title"],
            "summary_text": "Error during summarization.",
            "one_line_summary": "",
            "error_message": str(e)
        }
    return {"bill_summaries": [error_summary]}


# ---------------------------------------------------------------------------
# 7. Compile final research
# ---------------------------------------------------------------------------

def compile_final_research(state: ResearchGraphState, config: RunnableConfig) -> ResearchGraphState:
    summaries = state.get("bill_summaries", [])
    if not summaries:
        return {"final_research": "No relevant bill summaries were generated."}

    joined = []
    for bs in summaries:
        joined.append(f"Title: {bs['title']}\nSummary:\n{bs['summary_text']}\n---")
    prompt = compile_final_report_instructions.format(
        user_query=state["enhanced_query"],
        summaries_context="\n".join(joined),
    )
    report = get_llm("gpt-4o-mini").invoke([SystemMessage(content=prompt)])
    return {"messages": [AIMessage(content=report.content)]}
