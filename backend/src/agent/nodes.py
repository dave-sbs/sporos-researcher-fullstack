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

    docs = retriever.invoke({"query": enhanced_query, "k": 20, "filters": filter_kwargs or None})
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
    
    # print(f"graded_docs: {graded_docs}")

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
        print(f"full_text: yes" if full_text else "no")

        # Fetch full_text_url from bills_dup2
        bill_url_res = sb.table("bills_dup2").select("full_text_url").eq("id", bill_id).execute()
        print(f"bill_url_res: {bill_url_res}")
        full_text_url = None
        try:
            if bill_url_res.data and len(bill_url_res.data) > 0:
                full_text_url = bill_url_res.data[0].get("full_text_url")
                print("Bill data:", bill_url_res.data[0])
            else:
                print(f"[reconstruct_full_text] No full_text_url found for bill_id={bill_id}")
        except Exception as e:
            print(f"[reconstruct_full_text] Exception fetching full_text_url for bill_id={bill_id}: {e}")

        bills.append(
            {
                "id": bill_id,
                "bill_identifier": doc.metadata.get("bill_identifier", "N/A"),
                "year": doc.metadata.get("year", 0),
                "state": doc.metadata.get("state", "N/A"),
                "title": doc.metadata.get("title", "N/A"),
                "session_identifier": doc.metadata.get("session_identifier", "N/A"),
                "similarity_score": gd["score"],
                "status": doc.metadata.get("status", []),
                "full_text": full_text,
                "full_text_url": full_text_url,
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
    summary_text = summary
    # print(summary_text)
        
    bill_summary_output: BillSummary = {
        "bill_id": bill["bill_id"],
        "title": bill["title"],
        "summary_text": summary_text,
        "one_line_summary": ""
    }
    return {"bill_summaries": [bill_summary_output]} # operator.add appends this list


# ---------------------------------------------------------------------------
# 7. Compile final research
# ---------------------------------------------------------------------------

def compile_final_research(state: ResearchGraphState, config: RunnableConfig) -> ResearchGraphState:
    print(f"state: {state.get('final_research_started')}")
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
    print(f"report: {report.content}")
    return {"messages": [AIMessage(content=report.content)]}


def emit_bill_card_data(state: ResearchGraphState, config: RunnableConfig) -> ResearchGraphState:
    """Join reconstructed_bills and bill_summaries, emit top 5 as BillCardData."""
    reconstructed_bills = state.get("reconstructed_bills", [])
    bill_summaries = state.get("bill_summaries", [])
    # print(f"reconstructed_bills: {reconstructed_bills}")
    # print(f"bill_summaries: {bill_summaries}")

    # Build a lookup for summaries by bill_id
    summary_lookup = {s.get("bill_id"): s for s in bill_summaries}
    card_data_list = []
    for bill in reconstructed_bills:
        bill_id = bill.get("id")
        summary = summary_lookup.get(bill_id, {})
        card_data = {
            "billId": bill.get("id"),
            "billIdentifier": bill.get("bill_identifier"),
            "title": bill.get("title"),
            "state": bill.get("state"),
            "year": bill.get("year"),
            "sessionIdentifier": bill.get("session_identifier"),
            "fullTextUrl": bill.get("full_text_url"),
            "fullText": bill.get("full_text"),
            "oneLineSummary": summary.get("one_line_summary", ""),
            "fullSummaryText": summary.get("summary_text", ""),
        }
        card_data_list.append(card_data)
    # Sort or filter top 5 if needed (e.g., by similarity_score)
    card_data_list = card_data_list[:5]
    return {"bill_card_data": card_data_list}


