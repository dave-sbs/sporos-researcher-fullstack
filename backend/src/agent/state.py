from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict, Optional, List, Tuple
from pydantic import BaseModel, Field

from langgraph.graph import add_messages
from langchain_core.documents import Document
from typing_extensions import Annotated


import operator
from dataclasses import dataclass, field
from typing_extensions import Annotated


class ResearchGraphState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    enhanced_query: Optional[str]
    filters: Optional[FilterResult]
    retrieved_docs: Optional[List[Tuple[Document, float]]]
    graded_docs: List[Tuple[Document, float]]
    reconstructed_bills: Optional[List[ReconstructedBill]]
    bill_summaries: Annotated[List[BillSummary], operator.add]
    final_research_started: bool
    final_research: Optional[str]
    bill_card_data: Optional[List[BillCardData]]

class FilterResult(BaseModel):
    bill_identifier: Optional[str] = Field(default=None)
    year: Optional[List[int]] = Field(default=None)
    state: Optional[str] = Field(default=None)

class ReconstructedBill(TypedDict, total=False):
    id: str
    bill_identifier: str
    year: int
    state: str
    title: str
    similarity_score: float
    status: List[str]
    full_text: str


class BillSummary(TypedDict, total=False):
    bill_id: str
    title: str
    summary_text: str
    one_line_summary: str
    error_message: Optional[str]

class BillCardData(TypedDict, total=False):
    billId: str
    billIdentifier: str
    title: str
    state: str
    year: int
    sessionIdentifier: str
    fullTextUrl: str
    fullText: str
    oneLineSummary: str
    fullSummaryText: str
