from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentGrade(BaseModel):
    doc_index: int = Field(description="0-based index in original list")
    is_relevant: bool
    reasoning: Optional[str] = None


class DocumentGrades(BaseModel):
    grades: List[DocumentGrade]

class BillSummaryLLM(BaseModel):
    summary_text: str = Field(description="A full, comprehensive summary of the bill")
    one_line_summary: str = Field(description="A one-line summary of the bill")