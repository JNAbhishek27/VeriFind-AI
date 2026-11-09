"""AI Reasoning and structured response models for VeriFind AI.

Pydantic models representing:
- Query intent and requested target fields.
- Passage evidence citations with exact quotes.
- Structured answers with confidence and verification status.
"""

from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class AIConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNVERIFIED = "UNVERIFIED"


class QueryUnderstanding(BaseModel):
    """Analysis of user's natural language question and extraction targets."""
    core_question: str = Field(description="The core normalized question being asked.")
    target_entity: Optional[str] = Field(default=None, description="The subject or entity investigated (e.g. 'AMD Instinct MI300X').")
    requested_fields: List[str] = Field(default_factory=list, description="Specific fields or metrics asked (e.g. ['memory_capacity', 'bandwidth']).")
    expected_format: str = Field(default="sentence", description="Expected answer format: 'numeric', 'date', 'definition', 'list', 'sentence'.")
    search_keywords: List[str] = Field(default_factory=list, description="Target search and ranking terms.")


class GroundedCitation(BaseModel):
    """An exact verbatim quotation from a source page serving as evidence."""
    quote: str = Field(description="Exact verbatim quote from the provided webpage text.")
    source_url: str = Field(description="The source URL where this quote was extracted.")
    source_title: str = Field(default="Source Document", description="The title of the page.")
    supports_field: Optional[str] = Field(default=None, description="The specific field or claim this citation supports.")
    relevance_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Estimated relevance to the user question.")


class StructuredAnswer(BaseModel):
    """Structured fact-checked answer synthesized strictly from evidence."""
    direct_answer: str = Field(description="Clear, concise, direct answer based strictly on verified evidence.")
    extracted_fields: Dict[str, Any] = Field(default_factory=dict, description="Dictionary of requested fields and their exact extracted values.")
    citations: List[GroundedCitation] = Field(default_factory=list, description="Verbatim quotations from retrieved sources backing the answer.")
    has_sufficient_evidence: bool = Field(description="True if the provided pages contain sufficient facts to answer the question, False if facts are missing.")
    missing_information: Optional[str] = Field(default=None, description="Description of any requested details that could not be found in the evidence.")
    confidence: AIConfidenceLevel = Field(default=AIConfidenceLevel.MEDIUM, description="Confidence level based on corroboration and clarity.")
    reasoning_summary: str = Field(description="Brief summary of how the answer was established from the retrieved evidence.")

