"""VeriFind AI Backend Package.

Modular components for intelligent web information retrieval, navigation,
content extraction, ranking, and fact verification.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class VerificationStatus(str, Enum):
    """Verification classification status for findings."""
    VERIFIED = "VERIFIED"
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    CONTRADICTORY = "CONTRADICTORY"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass
class EvidenceItem:
    """An atomic piece of extracted evidence backed by source data."""
    snippet: str
    source_url: str
    source_title: str
    relevance_score: float
    confidence_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PageVisitRecord:
    """Metadata detailing a web page visited by the retrieval pipeline."""
    url: str
    title: str
    status_code: int
    content_length: int
    extraction_method: str
    fetch_latency_ms: float
    error_message: Optional[str] = None


@dataclass
class RetrievalResult:
    """Structured response container returned to the user or frontend."""
    query: str
    direct_answer: Optional[str]  # None if unsupported
    status: VerificationStatus
    confidence_score: float
    evidence: List[EvidenceItem] = field(default_factory=list)
    visited_pages: List[PageVisitRecord] = field(default_factory=list)
    execution_time_seconds: float = 0.0
    notes: Optional[str] = None
    extracted_fields: Dict[str, Any] = field(default_factory=dict)
    missing_information: Optional[str] = None
    ai_reasoning: Optional[str] = None
    conflicts_detected: List[Dict[str, str]] = field(default_factory=list)
    invalid_citations: List[str] = field(default_factory=list)

