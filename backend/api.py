"""FastAPI REST Service for VeriFind AI.

Provides API endpoints for programmatic information retrieval, headless testing,
and integration with external workflows or AMD ROCm pipelines.
"""

from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.answer_engine import AnswerEngine
from backend import RetrievalResult, VerificationStatus
from backend.browser import validate_and_normalize_url

app = FastAPI(
    title="VeriFind AI - Intelligent Web Information Retrieval API",
    description="AMD AI League Match 4: Evidence-Backed Web Retrieval & Fact Verification Engine",
    version="0.3.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    """Payload for submitting a retrieval and fact-verification query."""
    query: str = Field(..., min_length=3, description="Natural language question or target query.")
    starting_url: Optional[str] = Field(default=None, description="Optional seed URL or specific domain to restrict exploration.")
    max_search_results: int = Field(default=4, ge=1, le=10, description="Maximum search results / seeds to consider.")
    headless: bool = Field(default=True, description="Run browser in headless mode.")
    timeout_ms: int = Field(default=15000, ge=3000, le=45000, description="Per-page navigation timeout in milliseconds.")


class CitationResponse(BaseModel):
    snippet: str
    source_url: str
    source_title: str
    relevance_score: float
    confidence_score: float
    metadata: Dict[str, Any] = {}


class PageVisitResponse(BaseModel):
    url: str
    title: str
    status_code: int
    content_length: int
    extraction_method: str
    fetch_latency_ms: float
    error_message: Optional[str] = None


class QueryResponse(BaseModel):
    query: str
    direct_answer: Optional[str] = Field(default=None, description="Direct answer or null if unsupported by evidence.")
    status: VerificationStatus
    confidence_score: float
    extracted_fields: Dict[str, Any]
    missing_information: Optional[str]
    ai_reasoning: Optional[str]
    evidence: List[CitationResponse]
    visited_pages: List[PageVisitResponse]
    execution_time_seconds: float
    notes: Optional[str]
    conflicts_detected: List[Dict[str, str]] = []
    invalid_citations: List[str] = []


@app.get("/health", tags=["System"])
async def health_check():
    """Returns system status and engine readiness."""
    return {
        "status": "healthy",
        "service": "verifind-ai",
        "version": "0.3.0",
        "browser_engine": "Playwright Chromium + HTTPX fallback",
        "ai_engine": "Google GenAI (gemini-3.8-flash) / Local ROCm compatible"
    }


@app.post("/api/v1/retrieve", response_model=QueryResponse, tags=["Retrieval"])
async def retrieve_information(request: QueryRequest):
    """Executes the complete VeriFind AI retrieval, navigation, extraction, and verification pipeline."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be blank.")

    if request.starting_url:
        try:
            validate_and_normalize_url(request.starting_url)
        except ValueError as err:
            raise HTTPException(status_code=400, detail=f"Invalid starting_url: {str(err)}")

    engine = AnswerEngine(
        max_search_results=request.max_search_results,
        headless=request.headless,
        timeout_ms=request.timeout_ms
    )

    try:
        result: RetrievalResult = await engine.execute_query(
            query_str=request.query,
            starting_url=request.starting_url
        )

        return QueryResponse(
            query=result.query,
            direct_answer=result.direct_answer,
            status=result.status,
            confidence_score=result.confidence_score,
            extracted_fields=result.extracted_fields,
            missing_information=result.missing_information,
            ai_reasoning=result.ai_reasoning,
            evidence=[
                CitationResponse(
                    snippet=ev.snippet,
                    source_url=ev.source_url,
                    source_title=ev.source_title,
                    relevance_score=ev.relevance_score,
                    confidence_score=ev.confidence_score,
                    metadata=ev.metadata
                )
                for ev in result.evidence
            ],
            visited_pages=[
                PageVisitResponse(
                    url=p.url,
                    title=p.title,
                    status_code=p.status_code,
                    content_length=p.content_length,
                    extraction_method=p.extraction_method,
                    fetch_latency_ms=p.fetch_latency_ms,
                    error_message=p.error_message
                )
                for p in result.visited_pages
            ],
            execution_time_seconds=result.execution_time_seconds,
            notes=result.notes,
            conflicts_detected=result.conflicts_detected,
            invalid_citations=result.invalid_citations
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal retrieval error: {str(exc)}")
