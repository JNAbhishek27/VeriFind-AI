"""FastAPI REST Service for VeriFind AI.

Provides API endpoints for programmatic information retrieval, headless testing,
and integration with external workflows or AMD ROCm pipelines.
"""

import os
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.answer_engine import AnswerEngine
from backend import RetrievalResult, VerificationStatus
from backend.browser import validate_and_normalize_url
from backend.model_factory import ModelProviderFactory
from backend.provider_models import ProviderHealthStatus, InferenceBenchmarkResult

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
    model_provider: Optional[str] = Field(default=None, description="Optional provider override: 'gemini', 'amd_rocm', 'openai_compatible'.")
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
    """Returns system status and active AI model provider health."""
    active_provider = ModelProviderFactory.get_provider()
    provider_health = await active_provider.health_check()
    return {
        "status": "healthy",
        "service": "verifind-ai",
        "version": "0.3.0",
        "browser_engine": "Playwright Chromium + HTTPX fallback",
        "configured_provider": os.getenv("MODEL_PROVIDER", "gemini"),
        "provider_health": {
            "provider": provider_health.provider.value,
            "model_name": provider_health.model_name,
            "is_available": provider_health.is_available,
            "endpoint": provider_health.endpoint,
            "latency_ms": provider_health.latency_ms,
            "device_info": provider_health.device_info,
            "error_message": provider_health.error_message
        }
    }


@app.get("/api/v1/providers/health", tags=["AI Providers"])
async def check_all_providers():
    """Checks availability across all configured model providers (Gemini, AMD ROCm)."""
    gemini_prov = ModelProviderFactory.get_provider("gemini")
    rocm_prov = ModelProviderFactory.get_provider("amd_rocm")

    gemini_health = await gemini_prov.health_check()
    rocm_health = await rocm_prov.health_check()

    return {
        "gemini": {
            "model": gemini_health.model_name,
            "available": gemini_health.is_available,
            "latency_ms": gemini_health.latency_ms,
            "device": gemini_health.device_info,
            "error": gemini_health.error_message
        },
        "amd_rocm": {
            "model": rocm_health.model_name,
            "endpoint": rocm_health.endpoint,
            "available": rocm_health.is_available,
            "latency_ms": rocm_health.latency_ms,
            "device": rocm_health.device_info,
            "error": rocm_health.error_message
        }
    }


@app.post("/api/v1/providers/benchmark", tags=["AI Providers"])
async def benchmark_provider(provider: str = Query("amd_rocm", description="Provider to benchmark: 'amd_rocm' or 'gemini'")):
    """Benchmarks inference throughput, latency, and tokens/sec for the requested provider."""
    ai_engine = ModelProviderFactory.get_provider(provider)
    result = await ai_engine.benchmark()
    return {
        "provider": result.provider.value,
        "model_name": result.model_name,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "ttft_ms": result.ttft_ms,
        "total_latency_ms": result.total_latency_ms,
        "tokens_per_second": result.tokens_per_second,
        "device": result.device,
        "success": result.success,
        "error": result.error
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
        timeout_ms=request.timeout_ms,
        model_provider=request.model_provider
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
