"""Core orchestration engine for VeriFind AI.

Coordinates the full end-to-end retrieval flow:
1. Query parsing & intent classification
2. Candidate URL discovery
3. Web navigation & dynamic fetching (Playwright / HTTPX fallback)
4. Trafilatura / BeautifulSoup content extraction
5. Passage ranking & scoring
6. Evidence verification and citation generation
"""

import time
from typing import List, Optional, Callable, Awaitable
from backend import (
    RetrievalResult,
    EvidenceItem,
    PageVisitRecord,
    VerificationStatus
)
from backend.query_parser import QueryParser
from backend.web_search import WebSearchEngine
from backend.browser import BrowserNavigator
from backend.extractor import ContentExtractor, ExtractedContent
from backend.ranker import PassageRanker
from backend.verifier import EvidenceVerificationEngine, VerificationReport
from backend.navigation_engine import WebsiteNavigationEngine
from backend.gemini_engine import GeminiAIEngine


class AnswerEngine:
    """Orchestrates intelligent web information retrieval and evidence synthesis."""

    def __init__(
        self,
        max_search_results: int = 4,
        headless: bool = True,
        timeout_ms: int = 15000
    ):
        self.parser = QueryParser()
        self.search_engine = WebSearchEngine(max_results=max_search_results)
        self.browser = BrowserNavigator(headless=headless, timeout_ms=timeout_ms)
        self.extractor = ContentExtractor()
        self.ranker = PassageRanker()
        self.verifier = EvidenceVerificationEngine()
        self.ai_engine = GeminiAIEngine()
        self.nav_engine = WebsiteNavigationEngine(
            max_depth=2,
            max_pages=8,
            timeout_ms=timeout_ms,
            headless=headless
        )

    async def execute_query(
        self,
        query_str: str,
        starting_url: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], Awaitable[None]]] = None
    ) -> RetrievalResult:
        """Executes the complete retrieval and verification workflow.

        Args:
            query_str: The natural language question.
            starting_url: Optional initial URL constraint.
            progress_callback: Optional async callback for UI progress reporting (stage_name, fraction).

        Returns:
            RetrievalResult containing the answer, evidence citations, visited pages, and status.
        """
        start_time = time.perf_counter()

        async def report(stage: str, fraction: float):
            if progress_callback:
                await progress_callback(stage, fraction)

        # 1. Parse Query & Understand Intent
        await report("Analyzing natural language query intent...", 0.15)
        parsed_query = self.parser.parse(query_str, starting_url=starting_url)
        query_understanding = await self.ai_engine.understand_query(query_str)

        # 2. Discover Candidate URLs
        await report("Discovering relevant web sources...", 0.30)
        candidates = await self.search_engine.search(parsed_query.cleaned_query, starting_url=starting_url)

        visited_pages: List[PageVisitRecord] = []
        extracted_docs: List[ExtractedContent] = []

        # If a starting URL was explicitly provided, perform multi-level deep navigation
        if starting_url and starting_url.strip():
            await report(f"Navigating website hierarchy for seed URL: {starting_url}...", 0.35)
            nav_session = await self.nav_engine.navigate_website(
                starting_url=starting_url,
                query_keywords=parsed_query.keywords
            )
            extracted_docs = nav_session.pages
            visited_pages = nav_session.visit_records
        else:
            if not candidates:
                elapsed = time.perf_counter() - start_time
                return RetrievalResult(
                    query=query_str,
                    direct_answer="No accessible web sources could be found or reached for this query.",
                    status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                    confidence_score=0.0,
                    execution_time_seconds=round(elapsed, 2),
                    notes="Network discovery yielded 0 reachable candidate URLs."
                )

            # 3. Navigate & Extract Pages
            step_increment = 0.40 / max(1, len(candidates))
            current_progress = 0.35

            for i, candidate in enumerate(candidates):
                await report(f"Navigating source ({i+1}/{len(candidates)}): {candidate.url[:50]}...", current_progress)
                t_fetch_start = time.perf_counter()
                html, status_code, method, error = await self.browser.fetch_page(candidate.url)
                t_fetch_end = time.perf_counter()
                latency_ms = (t_fetch_end - t_fetch_start) * 1000

                if html and status_code == 200:
                    doc = self.extractor.extract(html, candidate.url)
                    extracted_docs.append(doc)
                    visited_pages.append(
                        PageVisitRecord(
                            url=candidate.url,
                            title=doc.title,
                            status_code=status_code,
                            content_length=doc.text_length,
                            extraction_method=f"{method} + {doc.extraction_method}",
                            fetch_latency_ms=round(latency_ms, 2)
                        )
                    )
                else:
                    visited_pages.append(
                        PageVisitRecord(
                            url=candidate.url,
                            title=candidate.title or "Failed",
                            status_code=status_code,
                            content_length=0,
                            extraction_method=method,
                            fetch_latency_ms=round(latency_ms, 2),
                            error_message=error or f"HTTP {status_code}"
                        )
                    )
                current_progress += step_increment

        # 4. Rank Evidence Passages
        await report("Ranking extracted passages and evidence...", 0.80)
        ranked_passages = self.ranker.rank_passages(extracted_docs, parsed_query, top_k=5)

        evidence_items: List[EvidenceItem] = []
        for text, url, title, score in ranked_passages:
            evidence_items.append(
                EvidenceItem(
                    snippet=text,
                    source_url=url,
                    source_title=title,
                    relevance_score=score,
                    confidence_score=round(score * 0.95, 2)
                )
            )

        # 5. AI Reasoning & Structured Answer Synthesis
        await report("Synthesizing evidence-backed answer with Gemini AI...", 0.90)
        structured_ai_ans = await self.ai_engine.synthesize_answer(
            query=query_str,
            query_understanding=query_understanding,
            retrieved_documents=extracted_docs
        )

        # 6. Rigorous Fact Verification & Source Evidence Grounding
        await report("Verifying citations, validating sources and checking contradictions...", 0.96)
        claimed_cits = [
            {
                "quote": cit.quote,
                "source_url": cit.source_url,
                "source_title": cit.source_title
            }
            for cit in structured_ai_ans.citations
        ]

        verification_report = self.verifier.verify_answer(
            candidate_answer=structured_ai_ans.direct_answer,
            claimed_citations=claimed_cits,
            retrieved_documents=extracted_docs
        )

        # Build clean verified EvidenceItems with exact provenance metadata
        evidence_items = []
        for ve in verification_report.verified_evidence:
            evidence_items.append(
                EvidenceItem(
                    snippet=ve.quote,
                    source_url=ve.source_url,
                    source_title=ve.source_title,
                    relevance_score=ve.match_similarity,
                    confidence_score=ve.match_similarity,
                    metadata={
                        "retrieval_timestamp": ve.retrieval_timestamp,
                        "is_exact_match": ve.is_exact_match,
                        "supports_answer": ve.supports_answer
                    }
                )
            )

        # Fallback to ranked passages if verified evidence is sparse but valid
        if not evidence_items and ranked_passages:
            for text, url, title, score in ranked_passages[:3]:
                evidence_items.append(
                    EvidenceItem(
                        snippet=text,
                        source_url=url,
                        source_title=title,
                        relevance_score=score,
                        confidence_score=round(score * 0.90, 2)
                    )
                )

        await report("Completed.", 1.0)
        elapsed = time.perf_counter() - start_time

        conflicts_list = [
            {
                "claim": c.conflicting_claim,
                "source_a": c.source_a_url,
                "source_b": c.source_b_url,
                "reason": c.reason
            }
            for c in verification_report.conflicts_detected
        ]

        return RetrievalResult(
            query=query_str,
            direct_answer=verification_report.answer,  # None (null) if unsupported
            status=verification_report.verification_status,
            confidence_score=verification_report.confidence_score,
            evidence=evidence_items,
            visited_pages=visited_pages,
            execution_time_seconds=round(elapsed, 2),
            notes=verification_report.verification_notes,
            extracted_fields=structured_ai_ans.extracted_fields if verification_report.is_supported else {},
            missing_information=structured_ai_ans.missing_information,
            ai_reasoning=structured_ai_ans.reasoning_summary,
            conflicts_detected=conflicts_list,
            invalid_citations=verification_report.invalid_citations
        )

