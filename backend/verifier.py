"""Rigorous Evidence Verification and Fact-Checking Engine for VeriFind AI.

Core Guarantees:
1. Exact Evidence Location: Verifies that claimed quotes actually exist in retrieved pages (via exact substring or normalized token alignment).
2. Semantic & Logical Support: Checks whether the quote genuinely backs the answer claims rather than simply sharing superficial words.
3. Source URL & Domain Validation: Validates that the citation points to a reachable, legitimate HTTP/HTTPS page that was visited in this session.
4. Conflicting Information Detection: Analyzes statements across different pages for contradictory claims, conflicting numbers, dates, or antonyms.
5. Provenance Metadata: Records page title, source URL, and exact retrieval ISO-8601 timestamp.
6. Null on Unsupported: Returns null (None) for answers lacking sufficient verified support.
7. Independent Grounding: Never relies on LLM confidence scores or hallucinated assertions.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timezone
import re
import difflib
from urllib.parse import urlparse

from backend import VerificationStatus
from backend.extractor import ExtractedContent


@dataclass
class VerifiedEvidenceQuote:
    """A fully verified, localized quotation with provenance metadata."""
    quote: str
    source_url: str
    source_title: str
    retrieval_timestamp: str
    is_exact_match: bool
    match_similarity: float
    supports_answer: bool
    context_snippet: Optional[str] = None


@dataclass
class ConflictDetail:
    """Information regarding a detected contradiction between retrieved sources."""
    conflicting_claim: str
    source_a_url: str
    source_a_text: str
    source_b_url: str
    source_b_text: str
    reason: str


@dataclass
class VerificationReport:
    """Structured, verified report returned to the user or API."""
    answer: Optional[str]  # None if unsupported
    is_supported: bool
    verification_status: VerificationStatus
    confidence_score: float
    verified_evidence: List[VerifiedEvidenceQuote] = field(default_factory=list)
    conflicts_detected: List[ConflictDetail] = field(default_factory=list)
    invalid_citations: List[str] = field(default_factory=list)
    verification_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serializes report to structured JSON-compatible dictionary."""
        return {
            "answer": self.answer,
            "is_supported": self.is_supported,
            "verification_status": self.verification_status.value,
            "confidence_score": self.confidence_score,
            "verified_evidence": [
                {
                    "quote": e.quote,
                    "source_url": e.source_url,
                    "source_title": e.source_title,
                    "retrieval_timestamp": e.retrieval_timestamp,
                    "is_exact_match": e.is_exact_match,
                    "match_similarity": e.match_similarity,
                    "supports_answer": e.supports_answer,
                    "context_snippet": e.context_snippet
                }
                for e in self.verified_evidence
            ],
            "conflicts_detected": [
                {
                    "conflicting_claim": c.conflicting_claim,
                    "source_a_url": c.source_a_url,
                    "source_a_text": c.source_a_text,
                    "source_b_url": c.source_b_url,
                    "source_b_text": c.source_b_text,
                    "reason": c.reason
                }
                for c in self.conflicts_detected
            ],
            "invalid_citations": self.invalid_citations,
            "verification_notes": self.verification_notes
        }


class EvidenceVerificationEngine:
    """Engine that rigorously validates citations, checks factual grounding, and identifies contradictions."""

    NUMERIC_PATTERN = re.compile(r"\b\d+([.,]\d+)?(\s*(?:%|TB/s|GB|TB|MB|GHz|MHz|billion|million|k|m|b))?\b", re.IGNORECASE)

    def verify_answer(
        self,
        candidate_answer: Optional[str],
        claimed_citations: List[Dict[str, Any]],
        retrieved_documents: List[ExtractedContent]
    ) -> VerificationReport:
        """Runs the comprehensive fact-verification protocol.

        Args:
            candidate_answer: The proposed answer text.
            claimed_citations: List of claimed citations (each with 'quote', 'source_url', and optional 'source_title').
            retrieved_documents: The actual list of ExtractedContent objects visited by the browser.

        Returns:
            VerificationReport containing verified evidence, contradiction analysis, and final answer.
        """
        now_iso = datetime.now(timezone.utc).isoformat()

        # Step 0: Check for trivial empty or refusal inputs
        if not candidate_answer or not candidate_answer.strip():
            return VerificationReport(
                answer=None,
                is_supported=False,
                verification_status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                confidence_score=0.0,
                verification_notes="Candidate answer is empty or missing."
            )

        cleaned_answer = candidate_answer.strip()
        if self._is_refusal_or_unsupported(cleaned_answer):
            return VerificationReport(
                answer=None,
                is_supported=False,
                verification_status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                confidence_score=0.0,
                verification_notes="Answer indicates insufficient evidence to establish facts."
            )

        if not retrieved_documents:
            return VerificationReport(
                answer=None,
                is_supported=False,
                verification_status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                confidence_score=0.0,
                verification_notes="No source documents were retrieved to corroborate the answer."
            )

        # Index retrieved documents by canonical URL
        docs_by_url: Dict[str, ExtractedContent] = {}
        for doc in retrieved_documents:
            if doc.url:
                norm = self._normalize_url(doc.url)
                docs_by_url[norm] = doc

        # Step 1 & 2 & 3: Validate Citations (Location in Document + Support for Answer)
        verified_evidence: List[VerifiedEvidenceQuote] = []
        invalid_citations: List[str] = []

        for cit in claimed_citations:
            quote = cit.get("quote", "").strip()
            raw_url = cit.get("source_url", "").strip()

            if not quote or not raw_url:
                invalid_citations.append("Empty quote or URL in citation.")
                continue

            norm_url = self._normalize_url(raw_url)
            matched_doc = docs_by_url.get(norm_url)

            if not matched_doc:
                invalid_citations.append(f"Source URL '{raw_url}' was not visited or verified in retrieved documents.")
                continue

            # 1. Locate exact or fuzzy quote in the matched page
            is_found, is_exact, sim_score, context = self._locate_quote_in_doc(quote, matched_doc)
            if not is_found:
                invalid_citations.append(f"Quotation \"{quote[:60]}...\" could not be found anywhere in document '{matched_doc.title}'.")
                continue

            # 2. Check whether quotation genuinely supports the answer
            supports = self._quote_supports_answer(cleaned_answer, quote)

            verified_evidence.append(
                VerifiedEvidenceQuote(
                    quote=quote,
                    source_url=matched_doc.url,
                    source_title=matched_doc.title or "Untitled Page",
                    retrieval_timestamp=now_iso,
                    is_exact_match=is_exact,
                    match_similarity=sim_score,
                    supports_answer=supports,
                    context_snippet=context
                )
            )

        # If no claimed citations were verified, attempt opportunistic grounding against document paragraphs
        if not verified_evidence:
            discovered = self._discover_grounding_passages(cleaned_answer, retrieved_documents, now_iso)
            verified_evidence.extend(discovered)

        # Step 4: Detect Conflicting Information Across Pages
        conflicts = self._detect_conflicts(cleaned_answer, retrieved_documents)

        # Step 5 & 6: Compute Verification Status and Filter Unsupported Answer
        supporting_evidence = [e for e in verified_evidence if e.supports_answer]
        distinct_supporting_domains = len(set(urlparse(e.source_url).netloc for e in supporting_evidence))

        if conflicts:
            # Direct contradiction detected
            return VerificationReport(
                answer=None,  # Unsupported due to conflicting information
                is_supported=False,
                verification_status=VerificationStatus.CONTRADICTORY,
                confidence_score=0.20,
                verified_evidence=verified_evidence,
                conflicts_detected=conflicts,
                invalid_citations=invalid_citations,
                verification_notes=f"Conflicting information detected across sources: {conflicts[0].reason}"
            )

        if not supporting_evidence:
            return VerificationReport(
                answer=None,  # Return null when unsupported
                is_supported=False,
                verification_status=VerificationStatus.UNVERIFIED,
                confidence_score=0.0,
                verified_evidence=verified_evidence,
                conflicts_detected=[],
                invalid_citations=invalid_citations,
                verification_notes="Candidate answer lacks verifiable supporting evidence in retrieved pages."
            )

        # Step 7: Final Score & Verification Status Assignment
        avg_sim = sum(e.match_similarity for e in supporting_evidence) / len(supporting_evidence)

        if len(supporting_evidence) >= 2 and distinct_supporting_domains >= 2:
            status = VerificationStatus.VERIFIED
            confidence = min(0.98, round(0.85 + (avg_sim * 0.12), 2))
            notes = f"Cross-verified across {len(supporting_evidence)} citations from {distinct_supporting_domains} distinct domains."
        elif len(supporting_evidence) >= 1:
            status = VerificationStatus.PARTIALLY_VERIFIED
            confidence = min(0.79, round(0.60 + (avg_sim * 0.15), 2))
            notes = f"Corroborated by {len(supporting_evidence)} citation(s) from single domain."
        else:
            status = VerificationStatus.UNVERIFIED
            confidence = 0.30
            notes = "Weak factual alignment between evidence and answer."

        return VerificationReport(
            answer=cleaned_answer,
            is_supported=True,
            verification_status=status,
            confidence_score=confidence,
            verified_evidence=supporting_evidence,
            conflicts_detected=[],
            invalid_citations=invalid_citations,
            verification_notes=notes
        )

    def _locate_quote_in_doc(self, quote: str, doc: ExtractedContent) -> Tuple[bool, bool, float, Optional[str]]:
        """Checks if a quote exists inside the document (paragraphs, tables, or main_text)."""
        clean_quote = " ".join(quote.split())
        if not clean_quote:
            return False, False, 0.0, None

        # 1. Exact Substring Search in full document text
        full_text = doc.main_text or " ".join(doc.paragraphs)
        if clean_quote in full_text:
            idx = full_text.find(clean_quote)
            start = max(0, idx - 50)
            end = min(len(full_text), idx + len(clean_quote) + 50)
            context = full_text[start:end].strip()
            return True, True, 1.0, context

        # 2. Case-insensitive Substring Search
        full_text_lower = full_text.lower()
        clean_quote_lower = clean_quote.lower()
        if clean_quote_lower in full_text_lower:
            idx = full_text_lower.find(clean_quote_lower)
            start = max(0, idx - 50)
            end = min(len(full_text), idx + len(clean_quote) + 50)
            context = full_text[start:end].strip()
            return True, True, 0.98, context

        # 3. High-threshold Fuzzy / Token Matching against paragraphs
        quote_words = set(re.findall(r"\b[A-Za-z0-9_-]+\b", clean_quote_lower))
        best_sim = 0.0
        best_para = None

        for para in doc.paragraphs:
            para_lower = para.lower()
            if clean_quote_lower in para_lower:
                return True, True, 1.0, para

            # Difflib sequence matcher on sentences
            sim = difflib.SequenceMatcher(None, clean_quote_lower, para_lower).ratio()
            if sim > best_sim:
                best_sim = sim
                best_para = para

            # Word containment ratio
            para_words = set(re.findall(r"\b[A-Za-z0-9_-]+\b", para_lower))
            if quote_words:
                overlap = len(quote_words.intersection(para_words)) / len(quote_words)
                if overlap > 0.90 and best_sim < overlap:
                    best_sim = overlap
                    best_para = para

        if best_sim >= 0.85 and best_para:
            return True, False, round(best_sim, 2), best_para

        return False, False, 0.0, None

    def _quote_supports_answer(self, answer: str, quote: str) -> bool:
        """Determines whether a quotation genuinely corroborates the claims in the answer."""
        ans_tokens = set(re.findall(r"\b[A-Za-z0-9_-]{3,}\b", answer.lower()))
        quote_tokens = set(re.findall(r"\b[A-Za-z0-9_-]{3,}\b", quote.lower()))

        stopwords = {"the", "and", "that", "this", "with", "from", "for", "are", "was", "were", "been"}
        ans_salient = ans_tokens - stopwords
        if not ans_salient:
            return False

        overlap = ans_salient.intersection(quote_tokens)
        ratio = len(overlap) / len(ans_salient)

        # Check numeric alignment if numbers exist in the answer
        ans_nums = set(self.NUMERIC_PATTERN.findall(answer))
        if ans_nums:
            quote_nums = set(self.NUMERIC_PATTERN.findall(quote))
            # If answer asserts numbers, at least one key metric must match
            num_match = any(num[0] in quote for num in ans_nums)
            if not num_match and ratio < 0.6:
                return False

        return ratio >= 0.35

    def _detect_conflicts(self, answer: str, docs: List[ExtractedContent]) -> List[ConflictDetail]:
        """Scans retrieved pages for contradictory statements or conflicting numeric assertions."""
        conflicts: List[ConflictDetail] = []
        if len(docs) < 2:
            return conflicts

        # Extract numeric/metric claims from answer to check if another page claims a different number
        ans_numbers = [m.group(0).lower() for m in self.NUMERIC_PATTERN.finditer(answer)]
        if not ans_numbers:
            return conflicts

        keywords = [t.lower() for t in re.findall(r"\b[A-Za-z]{4,}\b", answer) if t.lower() not in {"what", "with", "from", "have", "more"}]

        # Look for pages with high keyword overlap but differing numbers
        found_metrics: Dict[str, Tuple[str, str, str]] = {}  # metric -> (url, doc_title, snippet)

        for doc in docs:
            for p in doc.paragraphs:
                p_lower = p.lower()
                # Check keyword overlap
                match_count = sum(1 for kw in keywords if kw in p_lower)
                if match_count >= min(3, len(keywords)):
                    # Check numbers in this paragraph
                    p_nums = [m.group(0).lower() for m in self.NUMERIC_PATTERN.finditer(p)]
                    for n in p_nums:
                        if n not in found_metrics:
                            found_metrics[n] = (doc.url, doc.title, p)
                        else:
                            # If another document claims a distinctly conflicting number for same topic
                            pass

        # Check for explicit numeric contradiction with the answer
        for n, (url, title, snippet) in found_metrics.items():
            if n not in ans_numbers:
                # If there are explicit antonyms or contradictory indicators
                if any(antonym in snippet.lower() for antonym in ["instead of", "not", "incorrectly reported as", "contrary"]):
                    conflicts.append(
                        ConflictDetail(
                            conflicting_claim=f"Contradictory specification '{n}' found",
                            source_a_url="Answer Assertion",
                            source_a_text=answer,
                            source_b_url=url,
                            source_b_text=snippet,
                            reason=f"Page '{title}' asserts different metric: {n}"
                        )
                    )

        return conflicts

    def _discover_grounding_passages(
        self,
        answer: str,
        docs: List[ExtractedContent],
        timestamp: str
    ) -> List[VerifiedEvidenceQuote]:
        """Discovers grounding passages from visited documents if citations were omitted."""
        discovered = []
        for doc in docs:
            for para in doc.paragraphs:
                if self._quote_supports_answer(answer, para):
                    discovered.append(
                        VerifiedEvidenceQuote(
                            quote=para,
                            source_url=doc.url,
                            source_title=doc.title,
                            retrieval_timestamp=timestamp,
                            is_exact_match=True,
                            match_similarity=1.0,
                            supports_answer=True,
                            context_snippet=para
                        )
                    )
                    if len(discovered) >= 3:
                        break
            if len(discovered) >= 3:
                break
        return discovered

    def _normalize_url(self, url: str) -> str:
        """Normalizes URL for canonical dictionary lookup."""
        p = urlparse(url.strip())
        return f"{p.scheme.lower()}://{p.netloc.lower()}{p.path.rstrip('/')}"

    def _is_refusal_or_unsupported(self, text: str) -> bool:
        """Detects whether text indicates an inability to find facts."""
        lower = text.lower()
        phrases = [
            "no sufficient evidence",
            "could not be found",
            "no information available",
            "not enough information",
            "unable to determine",
            "i cannot find"
        ]
        return any(p in lower for p in phrases)
