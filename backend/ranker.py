"""Passage and evidence ranking engine.

Computes semantic and lexical relevance scores between extracted text passages
and parsed queries to identify the highest-value evidence candidates.
"""

from typing import List, Tuple
import re
from backend.query_parser import ParsedQuery
from backend.extractor import ExtractedContent


class PassageRanker:
    """Ranks extracted text snippets based on keyword density, position, and coverage."""

    def rank_passages(
        self,
        extracted_documents: List[ExtractedContent],
        query: ParsedQuery,
        top_k: int = 5
    ) -> List[Tuple[str, str, str, float]]:
        """Scores passages across extracted documents against the query.

        Args:
            extracted_documents: List of ExtractedContent objects.
            query: ParsedQuery representing the search criteria.
            top_k: Number of highest-scoring passages to return.

        Returns:
            List of tuples: (passage_text, source_url, source_title, score)
        """
        scored_candidates: List[Tuple[str, str, str, float]] = []

        keywords_lower = [k.lower() for k in query.keywords]
        if not keywords_lower:
            keywords_lower = [token.lower() for token in query.cleaned_query.split() if len(token) > 2]

        for doc in extracted_documents:
            for para in doc.paragraphs:
                score = self._compute_relevance(para, keywords_lower, query.expected_type)
                if score > 0.05:
                    scored_candidates.append((para, doc.url, doc.title, score))

        # Sort descending by score
        scored_candidates.sort(key=lambda item: item[3], reverse=True)
        return scored_candidates[:top_k]

    def _compute_relevance(self, passage: str, keywords: List[str], expected_type: str) -> float:
        """Calculates a normalized score [0.0 - 1.0] for a passage."""
        if not passage or not keywords:
            return 0.0

        p_lower = passage.lower()
        matched_keywords = sum(1 for kw in keywords if re.search(r"\b" + re.escape(kw) + r"\b", p_lower))
        keyword_density = matched_keywords / len(keywords)

        # Expected type boosts
        type_boost = 0.0
        if expected_type == "numeric" and re.search(r"\b\d+([.,]\d+)?%?\b", passage):
            type_boost = 0.15
        elif expected_type == "temporal" and re.search(r"\b(19\d\d|20\d\d|january|february|march|april|may|june|july|august|september|october|november|december)\b", p_lower):
            type_boost = 0.15
        elif expected_type == "entity" and re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", passage):
            type_boost = 0.10

        # Penalize excessively long or tiny passages
        length_factor = 1.0
        word_count = len(passage.split())
        if word_count < 8:
            length_factor = 0.5
        elif word_count > 120:
            length_factor = 0.85

        final_score = (keyword_density * 0.75 + type_boost) * length_factor
        return round(min(1.0, max(0.0, final_score)), 4)

