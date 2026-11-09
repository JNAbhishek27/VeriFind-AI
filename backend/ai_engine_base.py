"""Abstract base interface for AI reasoning and synthesis engines.

Decouples the reasoning pipeline so Gemini API can be swapped with an
AMD ROCm-hosted local inference backend (e.g. vLLM or TGI running on MI300X/ROCm).
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from backend.ai_models import QueryUnderstanding, StructuredAnswer
from backend.extractor import ExtractedContent


class BaseAIEngine(ABC):
    """Abstract interface for natural-language query understanding and evidence synthesis."""

    @abstractmethod
    async def understand_query(self, query: str) -> QueryUnderstanding:
        """Analyzes a natural language question to extract target entities and requested fields.

        Args:
            query: The user's raw question.

        Returns:
            QueryUnderstanding object.
        """
        pass

    @abstractmethod
    async def synthesize_answer(
        self,
        query: str,
        query_understanding: QueryUnderstanding,
        retrieved_documents: List[ExtractedContent]
    ) -> StructuredAnswer:
        """Synthesizes a structured, citation-backed answer strictly grounded in retrieved documents.

        Args:
            query: The user's question.
            query_understanding: Analysis of query intent and requested fields.
            retrieved_documents: Structured content extracted from visited websites.

        Returns:
            StructuredAnswer object with verbatim citations, extracted fields, and missing info flags.
        """
        pass

