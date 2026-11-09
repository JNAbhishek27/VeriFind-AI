import pytest
import asyncio
from unittest.mock import MagicMock
from backend.ai_models import (
    QueryUnderstanding,
    StructuredAnswer,
    GroundedCitation,
    AIConfidenceLevel
)
from backend.gemini_engine import GeminiAIEngine
from backend.extractor import ExtractedContent, HeadingItem, TableData


def test_query_understanding_model():
    qu = QueryUnderstanding(
        core_question="What is the memory bandwidth of the AMD Instinct MI300X?",
        target_entity="AMD Instinct MI300X",
        requested_fields=["memory_bandwidth", "memory_capacity"],
        expected_format="numeric",
        search_keywords=["AMD", "MI300X", "memory", "bandwidth"]
    )
    assert qu.target_entity == "AMD Instinct MI300X"
    assert "memory_bandwidth" in qu.requested_fields
    assert len(qu.search_keywords) == 4


def test_structured_answer_schema_validation():
    citation = GroundedCitation(
        quote="AMD Instinct MI300X provides up to 5.3 TB/s peak memory bandwidth.",
        source_url="https://amd.com/mi300x",
        source_title="AMD Instinct MI300X Product Page",
        supports_field="memory_bandwidth",
        relevance_score=0.98
    )

    ans = StructuredAnswer(
        direct_answer="The AMD Instinct MI300X offers a peak memory bandwidth of 5.3 TB/s with 192GB of HBM3 memory.",
        extracted_fields={
            "memory_bandwidth": "5.3 TB/s",
            "memory_capacity": "192 GB"
        },
        citations=[citation],
        has_sufficient_evidence=True,
        confidence=AIConfidenceLevel.HIGH,
        reasoning_summary="Extracted directly from AMD specifications table."
    )

    json_str = ans.model_dump_json()
    reloaded = StructuredAnswer.model_validate_json(json_str)
    assert reloaded.extracted_fields["memory_bandwidth"] == "5.3 TB/s"
    assert reloaded.citations[0].supports_field == "memory_bandwidth"
    assert reloaded.has_sufficient_evidence is True


def test_gemini_engine_missing_evidence_handling():
    async def _run():
        engine = GeminiAIEngine(api_key=None)
        qu = QueryUnderstanding(
            core_question="What is the price of the quantum computer?",
            requested_fields=["price"]
        )

        answer = await engine.synthesize_answer(
            query="What is the price of the quantum computer?",
            query_understanding=qu,
            retrieved_documents=[]
        )

        assert answer.has_sufficient_evidence is False
        assert answer.missing_information is not None
        assert answer.confidence == AIConfidenceLevel.UNVERIFIED

    asyncio.run(_run())


def test_gemini_engine_mock_client_structured_generation():
    async def _run():
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = """{
            "direct_answer": "AMD MI300X has 192GB HBM3 memory.",
            "extracted_fields": {"memory": "192GB"},
            "citations": [
                {
                    "quote": "Features 192GB HBM3 memory.",
                    "source_url": "https://example.com/amd",
                    "source_title": "AMD Specs",
                    "supports_field": "memory",
                    "relevance_score": 0.95
                }
            ],
            "has_sufficient_evidence": true,
            "missing_information": null,
            "confidence": "HIGH",
            "reasoning_summary": "Extracted from technical summary."
        }"""
        mock_client.models.generate_content.return_value = mock_response

        engine = GeminiAIEngine(client=mock_client)
        qu = QueryUnderstanding(
            core_question="What is the memory size?",
            requested_fields=["memory"]
        )

        doc = ExtractedContent(
            url="https://example.com/amd",
            title="AMD Specs",
            headings=[HeadingItem(level=1, text="Specs")],
            paragraphs=["Features 192GB HBM3 memory."]
        )

        answer = await engine.synthesize_answer("What is the memory size?", qu, [doc])

        assert answer.has_sufficient_evidence is True
        assert answer.extracted_fields["memory"] == "192GB"
        assert len(answer.citations) == 1
        assert answer.citations[0].quote == "Features 192GB HBM3 memory."
        assert answer.confidence == AIConfidenceLevel.HIGH

    asyncio.run(_run())
