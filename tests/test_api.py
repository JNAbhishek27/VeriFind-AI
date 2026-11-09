import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from backend.api import app
from backend import RetrievalResult, VerificationStatus, EvidenceItem, PageVisitRecord

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "verifind-ai" in data["service"]


def test_retrieve_endpoint_invalid_url():
    response = client.post(
        "/api/v1/retrieve",
        json={"query": "Test query", "starting_url": "ftp://bad-url.com"}
    )
    assert response.status_code == 400
    assert "Invalid starting_url" in response.json()["detail"]


def test_retrieve_endpoint_mock_execution():
    mock_result = RetrievalResult(
        query="What is MI300X memory?",
        direct_answer="AMD MI300X features 192GB HBM3.",
        status=VerificationStatus.VERIFIED,
        confidence_score=0.95,
        extracted_fields={"memory": "192GB"},
        missing_information=None,
        ai_reasoning="Directly corroborated.",
        evidence=[
            EvidenceItem(
                snippet="Features 192GB HBM3 memory.",
                source_url="https://amd.com/mi300x",
                source_title="AMD Specs",
                relevance_score=0.95,
                confidence_score=0.95
            )
        ],
        visited_pages=[
            PageVisitRecord(
                url="https://amd.com/mi300x",
                title="AMD Specs",
                status_code=200,
                content_length=1500,
                extraction_method="trafilatura",
                fetch_latency_ms=120.0
            )
        ],
        execution_time_seconds=1.25,
        notes="All fields verified."
    )

    with patch("backend.api.AnswerEngine.execute_query", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = mock_result
        response = client.post(
            "/api/v1/retrieve",
            json={"query": "What is MI300X memory?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["direct_answer"] == "AMD MI300X features 192GB HBM3."
        assert data["status"] == "VERIFIED"
        assert data["extracted_fields"]["memory"] == "192GB"
        assert len(data["evidence"]) == 1
        assert len(data["visited_pages"]) == 1

