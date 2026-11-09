import pytest
from backend import VerificationStatus
from backend.verifier import EvidenceVerificationEngine, VerificationReport
from backend.extractor import ExtractedContent, HeadingItem, TableData


DOC_A = ExtractedContent(
    url="https://amd.com/products/mi300x",
    title="AMD Instinct MI300X Product Page",
    headings=[HeadingItem(level=1, text="AMD Instinct MI300X Accelerator")],
    paragraphs=[
        "The AMD Instinct MI300X accelerator provides leadership memory capacity and bandwidth for generative AI workloads.",
        "With 192GB of HBM3 memory and 5.3 TB/s peak memory bandwidth, MI300X enables single-node execution of large language models."
    ],
    main_text="The AMD Instinct MI300X accelerator provides leadership memory capacity. With 192GB of HBM3 memory and 5.3 TB/s peak memory bandwidth, MI300X enables single-node execution of large language models."
)

DOC_B = ExtractedContent(
    url="https://tech-analysis.org/mi300x-benchmarks",
    title="Tech Analysis MI300X Benchmark Report",
    headings=[HeadingItem(level=1, text="MI300X Benchmarks")],
    paragraphs=[
        "In our independent laboratory tests, the AMD Instinct MI300X demonstrated 5.3 TB/s memory bandwidth, corroborating official specifications.",
        "The 192GB HBM3 configuration allows running massive 70B parameter models without memory bottlenecks."
    ],
    main_text="In our independent laboratory tests, the AMD Instinct MI300X demonstrated 5.3 TB/s memory bandwidth, corroborating official specifications. The 192GB HBM3 configuration allows running massive 70B parameter models."
)

DOC_CONTRADICTORY = ExtractedContent(
    url="https://errata-tech.com/amd-correction",
    title="Errata Tech News",
    paragraphs=[
        "Earlier rumors claimed 128GB, but AMD confirmed MI300X memory is 192GB instead of 128GB.",
        "Contrary to initial reports claiming 3.2 TB/s instead of 5.3 TB/s, the verified peak bandwidth is 5.3 TB/s."
    ],
    main_text="Earlier rumors claimed 128GB, but AMD confirmed MI300X memory is 192GB instead of 128GB. Contrary to initial reports claiming 3.2 TB/s instead of 5.3 TB/s, the verified peak bandwidth is 5.3 TB/s."
)


def test_correct_answer_multi_source_verified():
    """Validates that a correct answer with legitimate citations across 2 domains is VERIFIED."""
    verifier = EvidenceVerificationEngine()
    answer = "The AMD Instinct MI300X features 192GB HBM3 memory and 5.3 TB/s peak memory bandwidth."

    claimed_citations = [
        {
            "quote": "With 192GB of HBM3 memory and 5.3 TB/s peak memory bandwidth, MI300X enables single-node execution of large language models.",
            "source_url": "https://amd.com/products/mi300x",
            "source_title": "AMD Instinct MI300X Product Page"
        },
        {
            "quote": "In our independent laboratory tests, the AMD Instinct MI300X demonstrated 5.3 TB/s memory bandwidth, corroborating official specifications.",
            "source_url": "https://tech-analysis.org/mi300x-benchmarks",
            "source_title": "Tech Analysis MI300X Benchmark Report"
        }
    ]

    report = verifier.verify_answer(
        candidate_answer=answer,
        claimed_citations=claimed_citations,
        retrieved_documents=[DOC_A, DOC_B]
    )

    assert report.is_supported is True
    assert report.answer == answer
    assert report.verification_status == VerificationStatus.VERIFIED
    assert report.confidence_score >= 0.85
    assert len(report.verified_evidence) == 2
    assert report.verified_evidence[0].is_exact_match is True
    assert report.verified_evidence[0].retrieval_timestamp is not None
    assert report.to_dict()["answer"] == answer


def test_incorrect_citations_returns_null_answer():
    """Validates that fabricated or hallucinated quotes not found in the documents return answer=None."""
    verifier = EvidenceVerificationEngine()
    answer = "The quantum chip operates at absolute zero with 500 qubits."

    # Citation has quote that is fabricated / not in DOC_A
    claimed_citations = [
        {
            "quote": "The quantum chip operates at absolute zero with 500 superconducting qubits.",
            "source_url": "https://amd.com/products/mi300x",
            "source_title": "Fabricated Document"
        }
    ]

    report = verifier.verify_answer(
        candidate_answer=answer,
        claimed_citations=claimed_citations,
        retrieved_documents=[DOC_A]
    )

    # Must return None for unsupported answers
    assert report.answer is None
    assert report.is_supported is False
    assert report.verification_status == VerificationStatus.UNVERIFIED
    assert len(report.invalid_citations) >= 1
    assert "could not be found" in report.invalid_citations[0]


def test_unvisited_url_citation_rejected():
    """Validates that citations referencing URLs never visited by the browser are rejected."""
    verifier = EvidenceVerificationEngine()
    answer = "AMD MI300X offers 192GB memory."

    claimed_citations = [
        {
            "quote": "With 192GB of HBM3 memory",
            "source_url": "https://unvisited-site.com/fake-page",
            "source_title": "Unvisited Fake Page"
        }
    ]

    report = verifier.verify_answer(
        candidate_answer=answer,
        claimed_citations=claimed_citations,
        retrieved_documents=[DOC_A]  # unvisited-site is not in retrieved docs
    )

    assert len(report.invalid_citations) >= 1
    assert "was not visited" in report.invalid_citations[0]


def test_missing_information_returns_null_answer():
    """Validates that missing info or refusal responses return answer=None with INSUFFICIENT_EVIDENCE."""
    verifier = EvidenceVerificationEngine()

    report_refusal = verifier.verify_answer(
        candidate_answer="No sufficient evidence found to determine the founding date.",
        claimed_citations=[],
        retrieved_documents=[DOC_A]
    )
    assert report_refusal.answer is None
    assert report_refusal.verification_status == VerificationStatus.INSUFFICIENT_EVIDENCE
    assert report_refusal.is_supported is False

    report_empty = verifier.verify_answer(
        candidate_answer="",
        claimed_citations=[],
        retrieved_documents=[]
    )
    assert report_empty.answer is None
    assert report_empty.verification_status == VerificationStatus.INSUFFICIENT_EVIDENCE


def test_conflicting_information_detection():
    """Validates that contradictory reports across different pages are flagged and invalidate the answer."""
    verifier = EvidenceVerificationEngine()
    # Answer claiming 3.2 TB/s while errata news documents that 3.2 TB/s is contrary to 5.3 TB/s
    answer = "The verified peak bandwidth is 3.2 TB/s for the MI300X."

    report = verifier.verify_answer(
        candidate_answer=answer,
        claimed_citations=[],
        retrieved_documents=[DOC_A, DOC_CONTRADICTORY]
    )

    assert report.conflicts_detected != []
    assert report.answer is None  # Conflicting information sets answer to null
    assert report.verification_status == VerificationStatus.CONTRADICTORY
