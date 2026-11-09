import pytest
from backend.extractor import ContentExtractor


SAMPLE_RICH_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AMD ROCm Software Specifications</title>
</head>
<body>
    <header>
        <nav><a href="/index">Home</a><a href="https://other.org/about">External</a></nav>
    </header>
    <main>
        <h1>ROCm Open Software Ecosystem</h1>
        <h2>GPU Architecture Support</h2>
        <p>AMD ROCm is designed for high-performance computing and ultra-fast generative AI model inference.</p>
        <p>It provides direct support for CDNA accelerators including the Instinct MI300 series.</p>

        <h3>Hardware Specifications</h3>
        <table id="specs">
            <caption>Accelerator Comparison Matrix</caption>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Memory Type</th>
                    <th>Capacity</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>MI300X</td>
                    <td>HBM3</td>
                    <td>192 GB</td>
                </tr>
                <tr>
                    <td>MI250X</td>
                    <td>HBM2e</td>
                    <td>128 GB</td>
                </tr>
            </tbody>
        </table>

        <h4>Related Documentation</h4>
        <p>Read more at <a href="/docs/installation">Installation Guide</a> or check <a href="/docs/benchmarks">Benchmarks</a>.</p>
        <a href="mailto:support@amd.com">Email Us</a>
        <a href="javascript:void(0)">Click Here</a>
    </main>
</body>
</html>
"""


def test_extractor_extracts_headings_tables_and_links():
    extractor = ContentExtractor()
    doc = extractor.extract(SAMPLE_RICH_HTML, url="https://rocm.docs.amd.com/en/latest/")

    # 1. Title
    assert doc.title == "AMD ROCm Software Specifications"

    # 2. Headings
    heading_texts = [h.text for h in doc.headings]
    assert "ROCm Open Software Ecosystem" in heading_texts
    assert "GPU Architecture Support" in heading_texts
    assert "Hardware Specifications" in heading_texts
    assert any(h.level == 1 for h in doc.headings)
    assert any(h.level == 2 for h in doc.headings)
    assert any(h.level == 3 for h in doc.headings)

    # 3. Tables
    assert len(doc.tables) == 1
    table = doc.tables[0]
    assert table.caption == "Accelerator Comparison Matrix"
    assert table.headers == ["Model", "Memory Type", "Capacity"]
    assert len(table.rows) == 2
    assert table.rows[0] == ["MI300X", "HBM3", "192 GB"]
    assert table.rows[1] == ["MI250X", "HBM2e", "128 GB"]

    # 4. Paragraphs
    assert len(doc.paragraphs) >= 2
    combined_paras = " ".join(doc.paragraphs)
    assert "generative AI model inference" in combined_paras

    # 5. Outbound Links (filtered for HTTP/HTTPS, resolved)
    link_urls = [l.url for l in doc.links]
    assert "https://rocm.docs.amd.com/docs/installation" in link_urls
    assert "https://rocm.docs.amd.com/docs/benchmarks" in link_urls
    assert "https://other.org/about" in link_urls
    # Ignored non-HTTP schemes
    assert not any(l.startswith("mailto:") for l in link_urls)
    assert not any(l.startswith("javascript:") for l in link_urls)


def test_extractor_empty_html():
    extractor = ContentExtractor()
    doc = extractor.extract("", url="https://example.com/empty")
    assert doc.title == "Empty Document"
    assert doc.headings == []
    assert doc.tables == []
    assert doc.links == []
    assert doc.paragraphs == []
