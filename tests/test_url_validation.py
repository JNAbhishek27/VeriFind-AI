import pytest
from backend.browser import validate_and_normalize_url


def test_valid_http_urls():
    assert validate_and_normalize_url("http://example.com") == "http://example.com/"
    assert validate_and_normalize_url("https://example.com/page?query=1#fragment") == "https://example.com/page?query=1"
    assert validate_and_normalize_url("example.com/docs") == "https://example.com/docs"


def test_reject_invalid_schemes():
    with pytest.raises(ValueError, match="Only HTTP and HTTPS are permitted"):
        validate_and_normalize_url("ftp://files.example.com")

    with pytest.raises(ValueError, match="Only HTTP and HTTPS are permitted"):
        validate_and_normalize_url("javascript:alert(1)")

    with pytest.raises(ValueError, match="Only HTTP and HTTPS are permitted"):
        validate_and_normalize_url("file:///etc/passwd")


def test_reject_empty_or_malformed_urls():
    with pytest.raises(ValueError, match="URL cannot be empty"):
        validate_and_normalize_url("")

    with pytest.raises(ValueError, match="URL cannot be empty"):
        validate_and_normalize_url("   ")

