"""Web search dispatcher and URL candidate generator.

Formulates target query payloads, navigates direct URLs, or submits queries to
search endpoints (such as DuckDuckGo HTML or custom discovery mechanisms)
without requiring paid API keys out-of-the-box.
"""

from dataclasses import dataclass
from typing import List, Optional
import urllib.parse
import httpx
from bs4 import BeautifulSoup


@dataclass
class SearchCandidate:
    """A search result link candidate with title and summary snippet."""
    url: str
    title: str
    snippet: str
    source_rank: int


class WebSearchEngine:
    """Dispatches search requests and yields candidate URLs."""

    def __init__(self, timeout: float = 12.0, max_results: int = 5):
        self.timeout = timeout
        self.max_results = max_results
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

    async def search(self, query: str, starting_url: Optional[str] = None) -> List[SearchCandidate]:
        """Discovers candidate URLs for a given search query.

        If a starting URL is explicitly provided, it prioritizes that URL as the
        primary candidate. Otherwise, it queries a public HTML search endpoint.

        Args:
            query: User's search query or question.
            starting_url: Optional user-specified seed URL.

        Returns:
            List of SearchCandidate objects.
        """
        candidates: List[SearchCandidate] = []

        # If a direct starting URL is supplied, make it rank #1
        if starting_url:
            cleaned_url = starting_url.strip()
            if not cleaned_url.startswith(("http://", "https://")):
                cleaned_url = "https://" + cleaned_url
            candidates.append(
                SearchCandidate(
                    url=cleaned_url,
                    title="User Specified Seed URL",
                    snippet=f"Direct seed URL target for: {query}",
                    source_rank=1
                )
            )

        # Query DuckDuckGo Lite / HTML as standard fallback
        try:
            encoded_query = urllib.parse.quote_plus(query)
            search_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

            async with httpx.AsyncClient(headers=self.headers, timeout=self.timeout, follow_redirects=True) as client:
                resp = await client.post("https://html.duckduckgo.com/html/", data={"q": query})
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    results = soup.select(".result")

                    rank = len(candidates) + 1
                    for res in results[:self.max_results]:
                        title_tag = res.select_one(".result__title a")
                        snippet_tag = res.select_one(".result__snippet")

                        if title_tag and title_tag.get("href"):
                            raw_href = title_tag["href"]
                            # DuckDuckGo wraps links in /l/?kh=-1&uddg=<actual_url>
                            parsed_url = self._extract_duckduckgo_url(raw_href)
                            title = title_tag.get_text(strip=True)
                            snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""

                            if parsed_url and not any(c.url == parsed_url for c in candidates):
                                candidates.append(
                                    SearchCandidate(
                                        url=parsed_url,
                                        title=title or "Web Result",
                                        snippet=snippet,
                                        source_rank=rank
                                    )
                                )
                                rank += 1
                                if len(candidates) >= self.max_results:
                                    break
        except Exception:
            # If network resolution fails or is restricted, return whatever candidates exist
            pass

        return candidates

    def _extract_duckduckgo_url(self, raw_href: str) -> Optional[str]:
        """Unwraps DuckDuckGo redirect link."""
        if "uddg=" in raw_href:
            parsed = urllib.parse.urlparse(raw_href)
            query_params = urllib.parse.parse_qs(parsed.query)
            if "uddg" in query_params:
                return query_params["uddg"][0]
        if raw_href.startswith("http"):
            return raw_href
        return None

