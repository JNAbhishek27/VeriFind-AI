"""Autonomous website navigation engine for VeriFind AI.

Orchestrates multi-level website exploration:
1. Validates starting URLs (HTTP/HTTPS strictly enforced).
2. Deep navigation up to 2 levels deep (depth 0 = root, depth 1 = child, depth 2 = grandchild).
3. Hard cap of maximum pages visited (default 8 pages per query).
4. Maintains full visited URL history avoiding duplicates.
5. Handles dynamic JavaScript rendering with Playwright and fallback mechanisms.
6. Returns rich structured page content (titles, headings, paragraphs, tables, links, sources).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Callable, Awaitable
import asyncio
import logging
import time
import urllib.parse
from collections import deque

from backend.browser import BrowserNavigator, validate_and_normalize_url
from backend.extractor import ContentExtractor, ExtractedContent
from backend.link_scorer import LinkRelevanceScorer
from backend import PageVisitRecord

logger = logging.getLogger(__name__)


@dataclass
class CrawlQueueItem:
    """An item in the crawl exploration queue."""
    url: str
    depth: int
    parent_url: Optional[str] = None


@dataclass
class NavigationSessionResult:
    """Complete structured output from a website navigation run."""
    starting_url: str
    total_pages_visited: int
    pages: List[ExtractedContent] = field(default_factory=list)
    visit_records: List[PageVisitRecord] = field(default_factory=list)
    visited_urls: List[str] = field(default_factory=list)
    failed_urls: Dict[str, str] = field(default_factory=dict)
    execution_time_seconds: float = 0.0


class WebsiteNavigationEngine:
    """Engine that systematically explores a website up to specified depth & page budget."""

    def __init__(
        self,
        max_depth: int = 2,
        max_pages: int = 8,
        timeout_ms: int = 20000,
        headless: bool = True
    ):
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.browser = BrowserNavigator(headless=headless, timeout_ms=timeout_ms)
        self.extractor = ContentExtractor()
        self.link_scorer = LinkRelevanceScorer()

    async def navigate_website(
        self,
        starting_url: str,
        query_keywords: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[str, int, int], Awaitable[None]]] = None
    ) -> NavigationSessionResult:
        """Navigates starting from a URL, crawling up to max_depth and max_pages.

        Args:
            starting_url: Entry point URL for website exploration.
            query_keywords: Optional keywords to prioritize which links to follow.
            progress_callback: Async hook (status_message, pages_crawled, total_max_pages).

        Returns:
            NavigationSessionResult containing all structured page extractions.
        """
        start_time = time.perf_counter()

        # Step 1: Validate and canonicalize starting URL
        normalized_root = validate_and_normalize_url(starting_url)
        root_domain = urllib.parse.urlparse(normalized_root).netloc.lower()

        visited_urls: Set[str] = set()
        extracted_pages: List[ExtractedContent] = []
        visit_records: List[PageVisitRecord] = []
        failed_urls: Dict[str, str] = {}

        # BFS Queue: stores CrawlQueueItem(url, depth)
        queue: deque[CrawlQueueItem] = deque()
        queue.append(CrawlQueueItem(url=normalized_root, depth=0))

        keywords = [k.lower() for k in (query_keywords or []) if len(k) > 2]

        while queue and len(visited_urls) < self.max_pages:
            current_item = queue.popleft()
            url = current_item.url
            depth = current_item.depth

            if url in visited_urls:
                continue

            if depth > self.max_depth:
                continue

            visited_urls.add(url)
            page_index = len(visited_urls)

            msg = f"Navigating (Page {page_index}/{self.max_pages}, Depth {depth}): {url}"
            logger.info(msg)
            if progress_callback:
                await progress_callback(msg, page_index, self.max_pages)

            t0 = time.perf_counter()
            html, status_code, method, error = await self.browser.fetch_page(url, wait_for_js=True)
            latency_ms = (time.perf_counter() - t0) * 1000

            if not html or status_code >= 400:
                err_msg = error or f"HTTP {status_code}"
                failed_urls[url] = err_msg
                visit_records.append(
                    PageVisitRecord(
                        url=url,
                        title="Failed",
                        status_code=status_code,
                        content_length=0,
                        extraction_method=method,
                        fetch_latency_ms=round(latency_ms, 2),
                        error_message=err_msg
                    )
                )
                continue

            # Extract structured content (title, headings, paragraphs, tables, links)
            doc = self.extractor.extract(html, url=url)
            extracted_pages.append(doc)

            visit_records.append(
                PageVisitRecord(
                    url=url,
                    title=doc.title,
                    status_code=status_code,
                    content_length=doc.text_length,
                    extraction_method=f"{method} + {doc.extraction_method}",
                    fetch_latency_ms=round(latency_ms, 2)
                )
            )

            # Discover and enqueue child links if we haven't reached maximum depth
            if depth < self.max_depth and len(visited_urls) < self.max_pages:
                remaining_budget = self.max_pages - len(visited_urls)
                # Prioritize links based on query relevance and internal domain
                candidate_links = self.link_scorer.prioritize_links(
                    links=doc.links,
                    query_keywords=keywords,
                    base_domain=root_domain,
                    limit=min(5, remaining_budget + 2)
                )

                for link_item in candidate_links:
                    child_url = link_item.url
                    if child_url not in visited_urls and not any(q.url == child_url for q in queue):
                        queue.append(CrawlQueueItem(url=child_url, depth=depth + 1, parent_url=url))

        elapsed = time.perf_counter() - start_time
        return NavigationSessionResult(
            starting_url=normalized_root,
            total_pages_visited=len(visited_urls),
            pages=extracted_pages,
            visit_records=visit_records,
            visited_urls=list(visited_urls),
            failed_urls=failed_urls,
            execution_time_seconds=round(elapsed, 2)
        )

