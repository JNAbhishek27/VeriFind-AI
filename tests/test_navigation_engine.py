import pytest
import asyncio
from unittest.mock import patch
from backend.navigation_engine import WebsiteNavigationEngine


def test_navigation_enforces_max_depth_and_max_pages():
    async def _run():
        engine = WebsiteNavigationEngine(max_depth=2, max_pages=4)

        mock_pages = {
            "https://test.site/": (
                """
                <html><head><title>Root Page</title></head><body>
                <h1>Root</h1><p>Welcome to test root.</p>
                <a href="/page1">Go to Page 1</a>
                <a href="/page2">Go to Page 2</a>
                </body></html>
                """, 200, "mock", None
            ),
            "https://test.site/page1": (
                """
                <html><head><title>Page 1</title></head><body>
                <h1>Page 1</h1><p>Details for page 1.</p>
                <a href="/page1/subpage">Subpage Level 2</a>
                </body></html>
                """, 200, "mock", None
            ),
            "https://test.site/page2": (
                """
                <html><head><title>Page 2</title></head><body>
                <h1>Page 2</h1><p>Details for page 2.</p>
                <a href="/page2/subpage">Subpage 2 Level 2</a>
                </body></html>
                """, 200, "mock", None
            ),
            "https://test.site/page1/subpage": (
                """
                <html><head><title>Page 1 Subpage</title></head><body>
                <h1>Level 2 Subpage</h1><p>Deep content.</p>
                <a href="/page1/subpage/level3">Level 3 Link Should Be Ignored</a>
                </body></html>
                """, 200, "mock", None
            ),
        }

        async def mock_fetch(url, wait_for_js=True):
            return mock_pages.get(url, ("<html><body>Not found</body></html>", 404, "mock", "Not Found"))

        with patch.object(engine.browser, "fetch_page", side_effect=mock_fetch):
            result = await engine.navigate_website(starting_url="https://test.site/")

            assert result.total_pages_visited <= 4
            assert len(result.visited_urls) <= 4
            assert "https://test.site/" in result.visited_urls
            # Level 3 links should never be visited
            assert "https://test.site/page1/subpage/level3" not in result.visited_urls

    asyncio.run(_run())


def test_navigation_records_visited_urls_and_prevents_loops():
    async def _run():
        engine = WebsiteNavigationEngine(max_depth=2, max_pages=5)

        # Circular link graph: A -> B -> A
        mock_pages = {
            "https://circle.site/a": (
                '<html><head><title>A</title></head><body><h1>A</h1><p>Content A</p><a href="/b">Go to B</a></body></html>',
                200, "mock", None
            ),
            "https://circle.site/b": (
                '<html><head><title>B</title></head><body><h1>B</h1><p>Content B</p><a href="/a">Back to A</a></body></html>',
                200, "mock", None
            )
        }

        async def mock_fetch(url, wait_for_js=True):
            return mock_pages.get(url, ("<html><body>404</body></html>", 404, "mock", "Not Found"))

        with patch.object(engine.browser, "fetch_page", side_effect=mock_fetch):
            result = await engine.navigate_website(starting_url="https://circle.site/a")

            # Must not get stuck in an infinite loop
            assert result.total_pages_visited == 2
            assert set(result.visited_urls) == {"https://circle.site/a", "https://circle.site/b"}

    asyncio.run(_run())
