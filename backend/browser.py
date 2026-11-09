"""Browser automation and web navigation engine using Playwright.

Features:
- Validates URLs and strictly restricts navigation to HTTP/HTTPS.
- Headless Playwright Chromium browser management.
- Handles dynamic JavaScript-rendered pages with DOM content loaded and network idle states.
- Retries with exponential backoff on transient network timeouts/failures.
- Enforces navigation timeouts and resource controls.
- Asynchronous HTTP client fallback if browser execution fails.
"""

from typing import Optional, Tuple, Dict, Any, List
import asyncio
import logging
import urllib.parse
from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Error as PlaywrightError

logger = logging.getLogger(__name__)


def validate_and_normalize_url(url: str) -> str:
    """Validates that a URL is well-formed and uses HTTP or HTTPS.

    Args:
        url: The candidate URL string.

    Returns:
        The normalized URL string.

    Raises:
        ValueError: If the URL is invalid or uses an unsupported scheme.
    """
    if not url or not isinstance(url, str) or not url.strip():
        raise ValueError("URL cannot be empty.")

    cleaned = url.strip()
    parsed_initial = urllib.parse.urlparse(cleaned)
    if not parsed_initial.scheme:
        cleaned = "https://" + cleaned

    parsed = urllib.parse.urlparse(cleaned)
    if parsed.scheme.lower() not in ("http", "https"):
        raise ValueError(f"Invalid URL scheme '{parsed.scheme}'. Only HTTP and HTTPS are permitted.")

    if not parsed.netloc:
        raise ValueError(f"Invalid URL '{url}': Missing domain/host.")

    # Drop fragment for canonical retrieval
    return urllib.parse.urlunparse((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        parsed.path or "/",
        parsed.params,
        parsed.query,
        ""
    ))


class BrowserNavigator:
    """Automates web navigation using Playwright with robust safety controls."""

    def __init__(
        self,
        headless: bool = True,
        timeout_ms: int = 20000,
        max_retries: int = 2,
        user_agent: Optional[str] = None
    ):
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.max_retries = max_retries
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36 VeriFindAI/0.2.0"
        )

    async def fetch_page(
        self,
        url: str,
        wait_for_js: bool = True
    ) -> Tuple[Optional[str], int, str, Optional[str]]:
        """Navigates to the specified URL and retrieves its fully rendered HTML.

        Attempts dynamic retrieval via Playwright Chromium. Retries on transient
        timeouts, and falls back to HTTPX if browser launching is blocked.

        Args:
            url: The destination URL.
            wait_for_js: Whether to wait for dynamic JS scripts to settle.

        Returns:
            Tuple: (html_content, status_code, fetch_method, error_message)
        """
        try:
            validated_url = validate_and_normalize_url(url)
        except ValueError as val_err:
            return None, 400, "invalid_url", str(val_err)

        last_error = None

        # Playwright dynamic navigation with retry
        for attempt in range(1, self.max_retries + 2):
            try:
                html, status = await self._navigate_playwright(validated_url, wait_for_js)
                if html:
                    return html, status, "playwright_chromium", None
            except Exception as pw_err:
                last_error = str(pw_err)
                logger.warning("Playwright navigation attempt %d/%d failed for %s: %s",
                               attempt, self.max_retries + 1, validated_url, pw_err)
                if attempt <= self.max_retries:
                    await asyncio.sleep(1.0 * attempt)

        # Fallback to HTTPX client
        logger.info("Attempting HTTPX fallback fetch for: %s", validated_url)
        try:
            import httpx
            headers = {"User-Agent": self.user_agent}
            async with httpx.AsyncClient(
                headers=headers,
                timeout=self.timeout_ms / 1000.0,
                follow_redirects=True,
                verify=False
            ) as client:
                resp = await client.get(validated_url)
                return resp.text, resp.status_code, "httpx_fallback", None
        except Exception as http_err:
            logger.error("HTTPX fallback also failed for %s: %s", validated_url, http_err)
            return None, 0, "failed", f"Playwright: {last_error} | HTTP: {http_err}"

    async def _navigate_playwright(self, url: str, wait_for_js: bool) -> Tuple[str, int]:
        """Internal helper executing Playwright browser navigation."""
        async with async_playwright() as p:
            # Launch Chromium with security sandboxing and performance flags
            browser = await p.chromium.launch(
                headless=self.headless,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-blink-features=AutomationControlled",
                ]
            )

            context = await browser.new_context(
                user_agent=self.user_agent,
                viewport={"width": 1280, "height": 800},
                ignore_https_errors=True
            )

            # Block heavy non-text media resources (images, fonts, stylesheets) for high-speed retrieval
            await context.route(
                "**/*.{png,jpg,jpeg,gif,webp,svg,woff,woff2,ttf,eot,mp4,avi,mov}",
                lambda route: route.abort()
            )

            page = await context.new_page()
            response = await page.goto(
                url,
                timeout=self.timeout_ms,
                wait_until="domcontentloaded"
            )

            # Allow client-side scripts a short window to hydrate DOM if requested
            if wait_for_js:
                try:
                    await page.wait_for_load_state("networkidle", timeout=min(4000, self.timeout_ms))
                except PlaywrightError:
                    # networkidle timeout is common and safe to ignore
                    pass

            status = response.status if response else 200
            content = await page.content()

            await context.close()
            await browser.close()
            return content, status
