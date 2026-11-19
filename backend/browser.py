"""Browser automation and web navigation engine for VeriFind AI.

Features:
- Validates URLs and strictly restricts navigation to HTTP/HTTPS.
- Serverless-aware: when VERIFIND_SERVERLESS=true (Vercel/cloud functions),
  skips Playwright entirely and uses httpx directly (no browser binaries needed).
- When running locally, uses headless Playwright Chromium with httpx fallback.
- Retries with exponential backoff on transient network timeouts/failures.
- Enforces navigation timeouts and resource controls.
"""

import os
import asyncio
import logging
import urllib.parse
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Detect serverless environment — set by api/index.py on Vercel
_SERVERLESS = os.environ.get("VERIFIND_SERVERLESS", "").lower() in ("1", "true", "yes")

# Lazily import Playwright only in non-serverless mode to avoid import errors
# when the playwright package is not installed (Vercel runtime).
if not _SERVERLESS:
    try:
        from playwright.async_api import async_playwright, Error as PlaywrightError
        _PLAYWRIGHT_AVAILABLE = True
    except ImportError:
        _PLAYWRIGHT_AVAILABLE = False
        logger.warning("Playwright not installed; falling back to httpx-only scraping.")
else:
    _PLAYWRIGHT_AVAILABLE = False
    logger.info("Serverless mode: Playwright disabled. Using httpx-only scraping.")


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
    """Web navigator: Playwright (local) or httpx (serverless/fallback)."""

    DEFAULT_USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36 VeriFindAI/0.3.0"
    )

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
        self.user_agent = user_agent or self.DEFAULT_USER_AGENT

    async def fetch_page(
        self,
        url: str,
        wait_for_js: bool = True
    ) -> Tuple[Optional[str], int, str, Optional[str]]:
        """Fetches a URL and returns its HTML content.

        In serverless mode, uses httpx directly (no Playwright).
        Locally, tries Playwright first, then falls back to httpx.

        Returns:
            Tuple: (html_content, status_code, fetch_method, error_message)
        """
        try:
            validated_url = validate_and_normalize_url(url)
        except ValueError as val_err:
            return None, 400, "invalid_url", str(val_err)

        if _SERVERLESS or not _PLAYWRIGHT_AVAILABLE:
            # Serverless path: httpx only
            return await self._fetch_httpx(validated_url)

        # Local path: Playwright with httpx fallback
        last_error: Optional[str] = None
        for attempt in range(1, self.max_retries + 2):
            try:
                html, status = await self._navigate_playwright(validated_url, wait_for_js)
                if html:
                    return html, status, "playwright_chromium", None
            except Exception as pw_err:
                last_error = str(pw_err)
                logger.warning(
                    "Playwright attempt %d/%d failed for %s: %s",
                    attempt, self.max_retries + 1, validated_url, pw_err
                )
                if attempt <= self.max_retries:
                    await asyncio.sleep(1.0 * attempt)

        logger.info("Attempting httpx fallback for: %s", validated_url)
        html, status, method, err = await self._fetch_httpx(validated_url)
        if err:
            err = f"Playwright: {last_error} | HTTP: {err}"
        return html, status, method, err

    async def _fetch_httpx(self, url: str) -> Tuple[Optional[str], int, str, Optional[str]]:
        """Fetches a page via httpx (no browser required)."""
        import httpx
        try:
            async with httpx.AsyncClient(
                headers={"User-Agent": self.user_agent},
                timeout=self.timeout_ms / 1000.0,
                follow_redirects=True,
                verify=False
            ) as client:
                resp = await client.get(url)
                return resp.text, resp.status_code, "httpx", None
        except Exception as http_err:
            logger.error("httpx fetch failed for %s: %s", url, http_err)
            return None, 0, "failed", str(http_err)

    async def _navigate_playwright(self, url: str, wait_for_js: bool) -> Tuple[str, int]:
        """Internal helper executing Playwright browser navigation."""
        async with async_playwright() as p:
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
            # Block heavy media resources for fast retrieval
            await context.route(
                "**/*.{png,jpg,jpeg,gif,webp,svg,woff,woff2,ttf,eot,mp4,avi,mov}",
                lambda route: route.abort()
            )
            page = await context.new_page()
            response = await page.goto(url, timeout=self.timeout_ms, wait_until="domcontentloaded")

            if wait_for_js:
                try:
                    await page.wait_for_load_state("networkidle", timeout=min(4000, self.timeout_ms))
                except Exception:
                    pass  # networkidle timeout is safe to ignore

            status = response.status if response else 200
            content = await page.content()
            await context.close()
            await browser.close()
            return content, status
