"""VeriFind AI — Vercel Serverless Entry Point.

This file is the single Python handler for all /api/* routes on Vercel.
It wraps the FastAPI app so Vercel's @vercel/python runtime can serve it.

Vercel constraints addressed here:
- No Playwright (no browser binaries in serverless). Uses httpx-only scraping.
- Read-only filesystem (no Playwright cache writes).
- 50 MB Lambda limit.
- 10-second default timeout (configurable to 60s on Pro plan).
"""

import sys
import os

# Ensure the project root is on the Python path so backend.* imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force serverless scraping mode (disables Playwright, uses httpx only)
os.environ.setdefault("VERIFIND_SERVERLESS", "true")

from backend.api import app  # noqa: E402  — FastAPI app

# Vercel expects the ASGI app to be named `app` or exported as the module default
# The @vercel/python runtime detects FastAPI/Starlette ASGI apps automatically.

