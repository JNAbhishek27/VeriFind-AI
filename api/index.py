"""VeriFind AI — Vercel Serverless Entry Point.

Vercel's @vercel/python runtime detects FastAPI/Starlette ASGI apps automatically.
This file is the single handler for all /api/* routes.

Constraints handled:
- VERIFIND_SERVERLESS=true disables Playwright (no browser binaries on Vercel).
- Read-only filesystem (no Playwright cache writes).
- Python 3.12 runtime.
- 60-second max duration (set in vercel.json).
"""

import sys
import os

# Ensure the project root is on sys.path so `backend.*` imports resolve.
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

# Force serverless mode — disables Playwright, enables httpx-only scraping.
# Also set via vercel.json `env` block, but default here as a safety net.
os.environ.setdefault("VERIFIND_SERVERLESS", "true")

# Import the FastAPI app.  Vercel detects the ASGI app named `app`.
from backend.api import app  # noqa: F401, E402
