"""Live demonstration script for the VeriFind AI Website Navigation Engine.

Demonstrates:
- Validating and accepting starting URL.
- Playwright Chromium headless automation on public web sites.
- Extraction of Titles, Headings, Paragraphs, Tables, and Links.
- Multi-depth (up to 2 levels) navigation with a max page budget.
- Tracking visited URLs and generating structured output.
"""

import asyncio
import json
import sys
from backend.navigation_engine import WebsiteNavigationEngine


async def demonstrate_navigation(target_url: str):
    print("=" * 70)
    print(f"VERIFIND AI - WEBSITE NAVIGATION & EXTRACTION DEMO")
    print(f"Target Starting URL: {target_url}")
    print("=" * 70)

    engine = WebsiteNavigationEngine(
        max_depth=2,
        max_pages=4,      # Demonstrate multi-page crawl budget
        timeout_ms=15000,
        headless=True
    )

    async def on_progress(msg, current, total):
        print(f"[*] [{current}/{total}] {msg}")

    print("\nStarting autonomous navigation...")
    result = await engine.navigate_website(
        starting_url=target_url,
        query_keywords=["python", "documentation", "download", "community"],
        progress_callback=on_progress
    )

    print("\n" + "=" * 70)
    print("NAVIGATION SUMMARY")
    print("=" * 70)
    print(f"Total Pages Visited: {result.total_pages_visited}")
    print(f"Total Execution Time: {result.execution_time_seconds:.2f}s")
    print(f"Visited URLs:")
    for idx, u in enumerate(result.visited_urls, 1):
        print(f"  {idx}. {u}")

    print("\n" + "=" * 70)
    print("STRUCTURED PAGE CONTENT SAMPLES")
    print("=" * 70)
    for idx, page in enumerate(result.pages, 1):
        print(f"\n--- Page #{idx}: {page.title} ---")
        print(f"URL: {page.url}")
        print(f"Method: {page.extraction_method} | Content Bytes: {page.text_length}")

        # Headings
        if page.headings:
            print("Top Headings:")
            for h in page.headings[:4]:
                print(f"  - [H{h.level}] {h.text}")

        # Tables
        if page.tables:
            print(f"Tables Found: {len(page.tables)}")
            for t_idx, t in enumerate(page.tables[:2], 1):
                caption_str = f" ('{t.caption}')" if t.caption else ""
                print(f"  Table #{t_idx}{caption_str}: Headers={t.headers} | Row Count={len(t.rows)}")
                if t.rows:
                    print(f"    Sample Row: {t.rows[0]}")
        else:
            print("Tables: None on this page")

        # Paragraphs
        if page.paragraphs:
            print(f"Paragraphs Extracted: {len(page.paragraphs)}")
            print(f"  Excerpt: \"{page.paragraphs[0][:140]}...\"")

        # Outbound Links
        if page.links:
            print(f"Discovered Valid Links: {len(page.links)} total")
            for link in page.links[:3]:
                print(f"  -> [{link.text or 'No Anchor Text'}] ({link.url})")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "https://quotes.toscrape.com"
    asyncio.run(demonstrate_navigation(target))

