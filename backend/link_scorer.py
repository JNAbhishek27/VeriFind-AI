"""Link relevance evaluation and prioritization engine.

Scores hyperlinks discovered during crawling to decide which links are worth
following based on user query intent, semantic cues in the URL, and anchor text.
"""

from typing import List, Optional
import urllib.parse
import re
from backend.extractor import LinkItem


class LinkRelevanceScorer:
    """Evaluates and ranks candidate hyperlinks."""

    # Disregard common binary, asset, media, or non-informative path patterns
    IGNORE_EXTENSIONS = {
        ".pdf", ".zip", ".tar", ".gz", ".png", ".jpg", ".jpeg", ".gif",
        ".svg", ".mp4", ".mp3", ".wav", ".css", ".js", ".json", ".xml"
    }

    IGNORE_PATH_KEYWORDS = {
        "login", "signin", "signup", "register", "logout", "cart", "checkout",
        "privacy", "terms", "cookies", "legal", "subscribe", "wp-content"
    }

    def score_link(self, link: LinkItem, query_keywords: List[str], base_domain: str) -> float:
        """Assigns a relevance priority score in range [0.0, 1.0].

        Args:
            link: The LinkItem to evaluate.
            query_keywords: Key intent terms from the user question.
            base_domain: Origin domain for scope prioritization.

        Returns:
            Relevance score between 0.0 and 1.0.
        """
        parsed = urllib.parse.urlparse(link.url)
        path_lower = parsed.path.lower()

        # Reject disallowed file extensions
        for ext in self.IGNORE_EXTENSIONS:
            if path_lower.endswith(ext):
                return 0.0

        # Penalize administrative / non-content pages
        for kw in self.IGNORE_PATH_KEYWORDS:
            if kw in path_lower:
                return 0.05

        score = 0.20  # Base crawl value for any valid content page

        # Priority boost for internal links
        if link.is_internal or parsed.netloc.lower() == base_domain.lower():
            score += 0.20

        # Anchor text and path semantic matching
        anchor_lower = link.text.lower()
        combined_text = f"{anchor_lower} {path_lower}"

        if query_keywords:
            matched_keywords = sum(1 for kw in query_keywords if kw.lower() in combined_text)
            match_ratio = matched_keywords / len(query_keywords)
            score += match_ratio * 0.50
        else:
            # If no query keywords provided, prefer informative anchor text
            if len(anchor_lower.split()) >= 2:
                score += 0.20

        return round(min(1.0, max(0.0, score)), 3)

    def prioritize_links(
        self,
        links: List[LinkItem],
        query_keywords: List[str],
        base_domain: str,
        limit: int = 5
    ) -> List[LinkItem]:
        """Filters and ranks links, returning the top candidates."""
        scored = []
        for l in links:
            s = self.score_link(l, query_keywords, base_domain)
            if s > 0.15:
                scored.append((l, s))

        # Sort descending by relevance score
        scored.sort(key=lambda item: item[1], reverse=True)
        return [l for l, _ in scored[:limit]]

