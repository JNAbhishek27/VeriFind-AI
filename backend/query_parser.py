"""Query parser and intent normalization module for VeriFind AI.

Extracts core semantic keywords, expected answer types, and constraints from
natural language inputs.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import re


@dataclass
class ParsedQuery:
    """Normalized query representation with extracted constraints and intent."""
    raw_query: str
    cleaned_query: str
    target_entity: Optional[str] = None
    expected_type: str = "general"  # e.g., 'fact', 'number', 'definition', 'list'
    keywords: List[str] = field(default_factory=list)
    site_constraint: Optional[str] = None


class QueryParser:
    """Parses natural-language queries into structured retrieval representations."""

    STOPWORDS = {
        "what", "is", "the", "a", "an", "who", "where", "when", "how", "why",
        "of", "in", "for", "on", "with", "at", "by", "from", "up", "about",
        "into", "over", "after", "can", "could", "would", "should", "does",
        "did", "do", "are", "tell", "me", "find", "show", "give"
    }

    QUESTION_PATTERNS = [
        (re.compile(r"^(when|what year|what date)", re.IGNORECASE), "temporal"),
        (re.compile(r"^(how much|how many|what percentage|price|cost)", re.IGNORECASE), "numeric"),
        (re.compile(r"^(who|which person|which company|which founder)", re.IGNORECASE), "entity"),
        (re.compile(r"^(where|which country|which city|which place)", re.IGNORECASE), "location"),
        (re.compile(r"^(what is|define|explain)", re.IGNORECASE), "definition"),
    ]

    def parse(self, query: str, starting_url: Optional[str] = None) -> ParsedQuery:
        """Parses a user input query string.

        Args:
            query: The raw query input string.
            starting_url: An optional specific URL constraint provided by the user.

        Returns:
            ParsedQuery object with normalized components.

        Raises:
            ValueError: If the query is empty or contains only whitespace.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        cleaned = " ".join(query.strip().split())
        expected_type = "general"

        for pattern, qtype in self.QUESTION_PATTERNS:
            if pattern.search(cleaned):
                expected_type = qtype
                break

        tokens = re.findall(r"\b[A-Za-z0-9_-]+\b", cleaned)
        keywords = [
            token for token in tokens
            if token.lower() not in self.STOPWORDS and len(token) > 2
        ]

        # Extract domain constraint if provided
        site_constraint = starting_url.strip() if starting_url and starting_url.strip() else None

        return ParsedQuery(
            raw_query=query,
            cleaned_query=cleaned,
            expected_type=expected_type,
            keywords=keywords if keywords else tokens,
            site_constraint=site_constraint
        )

