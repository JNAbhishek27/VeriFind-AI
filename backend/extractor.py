"""Content extraction module using Trafilatura and BeautifulSoup.

Extracts structured text:
- Page title
- Headings (h1, h2, h3, h4)
- Paragraphs
- Tables (header rows, rows, and structured cells)
- Outbound hyperlinks with anchor text and context
Filters out scripts, styles, ad banners, and boilerplate navigation where appropriate.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import re
import urllib.parse
import trafilatura
from bs4 import BeautifulSoup


@dataclass
class HeadingItem:
    """Represents a heading tag on the page."""
    level: int  # 1 for h1, 2 for h2, etc.
    text: str


@dataclass
class TableData:
    """Structured representation of an HTML table."""
    headers: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    caption: Optional[str] = None


@dataclass
class LinkItem:
    """Extracted link with anchor text, resolved target URL, and context."""
    url: str
    text: str
    is_internal: bool = True
    context: str = ""


@dataclass
class ExtractedContent:
    """Structured container for all extracted page contents."""
    url: str
    title: str
    headings: List[HeadingItem] = field(default_factory=list)
    paragraphs: List[str] = field(default_factory=list)
    tables: List[TableData] = field(default_factory=list)
    links: List[LinkItem] = field(default_factory=list)
    main_text: str = ""
    extraction_method: str = "trafilatura+bs4"
    text_length: int = 0


class ContentExtractor:
    """Extracts clean headings, paragraphs, tables, and links from raw HTML."""

    def extract(self, html_content: str, url: str) -> ExtractedContent:
        """Extracts structured content from raw HTML.

        Args:
            html_content: Raw HTML text of the web page.
            url: Origin URL of the document for relative link resolution.

        Returns:
            ExtractedContent with titles, headings, paragraphs, tables, and links.
        """
        if not html_content or not html_content.strip():
            return ExtractedContent(
                url=url,
                title="Empty Document",
                headings=[],
                paragraphs=[],
                tables=[],
                links=[],
                main_text="",
                text_length=0
            )

        soup = BeautifulSoup(html_content, "html.parser")

        # 1. Extract Title
        title = "Untitled"
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.find("h1"):
            h1 = soup.find("h1")
            if h1:
                title = h1.get_text(strip=True)

        # 2. Extract Headings (before decomposing unwanted tags)
        headings: List[HeadingItem] = []
        for tag in soup.find_all(["h1", "h2", "h3", "h4"]):
            htext = " ".join(tag.get_text().split())
            if htext:
                level = int(tag.name[1])
                headings.append(HeadingItem(level=level, text=htext))

        # 3. Extract Tables
        tables: List[TableData] = []
        for table_tag in soup.find_all("table"):
            table_obj = self._parse_table(table_tag)
            if table_obj.rows or table_obj.headers:
                tables.append(table_obj)

        # 4. Extract Outbound Links (before stripping boilerplate)
        links = self._extract_links(soup, base_url=url)

        # 5. Extract Main Article Body via Trafilatura
        extracted_text = trafilatura.extract(
            html_content,
            include_comments=False,
            include_tables=True,
            no_fallback=False
        )

        method = "trafilatura+bs4"
        if not extracted_text:
            method = "bs4_fallback"
            for tag in soup(["script", "style", "nav", "footer", "aside", "noscript"]):
                tag.decompose()
            extracted_text = soup.get_text(separator="\n", strip=True)

        # 6. Extract Paragraphs
        paragraphs = self._split_paragraphs(extracted_text, soup)

        return ExtractedContent(
            url=url,
            title=title,
            headings=headings,
            paragraphs=paragraphs,
            tables=tables,
            links=links,
            main_text=extracted_text or "",
            extraction_method=method,
            text_length=len(extracted_text or "")
        )

    def _parse_table(self, table_tag) -> TableData:
        """Parses a <table> BeautifulSoup element into TableData."""
        headers: List[str] = []
        rows: List[List[str]] = []
        caption = None

        caption_tag = table_tag.find("caption")
        if caption_tag:
            caption = " ".join(caption_tag.get_text().split())

        # Extract headers from <th> tags
        th_tags = table_tag.find_all("th")
        if th_tags:
            headers = [" ".join(th.get_text().split()) for th in th_tags]

        # Extract rows from <tr>
        tr_tags = table_tag.find_all("tr")
        for tr in tr_tags:
            cells = tr.find_all(["td", "th"])
            if not cells:
                continue
            row_vals = [" ".join(c.get_text().split()) for c in cells]
            # Avoid duplicating headers in rows if <tr> contains only <th>
            if tr.find_all("td"):
                rows.append(row_vals)
            elif not headers and tr.find_all("th"):
                headers = row_vals

        return TableData(headers=headers, rows=rows, caption=caption)

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[LinkItem]:
        """Extracts valid, resolved HTTP/HTTPS links with anchor text."""
        links: List[LinkItem] = []
        seen_urls = set()
        base_parsed = urllib.parse.urlparse(base_url)

        for a_tag in soup.find_all("a", href=True):
            raw_href = a_tag["href"].strip()
            if not raw_href or raw_href.startswith(("#", "javascript:", "mailto:", "tel:", "data:")):
                continue

            # Resolve relative link to absolute URL
            resolved_url = urllib.parse.urljoin(base_url, raw_href)
            parsed_res = urllib.parse.urlparse(resolved_url)

            # Restrict strictly to HTTP and HTTPS
            if parsed_res.scheme not in ("http", "https"):
                continue

            # Strip fragments for deduplication
            clean_url = urllib.parse.urlunparse((
                parsed_res.scheme,
                parsed_res.netloc,
                parsed_res.path,
                parsed_res.params,
                parsed_res.query,
                ""
            ))

            if clean_url in seen_urls:
                continue
            seen_urls.add(clean_url)

            anchor_text = " ".join(a_tag.get_text().split())
            is_internal = parsed_res.netloc.lower() == base_parsed.netloc.lower()

            links.append(
                LinkItem(
                    url=clean_url,
                    text=anchor_text,
                    is_internal=is_internal,
                    context=anchor_text
                )
            )

        return links

    def _split_paragraphs(self, trafilatura_text: Optional[str], soup: BeautifulSoup) -> List[str]:
        """Combines paragraphs from Trafilatura and <p> elements."""
        cleaned_paras: List[str] = []
        seen = set()

        if trafilatura_text:
            for p in trafilatura_text.split("\n"):
                clean = " ".join(p.split())
                if len(clean) >= 20 and clean not in seen:
                    cleaned_paras.append(clean)
                    seen.add(clean)

        # Fallback / supplement with <p> tags if Trafilatura extracted very little
        if len(cleaned_paras) < 2:
            for p_tag in soup.find_all("p"):
                clean = " ".join(p_tag.get_text().split())
                if len(clean) >= 20 and clean not in seen:
                    cleaned_paras.append(clean)
                    seen.add(clean)

        return cleaned_paras
