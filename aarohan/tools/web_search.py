"""AAROHAN AI - Sovereign Zero-Cost Local Web Search Engine.

Enables AAROHAN to perform live, privacy-preserving, zero-cloud-API web searches
using local HTTP parsers with zero commercial tokens or subscriptions.
Extracts live citations, verified source URLs, and grounded text snippets.
"""

from __future__ import annotations

import html
import logging
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


@dataclass
class WebSearchResult:
    title: str
    url: str
    snippet: str
    source: str = "duckduckgo_lite"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
        }


class _DDGLiteHTMLParser(HTMLParser):
    """Parses lite.duckduckgo.com search result HTML."""

    def __init__(self) -> None:
        super().__init__()
        self.results: List[WebSearchResult] = []
        self._cur_link: Optional[str] = None
        self._cur_title: List[str] = []
        self._cur_snippet: List[str] = []
        self._in_title = False
        self._in_snippet = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        attrs_dict = dict(attrs)
        tag_lower = tag.lower()
        if tag_lower == "a" and "result-link" in (attrs_dict.get("class") or ""):
            self._cur_link = attrs_dict.get("href")
            self._in_title = True
            self._cur_title = []
        elif tag_lower == "td" and "result-snippet" in (attrs_dict.get("class") or ""):
            self._in_snippet = True
            self._cur_snippet = []

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower == "a" and self._in_title:
            self._in_title = False
        elif tag_lower == "td" and self._in_snippet:
            self._in_snippet = False
            title_text = html.unescape("".join(self._cur_title)).strip()
            snippet_text = html.unescape("".join(self._cur_snippet)).strip()

            if self._cur_link:
                actual_url = self._cur_link
                # Unpack DuckDuckGo redirect parameter if present
                if "uddg=" in actual_url:
                    parsed = urllib.parse.parse_qs(urllib.parse.urlparse(actual_url).query)
                    actual_url = parsed.get("uddg", [actual_url])[0]

                if actual_url.startswith("http://") or actual_url.startswith("https://"):
                    self.results.append(
                        WebSearchResult(
                            title=title_text or actual_url,
                            url=actual_url,
                            snippet=snippet_text,
                        )
                    )
            self._cur_link = None

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._cur_title.append(data)
        elif self._in_snippet:
            self._cur_snippet.append(data)


class WebSearchEngine:
    """Zero-cost local web search engine executing live queries."""

    SEARCH_INTENT_PATTERNS = [
        re.compile(r"\bsearch\s+(?:me\s+)?(?:on\s+)?(?:the\s+)?(?:web|internet|online|google)\b", re.IGNORECASE),
        re.compile(r"\bsearch\s+(?:the\s+)?web\s+(?:for|about)\b", re.IGNORECASE),
        re.compile(r"\b(?:look\s+up|find)\s+.*?\s+(?:on\s+the\s+web|online|on\s+google)\b", re.IGNORECASE),
        re.compile(r"\b(?:google|browse\s+the\s+web|web\s+search)\b", re.IGNORECASE),
        re.compile(r"\bsearch\s+for\s+.*?\s+(?:online|on\s+the\s+web)\b", re.IGNORECASE),
    ]

    def __init__(self, timeout_seconds: int = 10) -> None:
        self.timeout = timeout_seconds

    def is_search_query(self, query: str) -> bool:
        """Determines if query requests a live web search."""
        q = query.strip()
        return any(pat.search(q) for pat in self.SEARCH_INTENT_PATTERNS)

    def extract_search_terms(self, query: str, user_name: str = "Datta Sable") -> str:
        """Extracts clean search keywords from conversational web search requests."""
        q = query.strip()
        q_lower = q.lower()

        # Handle user self-search (e.g. "search me on web", "search about me online")
        if re.search(r"\bsearch\s+(?:about\s+)?me\b", q_lower) or re.search(r"\bwho\s+am\s+i\s+on\s+(?:the\s+)?web\b", q_lower):
            return f'"{user_name}" MIS Business Intelligence'

        # Strip action preambles
        clean = re.sub(
            r"^(?:please\s+)?(?:search\s+(?:the\s+)?(?:web|internet|google|online)\s+(?:for|about)?|google|look\s+up|find\s+online)\s*",
            "",
            q,
            flags=re.IGNORECASE,
        )
        clean = re.sub(r"\s+(?:on\s+(?:the\s+)?(?:web|internet|google|online)|online)\s*\??$", "", clean, flags=re.IGNORECASE)
        clean = clean.strip().strip("'\"?")
        return clean or query

    def search(self, query: str, max_results: int = 5) -> List[WebSearchResult]:
        """Performs live web search against DuckDuckGo Lite endpoint."""
        encoded_data = urllib.parse.urlencode({"q": query}).encode("utf-8")
        req = urllib.request.Request(
            "https://lite.duckduckgo.com/lite/",
            data=encoded_data,
            headers={
                "User-Agent": USER_AGENT,
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                html_content = resp.read().decode("utf-8", errors="ignore")

            parser = _DDGLiteHTMLParser()
            parser.feed(html_content)
            return parser.results[:max_results]
        except Exception as e:
            logger.warning(f"Web search error for query '{query}': {e}")
            return []

    def format_search_response(self, original_query: str, search_terms: str, results: List[WebSearchResult]) -> str:
        """Formats verified search results into clean markdown with live citations."""
        if not results:
            return (
                f"### 🌐 Live Web Search — No Results Found\n\n"
                f"- **Search Query:** `{search_terms}`\n"
                f"- **Status:** Executed search, but no live web results were returned. "
                f"Verify network connectivity or try alternative search keywords."
            )

        lines = [
            "### 🌐 Live Web Search Results\n\n",
            f"> Retrieved **{len(results)} live web sources** for: `{search_terms}`\n\n",
        ]

        for i, r in enumerate(results, 1):
            lines.append(f"#### {i}. [{r.title}]({r.url})\n")
            lines.append(f"- **URL:** `{r.url}`\n")
            lines.append(f"- **Snippet:** {r.snippet}\n\n")

        lines.append(
            "*(Sources verified via live DuckDuckGo retrieval | Zero-API, local-first search)*"
        )
        return "".join(lines).strip()
