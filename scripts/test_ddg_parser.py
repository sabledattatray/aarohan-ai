import urllib.request
import urllib.parse
from html.parser import HTMLParser

class DDGLiteParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results = []
        self._cur_link = None
        self._cur_title = []
        self._cur_snippet = []
        self._in_title = False
        self._in_snippet = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "a" and "result-link" in attrs_dict.get("class", ""):
            self._cur_link = attrs_dict.get("href")
            self._in_title = True
            self._cur_title = []
        elif tag == "td" and "result-snippet" in attrs_dict.get("class", ""):
            self._in_snippet = True
            self._cur_snippet = []

    def handle_endtag(self, tag):
        if tag == "a" and self._in_title:
            self._in_title = False
        elif tag == "td" and self._in_snippet:
            self._in_snippet = False
            title_text = "".join(self._cur_title).strip()
            snippet_text = "".join(self._cur_snippet).strip()
            if self._cur_link:
                # Clean duckduckgo redirect if present
                actual_url = self._cur_link
                if "uddg=" in actual_url:
                    parsed = urllib.parse.parse_qs(urllib.parse.urlparse(actual_url).query)
                    actual_url = parsed.get("uddg", [actual_url])[0]
                self.results.append({
                    "title": title_text,
                    "url": actual_url,
                    "snippet": snippet_text,
                })
            self._cur_link = None

    def handle_data(self, data):
        if self._in_title:
            self._cur_title.append(data)
        elif self._in_snippet:
            self._cur_snippet.append(data)

def search_ddg_lite(query: str, max_results: int = 5):
    encoded_q = urllib.parse.urlencode({"q": query})
    url = f"https://lite.duckduckgo.com/lite/"
    data = encoded_q.encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Content-Type": "application/x-www-form-urlencoded",
        }
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        html = resp.read().decode("utf-8", errors="ignore")

    parser = DDGLiteParser()
    parser.feed(html)
    return parser.results[:max_results]

if __name__ == "__main__":
    res = search_ddg_lite("Datta Sable MIS")
    print(f"Extracted {len(res)} results:")
    for r in res:
        print(f"- Title: {r['title']}")
        print(f"  URL: {r['url']}")
        print(f"  Snippet: {r['snippet'][:100]}...\n")
