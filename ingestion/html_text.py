"""Turn Stack Exchange post HTML into plain text for retrieval and prompting.

Post bodies are HTML fragments: paragraphs, lists, blockquotes and fenced code.
Feeding raw HTML to the embedder wastes tokens on tag names, and feeding it to
the LLM invites it to echo markup back at the user. We only need readable text
with the block structure preserved, which the standard library can do, so this
stays dependency-free.
"""

import re
from html.parser import HTMLParser

# Elements after which a line break belongs.
BLOCK_TAGS = {
    "p", "div", "pre", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "ul", "ol", "tr", "table", "hr", "br",
}

# Elements whose text content is noise.
SKIP_TAGS = {"script", "style"}

_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING_SPACE = re.compile(r"[ \t]+\n")


class _TextExtractor(HTMLParser):
    def __init__(self):
        # convert_charrefs=True means &amp; and &#39; arrive already decoded.
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "li":
            self.parts.append("\n- ")
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        # `li` is excluded: the next item's start tag already opens a new line,
        # and closing one here would put a blank line between every bullet.
        if tag in BLOCK_TAGS and tag != "li":
            self.parts.append("\n")

    def handle_startendtag(self, tag, attrs):
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0:
            self.parts.append(data)

    def text(self) -> str:
        joined = "".join(self.parts)
        joined = _TRAILING_SPACE.sub("\n", joined)
        # Two consecutive newlines read as a paragraph break; more is just noise.
        joined = _BLANK_LINES.sub("\n\n", joined)
        return joined.strip()


def html_to_text(html: str) -> str:
    if not html:
        return ""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return parser.text()


def truncate(text: str, max_chars: int) -> str:
    """Cut long text at a sentence-ish boundary so prompts stay predictable."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    # Prefer breaking at the last paragraph, then the last sentence.
    for boundary in ("\n\n", ". "):
        index = cut.rfind(boundary)
        if index > max_chars * 0.5:
            return cut[:index].rstrip() + " [...]"
    return cut.rstrip() + " [...]"
