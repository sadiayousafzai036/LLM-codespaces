"""Tests for the post-HTML cleaner.

These matter more than they look. Stack Exchange bodies are full of entities and
code blocks, and a cleaner that quietly leaves markup in place degrades both
retrieval and the prompt without ever raising an error.
"""

from ingestion.html_text import html_to_text, truncate


def test_decodes_entities():
    assert html_to_text("<p>85&ndash;95&deg;C &amp; rising</p>") == "85–95°C & rising"


def test_strips_tags_but_keeps_text():
    text = html_to_text("<p>Use <strong>cold</strong> butter</p>")
    assert text == "Use cold butter"
    assert "<" not in text


def test_drops_script_and_style_content():
    text = html_to_text("<p>Keep</p><script>var x = 1;</script><style>p{}</style>")
    assert text == "Keep"


def test_list_items_become_single_spaced_bullets():
    text = html_to_text("<ul><li>One</li><li>Two</li></ul>")
    assert text == "- One\n- Two"


def test_preserves_paragraph_breaks_without_piling_them_up():
    text = html_to_text("<p>A</p><div></div><div></div><p>B</p>")
    assert text == "A\n\nB"
    assert "\n\n\n" not in text


def test_keeps_code_block_contents():
    text = html_to_text("<pre><code>temp = 100\nprint(temp)</code></pre>")
    assert "temp = 100" in text
    assert "print(temp)" in text


def test_empty_and_none_like_input():
    assert html_to_text("") == ""


def test_truncate_leaves_short_text_alone():
    assert truncate("short", 100) == "short"


def test_truncate_prefers_a_paragraph_boundary():
    text = "First paragraph.\n\nSecond paragraph is longer than the limit."
    assert truncate(text, 30) == "First paragraph. [...]"


def test_truncate_falls_back_to_a_sentence_boundary():
    text = "One sentence here. Another sentence that runs past the limit."
    result = truncate(text, 30)
    assert result.endswith("[...]")
    assert "Another" not in result


def test_truncate_never_exceeds_the_budget_by_much():
    long_text = "word " * 500
    result = truncate(long_text, 100)
    assert len(result) <= 100 + len(" [...]")
