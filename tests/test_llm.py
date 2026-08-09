"""Tests for the LLM adapter's pure helpers.

Nothing here makes a network call. The parts worth testing offline are token
accounting and the fence-stripping that stands between a model's markdown habit
and a JSON parse error.
"""

from app import llm as llm_module
from app.llm import EMPTY_USAGE, Usage, _strip_fence


def test_strips_a_json_fence():
    assert _strip_fence('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_strips_a_bare_fence():
    assert _strip_fence('```\n{"a": 1}\n```') == '{"a": 1}'


def test_leaves_unfenced_json_alone():
    assert _strip_fence('{"a": 1}') == '{"a": 1}'


def test_strips_surrounding_whitespace():
    assert _strip_fence('  \n {"a": 1} \n ') == '{"a": 1}'


def test_usage_adds_componentwise():
    total = Usage(10, 5, 15) + Usage(2, 3, 5)
    assert (total.prompt_tokens, total.completion_tokens, total.total_tokens) == (
        12, 8, 20,
    )


def test_empty_usage_is_an_additive_identity():
    usage = Usage(7, 3, 10)
    combined = usage + EMPTY_USAGE
    assert combined.total_tokens == usage.total_tokens


def test_cost_is_zero_when_no_prices_are_configured(monkeypatch):
    monkeypatch.setattr(llm_module.settings, "PRICE_INPUT_PER_MTOK", 0.0)
    monkeypatch.setattr(llm_module.settings, "PRICE_OUTPUT_PER_MTOK", 0.0)
    assert Usage(1_000_000, 1_000_000, 2_000_000).cost == 0.0


def test_cost_uses_separate_input_and_output_rates(monkeypatch):
    monkeypatch.setattr(llm_module.settings, "PRICE_INPUT_PER_MTOK", 1.0)
    monkeypatch.setattr(llm_module.settings, "PRICE_OUTPUT_PER_MTOK", 3.0)

    # 1M input at $1 plus 1M output at $3.
    assert Usage(1_000_000, 1_000_000, 2_000_000).cost == 4.0
