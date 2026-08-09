"""LLM access layer.

The course code calls the OpenAI Responses API (`client.responses.create` and
`client.responses.parse`). Groq exposes an OpenAI-compatible endpoint but only
implements Chat Completions, so none of that transfers directly. This module is
the adapter: everything above it talks to `complete()` and `structured()`, and
swapping providers is a change of two environment variables.

Structured output is done by asking for a JSON object, then validating against a
Pydantic model and retrying on failure, rather than relying on native schema
enforcement. Support for `json_schema` response formats varies by model on Groq,
whereas `json_object` plus validation works everywhere and degrades gracefully.
"""

import json
import re
import time
from dataclasses import dataclass

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
from pydantic import BaseModel, ValidationError

from app import settings

MAX_RETRIES = 4

# Models like to wrap JSON in a markdown fence even when told not to.
_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class Usage:
    """Token counts for one or more calls.

    Frozen for two reasons: nothing should mutate a recorded token count, and a
    non-frozen dataclass has `__hash__ = None`, which makes it illegal as a
    field default. `EMPTY_USAGE` is used as exactly that in `Retrieval` and
    `RagResult`, so both of those modules would fail to import.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @property
    def cost(self) -> float:
        """Estimated USD. Zero on the free tier unless prices are configured."""
        return (
            self.prompt_tokens * settings.PRICE_INPUT_PER_MTOK
            + self.completion_tokens * settings.PRICE_OUTPUT_PER_MTOK
        ) / 1_000_000

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.total_tokens + other.total_tokens,
        )


EMPTY_USAGE = Usage(0, 0, 0)


@dataclass
class LLMResult:
    text: str
    model: str
    usage: Usage
    latency: float


def _usage_from_response(response) -> Usage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return EMPTY_USAGE
    prompt_tokens = usage.prompt_tokens or 0
    completion_tokens = usage.completion_tokens or 0
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=usage.total_tokens or prompt_tokens + completion_tokens,
    )


def _strip_fence(text: str) -> str:
    return _JSON_FENCE.sub("", text).strip()


class LLMClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 default_model: str | None = None):
        self.client = OpenAI(
            api_key=api_key or settings.require_api_key(),
            base_url=base_url or settings.LLM_BASE_URL,
        )
        self.default_model = default_model or settings.LLM_MODEL

    def _create(self, messages: list[dict], model: str, temperature: float,
                response_format: dict | None = None):
        """Chat completion with backoff on the errors that are worth retrying.

        The Groq free tier enforces per-minute token limits, so 429s are normal
        rather than exceptional and need to be handled rather than surfaced.
        """
        kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                return self.client.chat.completions.create(**kwargs)
            except RateLimitError as exc:
                last_error = exc
                wait = 5 * (attempt + 1)
                print(f"  rate limited, retrying in {wait}s")
                time.sleep(wait)
            except APIConnectionError as exc:
                last_error = exc
                time.sleep(2**attempt)
            except APIStatusError as exc:
                # A rejected response_format is worth one retry without it;
                # anything else is a real problem and should surface.
                if response_format is not None and exc.status_code == 400:
                    print("  model rejected response_format, retrying as plain text")
                    kwargs.pop("response_format", None)
                    response_format = None
                    last_error = exc
                    continue
                raise
        raise RuntimeError(
            f"{model} failed after {MAX_RETRIES} attempts"
        ) from last_error

    def complete(self, instructions: str, prompt: str, model: str | None = None,
                 temperature: float = 0.0) -> LLMResult:
        model = model or self.default_model
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt},
        ]

        started = time.monotonic()
        response = self._create(messages, model, temperature)
        latency = time.monotonic() - started

        return LLMResult(
            text=(response.choices[0].message.content or "").strip(),
            model=model,
            usage=_usage_from_response(response),
            latency=latency,
        )

    def structured(self, instructions: str, prompt: str,
                   output_model: type[BaseModel], model: str | None = None,
                   temperature: float = 0.0) -> tuple[BaseModel, Usage]:
        """Return a validated Pydantic object, retrying on malformed JSON.

        The schema goes into the system message because `json_object` mode only
        guarantees syntactically valid JSON, not the shape we asked for.
        """
        model = model or self.default_model
        schema = json.dumps(output_model.model_json_schema(), indent=2)
        system = (
            f"{instructions}\n\n"
            f"Reply with a single JSON object matching this schema. "
            f"No prose, no markdown fence.\n\n{schema}"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        total = EMPTY_USAGE
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            response = self._create(
                messages, model, temperature,
                response_format={"type": "json_object"},
            )
            total = total + _usage_from_response(response)
            raw = _strip_fence(response.choices[0].message.content or "")

            try:
                return output_model.model_validate_json(raw), total
            except (ValidationError, ValueError) as exc:
                last_error = exc
                print(f"  invalid JSON on attempt {attempt + 1}, asking again")
                # Show the model its own mistake instead of repeating the ask.
                messages = messages[:2] + [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            f"That did not match the schema: {exc}. "
                            f"Return only the corrected JSON object."
                        ),
                    },
                ]

        raise RuntimeError(
            f"{model} did not return valid {output_model.__name__} JSON"
        ) from last_error


_default_client: LLMClient | None = None


def get_client() -> LLMClient:
    """Process-wide client. The OpenAI SDK is thread-safe and pools connections."""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client
