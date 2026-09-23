"""
Thin wrapper around the Groq API (OpenAI-compatible chat completions).

Kept separate from the analysis logic so the model name, retry behaviour,
and JSON-parsing quirks live in exactly one place.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from groq import Groq, NotFoundError

# llama-3.3-70b-versatile: good balance of quality/cost/speed for grounded
# extraction + comparison tasks like this one. See README "Model choice".
DEFAULT_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODELS = ["openai/gpt-oss-120b", "llama-3.1-8b-instant"]


class LLMError(RuntimeError):
    pass


def get_client(api_key: Optional[str] = None) -> Groq:
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise LLMError(
            "No Groq API key found. Set GROQ_API_KEY as an environment "
            "variable, or paste it into the sidebar."
        )
    return Groq(api_key=key)


def _extract_text(completion) -> str:
    return completion.choices[0].message.content or ""


def call_claude(
    client: Groq,
    system: str,
    user: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> str:
    """Single-turn call, returns raw text. (Name kept as call_claude for
    drop-in compatibility with the rest of the pipeline/app code.)"""
    try:
        completion = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
    except NotFoundError:
        # model string not available on this account -- try fallbacks
        for fb in FALLBACK_MODELS:
            try:
                completion = client.chat.completions.create(
                    model=fb,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                break
            except NotFoundError:
                continue
        else:
            raise
    return _extract_text(completion)


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def call_claude_json(
    client: Groq,
    system: str,
    user: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> Any:
    """
    Calls the model with an instruction to return ONLY JSON, then parses it.
    Uses Groq's native JSON mode (response_format) as the primary guardrail,
    strips markdown code fences defensively, and retries once with a
    stricter reminder if parsing still fails.

    Note: Groq's response_format={"type": "json_object"} guarantees
    syntactically valid JSON but does not enforce our specific schema, and
    it requires the word "json" to appear somewhere in the prompt -- both
    handled below.
    """
    strict_system = (
        system
        + "\n\nYou must respond with ONLY valid JSON. No prose before or after. "
        "No markdown code fences."
    )
    if "json" not in strict_system.lower():
        strict_system += " Respond in JSON."

    try:
        completion = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": strict_system},
                {"role": "user", "content": user},
            ],
        )
        raw = _extract_text(completion)
    except Exception:
        # some models/endpoints may not support response_format -- fall back
        # to plain prompting + fence-stripping below
        raw = call_claude(client, strict_system, user, model=model, max_tokens=max_tokens, temperature=temperature)

    cleaned = _FENCE_RE.sub("", raw).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # one retry, being extra explicit about the failure
        retry_user = (
            user
            + "\n\nYour previous response could not be parsed as JSON. "
            "Respond again with ONLY a single valid JSON object/array, nothing else."
        )
        raw2 = call_claude(client, strict_system, retry_user, model=model, max_tokens=max_tokens, temperature=temperature)
        cleaned2 = _FENCE_RE.sub("", raw2).strip()
        parsed = json.loads(cleaned2)  # let this raise if it still fails -- caller should surface the error

    # Our schema is often "a JSON array" but json_object mode always returns
    # an object -- unwrap a single-key wrapper like {"results": [...]} if present.
    if isinstance(parsed, dict) and len(parsed) == 1:
        only_value = next(iter(parsed.values()))
        if isinstance(only_value, list):
            return only_value
    return parsed
