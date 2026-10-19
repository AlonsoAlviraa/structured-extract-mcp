"""Validate model output and, on schema errors only, ask again with the error."""

from __future__ import annotations

import json
import logging
import re

from pydantic import BaseModel, ValidationError

from structured_extract.exceptions import ExtractionFailedError
from structured_extract.providers import Provider
from structured_extract.schemas import get_schema, strict_json_schema

logger = logging.getLogger(__name__)

# Models often wrap JSON in a fence even when told not to. Stripping one
# fence is normalization before validation, not a successful fallback:
# the inner text still has to validate.
_FENCE = re.compile(r"^```(?:json)?[ \t]*\n?(.*?)\n?```$", re.DOTALL)

_MAX_ERRORS = 20


def extract(
    text: str,
    schema: str,
    provider: Provider,
    *,
    max_attempts: int = 3,
    timeout: float = 30.0,
) -> BaseModel:
    """Return a validated model for ``schema``.

    ``provider.complete`` is called up to ``max_attempts`` times. A response
    that is not valid JSON, or that does not match the Pydantic model, is
    sent back to the provider with the validation error. Timeouts and other
    provider failures propagate immediately.

    Raises:
        SchemaNotFoundError: ``schema`` is not registered.
        ProviderError: the provider failed, including timeouts.
        ExtractionFailedError: every attempt failed validation.
        ValueError: ``max_attempts`` or ``timeout`` is out of range.
    """

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    model = get_schema(schema)
    json_schema = strict_json_schema(model)
    history: list[tuple[str, str]] = []
    last_error = ""
    last_raw = ""

    for attempt in range(1, max_attempts + 1):
        messages = _messages(schema, json_schema, text, history)
        logger.info(
            "extract start schema=%s attempt=%s/%s text_chars=%s",
            schema,
            attempt,
            max_attempts,
            len(text),
        )
        raw = provider.complete(
            messages=messages,
            schema_name=schema,
            json_schema=json_schema,
            timeout=timeout,
        )
        cleaned = _strip_fence(raw)
        if cleaned != raw.strip():
            logger.debug("removed a markdown fence before validation")
        try:
            value = model.model_validate_json(cleaned)
        except ValidationError as exc:
            last_error = format_validation_error(exc)
            last_raw = raw
            history.append((raw, last_error))
            logger.warning(
                "validation failed schema=%s attempt=%s/%s error=%s",
                schema,
                attempt,
                max_attempts,
                last_error.replace("\n", " | "),
            )
            continue
        logger.info("extract ok schema=%s attempts=%s", schema, attempt)
        return value

    raise ExtractionFailedError(
        schema=schema,
        attempts=max_attempts,
        last_error=last_error,
        last_output=last_raw,
    )


def format_validation_error(exc: ValidationError) -> str:
    """Short, line-oriented error text to show the model on the next attempt."""

    lines: list[str] = []
    for err in exc.errors()[:_MAX_ERRORS]:
        loc = ".".join(str(part) for part in err.get("loc", ())) or "(root)"
        lines.append(f"{loc}: {err.get('msg', 'invalid')}")
    if not lines:
        return str(exc)
    hidden = len(exc.errors()) - len(lines)
    if hidden > 0:
        lines.append(f"... {hidden} more error(s)")
    return "\n".join(lines)


def _strip_fence(raw: str) -> str:
    text = raw.strip()
    match = _FENCE.fullmatch(text)
    if match is None:
        return text
    return match.group(1).strip()


def _messages(
    schema: str,
    json_schema: dict[str, object],
    text: str,
    history: list[tuple[str, str]],
) -> list[dict[str, str]]:
    system = (
        "You extract one JSON object from the user text. "
        "The object must validate against the JSON schema. "
        "Reply with JSON only: no markdown fence and no commentary.\n\n"
        f"Schema name: {schema}\n"
        "JSON schema:\n"
        f"{json.dumps(json_schema)}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Text:\n{text}"},
    ]
    for raw, error in history:
        messages.append({"role": "assistant", "content": raw})
        messages.append(
            {
                "role": "user",
                "content": (
                    "That response failed validation. Reply with corrected JSON only.\n"
                    f"Validation error:\n{error}"
                ),
            }
        )
    return messages
