"""Providers return raw text. They do not validate it and they do not guess.

The extractor validates. If a provider cannot produce text, it raises a
``ProviderError`` (or a subclass). It does not return a placeholder object.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol

import httpx

from structured_extract.config import Settings
from structured_extract.exceptions import (
    ConfigError,
    ProviderError,
    ProviderTimeoutError,
    UnrecognizedInputError,
)

logger = logging.getLogger(__name__)

_ITEM = re.compile(r"^(\d+) x (.+) @ (\d+\.\d{2}) = (\d+\.\d{2})$")


class Provider(Protocol):
    """One completion call. ``messages`` already include any validation feedback."""

    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        schema_name: str,
        json_schema: dict[str, Any],
        timeout: float,
    ) -> str:
        """Return the model message content. The caller checks that it is JSON."""


class ScriptedProvider:
    """Queue of canned replies or exceptions. Used by tests and local experiments.

    Each ``complete`` call consumes one item. Strings are returned as-is.
    Exceptions are raised. An empty queue is a ``ProviderError``, not a success.
    """

    def __init__(self, script: list[str | Exception]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        schema_name: str,
        json_schema: dict[str, Any],
        timeout: float,
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "schema_name": schema_name,
                "json_schema": json_schema,
                "timeout": timeout,
            }
        )
        if not self.script:
            raise ProviderError("scripted provider has no responses left")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeProvider:
    """Offline parser for the labeled texts in ``examples/``.

    It is deterministic and it does not invent fields. Missing labels raise
    ``UnrecognizedInputError``. It is not a language model, and it is not used
    as a fallback when another provider fails.
    """

    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        schema_name: str,
        json_schema: dict[str, Any],
        timeout: float,
    ) -> str:
        del json_schema, timeout
        text = _first_user_text(messages)
        parsers = {
            "invoice": _parse_invoice,
            "job_posting": _parse_job_posting,
            "support_ticket": _parse_support_ticket,
        }
        try:
            parser = parsers[schema_name]
        except KeyError:
            raise UnrecognizedInputError(
                f"fake provider has no parser for schema {schema_name!r}"
            ) from None
        return json.dumps(parser(text))


class OpenAICompatibleProvider:
    """POST ``{base_url}/chat/completions`` and return the assistant text.

    Works with any server that speaks this shape: OpenAI, NVIDIA NIM, Ollama's
    OpenAI-compatible endpoint, and similar. The API key is sent as a bearer
    token and is removed from error messages if the server echoes it.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        response_format: str = "json_schema",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.response_format = response_format
        self._transport = transport

    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        schema_name: str,
        json_schema: dict[str, Any],
        timeout: float,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
        }
        response_format = _response_format(self.response_format, schema_name, json_schema)
        if response_format is not None:
            body["response_format"] = response_format
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "structured-extract/0.1.0",
        }
        try:
            with httpx.Client(transport=self._transport) as client:
                response = client.post(url, json=body, headers=headers, timeout=timeout)
        except httpx.TimeoutException as exc:
            logger.warning("provider timed out after %ss", timeout)
            raise ProviderTimeoutError(timeout) from exc
        except httpx.HTTPError as exc:
            logger.warning("provider request failed: %s", exc.__class__.__name__)
            raise ProviderError(f"request failed: {exc.__class__.__name__}") from exc

        if response.status_code >= 400:
            detail = _error_detail(response, self.api_key)
            logger.warning("provider HTTP %s", response.status_code)
            raise ProviderError(f"provider returned HTTP {response.status_code}: {detail}")
        return _message_text(response)


def build_provider(
    settings: Settings,
    *,
    transport: httpx.BaseTransport | None = None,
) -> Provider:
    """Construct the provider named by ``settings``.

    Unknown names and a missing API key for ``openai`` raise ``ConfigError``.
    The fake provider is not substituted in those cases.
    """

    if settings.provider == "fake":
        return FakeProvider()
    if settings.provider == "openai":
        if not settings.api_key:
            raise ConfigError(
                "OPENAI_API_KEY is required when STRUCTURED_EXTRACT_PROVIDER is openai"
            )
        return OpenAICompatibleProvider(
            base_url=settings.base_url,
            api_key=settings.api_key,
            model=settings.model,
            response_format=settings.response_format,
            transport=transport,
        )
    raise ConfigError(f"Unknown provider {settings.provider!r}. Expected one of: fake, openai")


def _response_format(
    kind: str,
    schema_name: str,
    json_schema: dict[str, Any],
) -> dict[str, Any] | None:
    if kind == "none":
        return None
    if kind == "json_object":
        return {"type": "json_object"}
    if kind == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": json_schema,
            },
        }
    raise ConfigError(
        "STRUCTURED_EXTRACT_RESPONSE_FORMAT must be one of: json_schema, json_object, none"
    )


def _message_text(response: httpx.Response) -> str:
    try:
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise ProviderError("provider response did not contain choices[0].message.content") from exc
    text = _content_to_text(content)
    if text is None:
        raise ProviderError("provider response content was empty or not text")
    return text


def _content_to_text(content: object) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "".join(parts)
    return None


def _error_detail(response: httpx.Response, api_key: str) -> str:
    detail = response.text[:500]
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            detail = error["message"][:500]
    if api_key:
        detail = detail.replace(api_key, "***")
    return detail


def _first_user_text(messages: list[dict[str, str]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return message.get("content", "")
    raise UnrecognizedInputError("fake provider received no user message")


def _parse_invoice(text: str) -> dict[str, Any]:
    return {
        "vendor": _label(text, "Vendor"),
        "invoice_number": _label(text, "Invoice number"),
        "issued_on": _label(text, "Issued on"),
        "currency": _label(text, "Currency"),
        "payment_terms": _label(text, "Payment terms"),
        "line_items": _line_items(text),
        "total": _label(text, "Total"),
    }


def _parse_job_posting(text: str) -> dict[str, Any]:
    try:
        salary_min = int(_label(text, "Salary min"))
        salary_max = int(_label(text, "Salary max"))
    except ValueError as exc:
        raise UnrecognizedInputError("fake provider: salary bounds must be integers") from exc
    return {
        "title": _label(text, "Title"),
        "company": _label(text, "Company"),
        "location": _label(text, "Location"),
        "employment_type": _label(text, "Employment type"),
        "salary_min": salary_min,
        "salary_max": salary_max,
        "salary_currency": _label(text, "Salary currency"),
        "requirements": _bullets_after(text, "Requirements"),
    }


def _parse_support_ticket(text: str) -> dict[str, Any]:
    return {
        "ticket_id": _label(text, "Ticket"),
        "customer_email": _from_address(text),
        "priority": _label(text, "Priority").lower(),
        "category": _label(text, "Category"),
        "summary": _block(text, "Summary", stop={"Requested action"}),
        "requested_action": _block(text, "Requested action", stop=set()),
    }


def _label(text: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}:[ \t]*(.+?)\s*$", text, re.MULTILINE)
    if match is None:
        raise UnrecognizedInputError(f"fake provider: missing label {name!r}")
    return match.group(1).strip()


def _from_address(text: str) -> str:
    match = re.search(r"^From:[ \t]*(\S+)", text, re.MULTILINE)
    if match is None:
        raise UnrecognizedInputError("fake provider: missing 'From' address")
    return match.group(1)


def _line_items(text: str) -> list[dict[str, Any]]:
    match = re.search(r"^Line items:\s*$", text, re.MULTILINE)
    if match is None:
        raise UnrecognizedInputError("fake provider: missing label 'Line items'")
    items: list[dict[str, Any]] = []
    for line in text[match.end() :].splitlines():
        stripped = line.strip()
        if not stripped:
            if items:
                break
            continue
        if stripped.startswith("Total:"):
            break
        if stripped.startswith("- "):
            stripped = stripped[2:]
        item_match = _ITEM.match(stripped)
        if item_match is None:
            preview = line.strip()
            raise UnrecognizedInputError(f"fake provider: could not parse line item {preview!r}")
        quantity, description, unit_price, total = item_match.groups()
        items.append(
            {
                "description": description,
                "quantity": int(quantity),
                "unit_price": unit_price,
                "total": total,
            }
        )
    if not items:
        raise UnrecognizedInputError("fake provider: no line items")
    return items


def _bullets_after(text: str, name: str) -> list[str]:
    match = re.search(rf"^{re.escape(name)}:\s*$", text, re.MULTILINE)
    if match is None:
        raise UnrecognizedInputError(f"fake provider: missing label {name!r}")
    items: list[str] = []
    for line in text[match.end() :].splitlines():
        stripped = line.strip()
        if not stripped:
            if items:
                break
            continue
        if not stripped.startswith("- "):
            if items:
                break
            continue
        items.append(stripped[2:].strip())
    if not items:
        raise UnrecognizedInputError(f"fake provider: no bullets under {name!r}")
    return items


def _block(text: str, name: str, *, stop: set[str]) -> str:
    lines = text.splitlines()
    start: int | None = None
    for index, line in enumerate(lines):
        if line.strip() == f"{name}:":
            start = index + 1
            break
    if start is None:
        raise UnrecognizedInputError(f"fake provider: missing label {name!r}")
    collected: list[str] = []
    for line in lines[start:]:
        label = _label_name(line)
        if label is not None and (not stop or label in stop):
            break
        collected.append(line)
    value = "\n".join(collected).strip()
    if not value:
        raise UnrecognizedInputError(f"fake provider: empty {name!r}")
    return value


def _label_name(line: str) -> str | None:
    match = re.match(r"^([A-Za-z][A-Za-z ]*):[ \t]*$", line.strip())
    if match is None:
        return None
    return match.group(1)
