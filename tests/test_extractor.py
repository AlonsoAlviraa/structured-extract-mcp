"""Extractor retry policy. Providers here never touch the network."""

from __future__ import annotations

import json
import logging

import pytest

from structured_extract import ExtractionFailedError, SchemaNotFoundError, extract
from structured_extract.exceptions import (
    ProviderError,
    ProviderTimeoutError,
    UnrecognizedInputError,
)
from structured_extract.providers import FakeProvider, ScriptedProvider
from structured_extract.schemas import Invoice, JobPosting, SupportTicket
from tests.conftest import EXPECTED_INVOICE, EXPECTED_JOB, EXPECTED_TICKET


def test_happy_path_returns_validated_model() -> None:
    provider = ScriptedProvider([json.dumps(EXPECTED_INVOICE)])
    result = extract("unused because the script replies", "invoice", provider, timeout=5)

    assert isinstance(result, Invoice)
    assert result.model_dump() == EXPECTED_INVOICE
    assert provider.calls[0]["schema_name"] == "invoice"
    assert provider.calls[0]["timeout"] == 5
    assert "line_items" in provider.calls[0]["messages"][0]["content"]
    assert provider.calls[0]["json_schema"]["additionalProperties"] is False


def test_markdown_fence_is_stripped_before_validation() -> None:
    fenced = "```json\n" + json.dumps(EXPECTED_INVOICE) + "\n```"
    provider = ScriptedProvider([fenced])

    result = extract("text", "invoice", provider, max_attempts=1)

    assert result.model_dump() == EXPECTED_INVOICE
    assert len(provider.calls) == 1


def test_invalid_json_then_retry_succeeds() -> None:
    bad = "this is not json"
    provider = ScriptedProvider([bad, json.dumps(EXPECTED_INVOICE)])

    result = extract("body", "invoice", provider, max_attempts=3, timeout=5)

    assert result.invoice_number == "INV-2026-1042"
    assert len(provider.calls) == 2
    second = provider.calls[1]["messages"]
    assert second[-2] == {"role": "assistant", "content": bad}
    feedback = second[-1]["content"]
    assert "Validation error:" in feedback
    assert "(root)" in feedback


def test_schema_error_is_fed_back_and_retry_succeeds() -> None:
    first = json.loads(json.dumps(EXPECTED_INVOICE))
    first["note"] = "please ignore"
    provider = ScriptedProvider([json.dumps(first), json.dumps(EXPECTED_INVOICE)])

    result = extract("body", "invoice", provider)

    assert result.model_dump() == EXPECTED_INVOICE
    feedback = provider.calls[1]["messages"][-1]["content"]
    assert "Extra inputs are not permitted" in feedback
    assert "note" in feedback


def test_strict_types_reject_coercion_then_retry() -> None:
    first = json.loads(json.dumps(EXPECTED_INVOICE))
    first["line_items"][0]["quantity"] = "10"
    provider = ScriptedProvider([json.dumps(first), json.dumps(EXPECTED_INVOICE)])

    result = extract("body", "invoice", provider)

    assert result.line_items[0].quantity == 10
    assert "valid integer" in provider.calls[1]["messages"][-1]["content"]


def test_money_format_error_is_retried() -> None:
    first = json.loads(json.dumps(EXPECTED_INVOICE))
    first["total"] = "182.5"
    provider = ScriptedProvider([json.dumps(first), json.dumps(EXPECTED_INVOICE)])

    result = extract("body", "invoice", provider)

    assert result.total == "182.52"
    assert "two digits" in provider.calls[1]["messages"][-1]["content"]


def test_retries_exhausted_raises() -> None:
    provider = ScriptedProvider(["nope", "still nope", "no"])

    with pytest.raises(ExtractionFailedError) as exc_info:
        extract("body", "invoice", provider, max_attempts=3)

    error = exc_info.value
    assert error.attempts == 3
    assert error.schema == "invoice"
    assert error.last_output == "no"
    assert error.last_error
    assert len(provider.calls) == 3


def test_timeout_is_not_retried() -> None:
    provider = ScriptedProvider([ProviderTimeoutError(1.5), json.dumps(EXPECTED_INVOICE)])

    with pytest.raises(ProviderTimeoutError) as exc_info:
        extract("body", "invoice", provider, max_attempts=3)

    assert exc_info.value.timeout == 1.5
    assert len(provider.calls) == 1
    assert provider.script  # the successful reply was never consumed


def test_provider_error_is_not_retried() -> None:
    provider = ScriptedProvider([ProviderError("HTTP 500: upstream"), json.dumps(EXPECTED_INVOICE)])

    with pytest.raises(ProviderError, match="HTTP 500"):
        extract("body", "invoice", provider, max_attempts=3)

    assert len(provider.calls) == 1


def test_unknown_schema() -> None:
    provider = ScriptedProvider(["{}"])

    with pytest.raises(SchemaNotFoundError, match="nope") as exc_info:
        extract("body", "nope", provider)

    assert "invoice" in exc_info.value.available
    assert provider.calls == []


def test_max_attempts_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        extract("body", "invoice", ScriptedProvider([]), max_attempts=0)


def test_info_logs_do_not_include_source_text(caplog: pytest.LogCaptureFixture) -> None:
    secret = "UNIQUE-SECRET-PHRASE"
    provider = ScriptedProvider([json.dumps(EXPECTED_INVOICE)])

    with caplog.at_level(logging.INFO, logger="structured_extract"):
        extract(f"please do not log {secret}", "invoice", provider)

    assert secret not in caplog.text
    assert "text_chars=" in caplog.text


def test_validation_failure_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    provider = ScriptedProvider(["nope", json.dumps(EXPECTED_INVOICE)])

    with caplog.at_level(logging.WARNING, logger="structured_extract"):
        extract("body", "invoice", provider)

    assert "validation failed" in caplog.text


def test_fake_provider_extracts_bundled_examples(
    invoice_text: str,
    job_text: str,
    ticket_text: str,
) -> None:
    provider = FakeProvider()

    invoice = extract(invoice_text, "invoice", provider)
    job = extract(job_text, "job_posting", provider)
    ticket = extract(ticket_text, "support_ticket", provider)

    assert isinstance(invoice, Invoice)
    assert isinstance(job, JobPosting)
    assert isinstance(ticket, SupportTicket)
    assert invoice.model_dump() == EXPECTED_INVOICE
    assert job.model_dump() == EXPECTED_JOB
    assert ticket.model_dump() == EXPECTED_TICKET


def test_fake_provider_does_not_invent_success() -> None:
    provider = FakeProvider()

    with pytest.raises(UnrecognizedInputError, match="missing label"):
        extract("Hello, can you look at this when you have a minute?", "invoice", provider)
