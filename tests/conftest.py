"""Shared fixtures. Tests stay offline even if the shell has an API key set."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

EXPECTED_INVOICE = {
    "vendor": "Northwind Supply Co.",
    "invoice_number": "INV-2026-1042",
    "issued_on": "2026-03-12",
    "currency": "USD",
    "line_items": [
        {
            "description": "A4 paper reams",
            "quantity": 10,
            "unit_price": "4.50",
            "total": "45.00",
        },
        {
            "description": "Printer toner, black",
            "quantity": 2,
            "unit_price": "62.00",
            "total": "124.00",
        },
        {
            "description": "Sales tax",
            "quantity": 1,
            "unit_price": "13.52",
            "total": "13.52",
        },
    ],
    "total": "182.52",
    "payment_terms": "Net 30",
}

EXPECTED_JOB = {
    "title": "Senior Backend Engineer",
    "company": "Lumen Analytics",
    "location": "Remote (EU time zones)",
    "employment_type": "full-time",
    "salary_min": 85000,
    "salary_max": 110000,
    "salary_currency": "EUR",
    "requirements": [
        "4+ years of Python in production services",
        "PostgreSQL, including schema design and query plans",
        "A message queue or an append-only event log",
    ],
}

EXPECTED_TICKET = {
    "ticket_id": "SUP-4419",
    "customer_email": "maria.santos@client.example",
    "priority": "high",
    "category": "billing",
    "summary": (
        "I was charged twice for the March invoice INV-2026-1042. "
        "The duplicate amount is 182.52 USD and both charges posted on 2026-03-18."
    ),
    "requested_action": (
        "Refund the duplicate charge of 182.52 USD and email me when the refund is issued."
    ),
}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def _force_offline_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRUCTURED_EXTRACT_PROVIDER", "fake")


@pytest.fixture
def invoice_text() -> str:
    return (EXAMPLES / "invoice.txt").read_text(encoding="utf-8")


@pytest.fixture
def job_text() -> str:
    return (EXAMPLES / "job_posting.txt").read_text(encoding="utf-8")


@pytest.fixture
def ticket_text() -> str:
    return (EXAMPLES / "support_ticket.txt").read_text(encoding="utf-8")
