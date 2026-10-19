"""Schema contract: strict models and the JSON schema sent to providers."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from tests.conftest import EXPECTED_INVOICE, EXPECTED_JOB, EXPECTED_TICKET

from structured_extract import SchemaNotFoundError, get_schema
from structured_extract.schemas import Invoice, JobPosting, strict_json_schema


def test_examples_match_models() -> None:
    assert Invoice.model_validate(EXPECTED_INVOICE).invoice_number == "INV-2026-1042"
    assert get_schema("job_posting").model_validate(EXPECTED_JOB).company == "Lumen Analytics"
    assert get_schema("support_ticket").model_validate(EXPECTED_TICKET).priority == "high"


def test_unknown_schema_lists_available() -> None:
    with pytest.raises(SchemaNotFoundError, match="invoice") as exc_info:
        get_schema("receipt")
    assert exc_info.value.available == ["invoice", "job_posting", "support_ticket"]


def test_strict_schema_closes_objects() -> None:
    schema = strict_json_schema(Invoice)
    assert schema["additionalProperties"] is False
    assert "line_items" in schema["required"]
    line_item = schema["$defs"]["LineItem"]
    assert line_item["additionalProperties"] is False
    assert set(line_item["required"]) == set(line_item["properties"])


def test_models_reject_bad_invariants() -> None:
    extra = {**EXPECTED_INVOICE, "note": "nope"}
    with pytest.raises(ValidationError):
        Invoice.model_validate(extra)

    bad_date = {**EXPECTED_INVOICE, "issued_on": "12/03/2026"}
    with pytest.raises(ValidationError, match="ISO date"):
        Invoice.model_validate(bad_date)

    swapped = {**EXPECTED_JOB, "salary_min": 10, "salary_max": 2}
    with pytest.raises(ValidationError, match="salary_max"):
        JobPosting.model_validate(swapped)

    quoted_qty = json_invoice_with_string_quantity()
    with pytest.raises(ValidationError, match="integer"):
        Invoice.model_validate(quoted_qty)


def json_invoice_with_string_quantity() -> dict[str, object]:
    payload = json_copy(EXPECTED_INVOICE)
    items = payload["line_items"]
    assert isinstance(items, list)
    first = dict(items[0])
    first["quantity"] = "10"
    items[0] = first
    return payload


def json_copy(payload: dict[str, object]) -> dict[str, object]:
    import json

    return json.loads(json.dumps(payload))
