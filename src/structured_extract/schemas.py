"""Pydantic models for the bundled extraction schemas.

Money fields are decimal strings with two fractional digits. JSON numbers
are binary floats, which is a poor fit for invoice amounts. Salary figures
are whole currency units and stay integers.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from structured_extract.exceptions import SchemaNotFoundError

_MONEY = re.compile(r"\d+\.\d{2}")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_CURRENCY = re.compile(r"[A-Z]{3}")
_EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")

_STRICT = ConfigDict(extra="forbid", strict=True)


class LineItem(BaseModel):
    """One priced row on an invoice."""

    model_config = _STRICT

    description: str = Field(min_length=1, description="What was sold.")
    quantity: int = Field(gt=0, description="Count of units. Not a float.")
    unit_price: str = Field(description="Price for one unit, as a decimal string.")
    total: str = Field(description="quantity times unit price, as a decimal string.")

    @field_validator("unit_price", "total")
    @classmethod
    def money(cls, value: str) -> str:
        if not _MONEY.fullmatch(value):
            raise ValueError("must be a non-negative decimal with two digits after the point")
        return value


class Invoice(BaseModel):
    """A supplier invoice extracted from an email or similar text."""

    model_config = _STRICT

    vendor: str = Field(min_length=1)
    invoice_number: str = Field(min_length=1)
    issued_on: str = Field(description="Invoice date as YYYY-MM-DD.")
    currency: str = Field(description="ISO 4217 currency code.")
    line_items: list[LineItem] = Field(min_length=1)
    total: str = Field(description="Amount due, as a decimal string.")
    payment_terms: str = Field(min_length=1, description="For example 'Net 30'.")

    @field_validator("issued_on")
    @classmethod
    def iso_date(cls, value: str) -> str:
        if not _ISO_DATE.fullmatch(value):
            raise ValueError("must be an ISO date YYYY-MM-DD")
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("must be an ISO date YYYY-MM-DD") from exc
        return value

    @field_validator("currency")
    @classmethod
    def iso_currency(cls, value: str) -> str:
        if not _CURRENCY.fullmatch(value):
            raise ValueError("must be a three-letter ISO currency code")
        return value

    @field_validator("total")
    @classmethod
    def money(cls, value: str) -> str:
        if not _MONEY.fullmatch(value):
            raise ValueError("must be a non-negative decimal with two digits after the point")
        return value


class JobPosting(BaseModel):
    """A job ad reduced to the fields a recruiter would filter on."""

    model_config = _STRICT

    title: str = Field(min_length=1)
    company: str = Field(min_length=1)
    location: str = Field(min_length=1)
    employment_type: Literal["full-time", "part-time", "contract", "internship"]
    salary_min: int = Field(ge=0, description="Lower bound in whole currency units.")
    salary_max: int = Field(ge=0, description="Upper bound in whole currency units.")
    salary_currency: str
    requirements: list[str] = Field(min_length=1)

    @field_validator("salary_currency")
    @classmethod
    def iso_currency(cls, value: str) -> str:
        if not _CURRENCY.fullmatch(value):
            raise ValueError("must be a three-letter ISO currency code")
        return value

    @field_validator("requirements")
    @classmethod
    def non_empty_items(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("requirements must be non-empty strings")
        return value

    @model_validator(mode="after")
    def salary_range(self) -> JobPosting:
        if self.salary_max < self.salary_min:
            raise ValueError("salary_max must be greater than or equal to salary_min")
        return self


class SupportTicket(BaseModel):
    """A support email reduced to the fields a queue would store."""

    model_config = _STRICT

    ticket_id: str = Field(min_length=1)
    customer_email: str
    priority: Literal["low", "medium", "high", "urgent"]
    category: str = Field(min_length=1)
    summary: str = Field(min_length=1, description="What happened, in the customer's words.")
    requested_action: str = Field(min_length=1, description="What the customer asked us to do.")

    @field_validator("customer_email")
    @classmethod
    def email(cls, value: str) -> str:
        if not _EMAIL.fullmatch(value):
            raise ValueError("must be an email address")
        return value


SCHEMAS: dict[str, type[BaseModel]] = {
    "invoice": Invoice,
    "job_posting": JobPosting,
    "support_ticket": SupportTicket,
}


def available_schemas() -> list[str]:
    """Schema names in a stable order."""

    return sorted(SCHEMAS)


def get_schema(name: str) -> type[BaseModel]:
    try:
        return SCHEMAS[name]
    except KeyError:
        raise SchemaNotFoundError(name, available_schemas()) from None


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON schema for providers that implement OpenAI strict structured output.

    Every object gets ``additionalProperties: false`` and lists all of its
    properties as required. That matches the strict mode used by OpenAI and
    several compatible servers. It is not a full translation of every JSON
    Schema feature those servers reject.
    """

    schema = model.model_json_schema()
    _tighten(schema)
    return schema


def _tighten(node: object) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object" and isinstance(node.get("properties"), dict):
            node["additionalProperties"] = False
            node["required"] = list(node["properties"])
        for value in node.values():
            _tighten(value)
    elif isinstance(node, list):
        for item in node:
            _tighten(item)
