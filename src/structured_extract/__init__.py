"""Extract typed records from unstructured text."""

from structured_extract.exceptions import (
    ConfigError,
    ExtractionFailedError,
    ProviderError,
    ProviderTimeoutError,
    SchemaNotFoundError,
    StructuredExtractError,
    UnrecognizedInputError,
)
from structured_extract.extractor import extract
from structured_extract.schemas import available_schemas, get_schema

__version__ = "0.1.0"

__all__ = [
    "ConfigError",
    "ExtractionFailedError",
    "ProviderError",
    "ProviderTimeoutError",
    "SchemaNotFoundError",
    "StructuredExtractError",
    "UnrecognizedInputError",
    "available_schemas",
    "extract",
    "get_schema",
]
