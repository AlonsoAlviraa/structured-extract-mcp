"""Typed errors. Callers should not have to parse log lines to know what failed."""


class StructuredExtractError(Exception):
    """Base type for errors this package raises on purpose."""


class ConfigError(StructuredExtractError):
    """Environment or CLI settings are missing or invalid."""


class SchemaNotFoundError(StructuredExtractError):
    """The requested schema name is not in the registry."""

    def __init__(self, name: str, available: list[str]) -> None:
        self.name = name
        self.available = available
        names = ", ".join(available)
        super().__init__(f"Unknown schema {name!r}. Available: {names}")


class ProviderError(StructuredExtractError):
    """The provider failed before returning text that can be validated."""


class ProviderTimeoutError(ProviderError):
    """The provider did not respond within the request timeout.

    Timeouts are not retried. A retry would spend the same budget again
    with no new information to send the model.
    """

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout
        super().__init__(f"provider timed out after {timeout:g} seconds")


class UnrecognizedInputError(ProviderError):
    """The offline fake provider could not parse this text.

    This is a failure, not an empty extraction. The fake provider only
    understands the labeled layouts in ``examples/``.
    """


class ExtractionFailedError(StructuredExtractError):
    """Every attempt returned text that failed schema validation."""

    def __init__(self, *, schema: str, attempts: int, last_error: str, last_output: str) -> None:
        self.schema = schema
        self.attempts = attempts
        self.last_error = last_error
        self.last_output = last_output
        super().__init__(
            f"extraction failed for schema {schema!r} after {attempts} attempt(s): {last_error}"
        )
