"""Settings from the environment. Nothing here talks to a network."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from structured_extract.exceptions import ConfigError

ProviderName = Literal["fake", "openai"]
ResponseFormat = Literal["json_schema", "json_object", "none"]

_PROVIDERS = ("fake", "openai")
_FORMATS = ("json_schema", "json_object", "none")


@dataclass(frozen=True)
class Settings:
    """Resolved runtime settings.

    ``provider`` is stored even when it is not one of the known names.
    ``build_provider`` rejects unknown names after the CLI has had a chance
    to override the environment.
    """

    provider: str
    timeout: float
    max_attempts: int
    base_url: str
    api_key: str | None
    model: str
    response_format: str


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    """Read settings. Pass ``environ`` in tests to avoid the process environment."""

    env = os.environ if environ is None else environ
    provider = env.get("STRUCTURED_EXTRACT_PROVIDER", "fake").strip().lower() or "fake"
    response_format = (
        env.get("STRUCTURED_EXTRACT_RESPONSE_FORMAT", "json_schema").strip().lower()
        or "json_schema"
    )
    if response_format not in _FORMATS:
        raise ConfigError(
            "STRUCTURED_EXTRACT_RESPONSE_FORMAT must be one of: " + ", ".join(_FORMATS)
        )
    timeout = _positive_float(
        env.get("STRUCTURED_EXTRACT_TIMEOUT", "30"),
        "STRUCTURED_EXTRACT_TIMEOUT",
    )
    max_attempts = _positive_int(
        env.get("STRUCTURED_EXTRACT_MAX_ATTEMPTS", "3"),
        "STRUCTURED_EXTRACT_MAX_ATTEMPTS",
    )
    return Settings(
        provider=provider,
        timeout=timeout,
        max_attempts=max_attempts,
        base_url=env.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
        or "https://api.openai.com/v1",
        api_key=_blank_to_none(env.get("OPENAI_API_KEY")),
        model=env.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini",
        response_format=response_format,
    )


def known_providers() -> tuple[str, ...]:
    return _PROVIDERS


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _positive_float(raw: str, name: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number of seconds, got {raw!r}") from exc
    if value <= 0:
        raise ConfigError(f"{name} must be greater than 0, got {raw!r}")
    return value


def _positive_int(raw: str, name: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ConfigError(f"{name} must be at least 1, got {raw!r}")
    return value
