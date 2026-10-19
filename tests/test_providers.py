"""OpenAI-compatible client and settings. HTTP is mocked; no sockets."""

from __future__ import annotations

import json

import httpx
import pytest

from structured_extract.config import load_settings
from structured_extract.exceptions import ConfigError, ProviderError, ProviderTimeoutError
from structured_extract.providers import FakeProvider, OpenAICompatibleProvider, build_provider
from structured_extract.schemas import Invoice, strict_json_schema


def _provider(handler: httpx.Request, **kwargs: object) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        base_url="https://example.test/v1/",
        api_key="sk-test-secret",
        model="demo-model",
        transport=httpx.MockTransport(handler),
        **kwargs,  # type: ignore[arg-type]
    )


def _complete(provider: OpenAICompatibleProvider) -> str:
    return provider.complete(
        messages=[{"role": "user", "content": "Text:\nhi"}],
        schema_name="invoice",
        json_schema=strict_json_schema(Invoice),
        timeout=5,
    )


def test_json_schema_request_shape() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": '{"ok": true}'}}]},
        )

    text = _complete(_provider(handler))

    assert text == '{"ok": true}'
    assert seen["url"] == "https://example.test/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test-secret"
    body = seen["body"]
    assert isinstance(body, dict)
    assert body["model"] == "demo-model"
    assert body["temperature"] == 0
    response_format = body["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "invoice"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"]["additionalProperties"] is False


def test_json_object_and_none_formats() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content.decode()))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "{}"}}]},
        )

    _complete(_provider(handler, response_format="json_object"))
    _complete(_provider(handler, response_format="none"))

    assert bodies[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in bodies[1]


def test_content_parts_are_joined() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": '{"a": '},
                                {"type": "text", "text": "1}"},
                            ]
                        }
                    }
                ]
            },
        )

    assert _complete(_provider(handler)) == '{"a": 1}'


def test_timeout_maps_to_typed_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(ProviderTimeoutError, match="5"):
        _complete(_provider(handler))


def test_http_error_includes_status_and_redacts_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            401,
            json={"error": {"message": "bad key sk-test-secret"}},
        )

    with pytest.raises(ProviderError, match="HTTP 401") as exc_info:
        _complete(_provider(handler))

    assert "sk-test-secret" not in str(exc_info.value)
    assert "***" in str(exc_info.value)


def test_unexpected_payload_is_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={"choices": []})

    with pytest.raises(ProviderError, match="message.content"):
        _complete(_provider(handler))


def test_settings_defaults_and_openai_key() -> None:
    settings = load_settings({})
    assert settings.provider == "fake"
    assert settings.timeout == 30
    assert settings.max_attempts == 3
    assert settings.model == "gpt-4o-mini"
    assert settings.response_format == "json_schema"

    assert isinstance(build_provider(settings), FakeProvider)

    missing = load_settings({"STRUCTURED_EXTRACT_PROVIDER": "openai"})
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        build_provider(missing)

    configured = load_settings(
        {
            "STRUCTURED_EXTRACT_PROVIDER": "openai",
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_BASE_URL": "http://127.0.0.1:11434/v1",
            "OPENAI_MODEL": "local-model",
        }
    )
    transport = httpx.MockTransport(lambda request: httpx.Response(200))
    provider = build_provider(configured, transport=transport)
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.base_url == "http://127.0.0.1:11434/v1"
    assert provider.model == "local-model"


def test_invalid_settings() -> None:
    with pytest.raises(ConfigError, match="integer"):
        load_settings({"STRUCTURED_EXTRACT_MAX_ATTEMPTS": "many"})
    with pytest.raises(ConfigError, match="at least 1"):
        load_settings({"STRUCTURED_EXTRACT_MAX_ATTEMPTS": "0"})
    with pytest.raises(ConfigError, match="greater than 0"):
        load_settings({"STRUCTURED_EXTRACT_TIMEOUT": "0"})
    with pytest.raises(ConfigError, match="RESPONSE_FORMAT"):
        load_settings({"STRUCTURED_EXTRACT_RESPONSE_FORMAT": "yaml"})
    with pytest.raises(ConfigError, match="Unknown provider"):
        build_provider(load_settings({"STRUCTURED_EXTRACT_PROVIDER": "anthropic"}))
