"""MCP tool functions, an in-process client, and the stdio entry point."""

from __future__ import annotations

import json
import sys

import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.server.mcpserver.exceptions import ToolError

from structured_extract.mcp_server import extract, list_schemas, server
from tests.conftest import EXPECTED_INVOICE


def test_list_schemas_tool() -> None:
    assert list_schemas() == ["invoice", "job_posting", "support_ticket"]


def test_extract_tool_returns_validated_json(invoice_text: str) -> None:
    raw = extract(schema="invoice", text=invoice_text)
    assert json.loads(raw) == EXPECTED_INVOICE


def test_extract_tool_reports_failure() -> None:
    with pytest.raises(ToolError, match="missing label"):
        extract(schema="invoice", text="hello")


def test_tools_are_registered() -> None:
    names = {tool.name for tool in server._tool_manager.list_tools()}
    assert names == {"list_schemas", "extract"}


@pytest.mark.anyio
async def test_in_process_client_calls_tools(invoice_text: str) -> None:
    async with Client(server) as client:
        listed = await client.list_tools()
        names = {tool.name for tool in listed.tools}
        assert names == {"list_schemas", "extract"}

        extracted = await client.call_tool(
            "extract",
            {"schema": "invoice", "text": invoice_text},
        )

    assert extracted.is_error is False
    text = _text(extracted)
    assert json.loads(text) == EXPECTED_INVOICE


@pytest.mark.anyio
async def test_in_process_client_surfaces_tool_error() -> None:
    async with Client(server) as client:
        result = await client.call_tool("extract", {"schema": "missing", "text": "hello"})

    assert result.is_error is True
    assert "missing" in _text(result)


@pytest.mark.anyio
async def test_stdio_entrypoint_lists_and_extracts(invoice_text: str) -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "structured_extract.mcp_server"],
        env={"STRUCTURED_EXTRACT_PROVIDER": "fake"},
    )
    async with Client(params) as client:
        listed = await client.list_tools()
        names = {tool.name for tool in listed.tools}
        extracted = await client.call_tool(
            "extract",
            {"schema": "invoice", "text": invoice_text},
        )
        rejected = await client.call_tool(
            "extract",
            {"schema": "support_ticket", "text": invoice_text},
        )
        schemas = await client.call_tool("list_schemas", {})

    assert names == {"list_schemas", "extract"}
    assert extracted.is_error is False
    assert json.loads(_text(extracted))["invoice_number"] == "INV-2026-1042"
    # The invoice text is not a support ticket. The server must report failure.
    assert rejected.is_error is True
    assert "missing label" in _text(rejected)
    assert "invoice" in _text(schemas)
    assert "job_posting" in _text(schemas)


def _text(result: object) -> str:
    content = getattr(result, "content", [])
    return "\n".join(getattr(block, "text", "") for block in content)
