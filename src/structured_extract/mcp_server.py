"""Minimal MCP server. Transport is stdio. Do not write logs to stdout.

Register it with the official ``mcp`` SDK's ``MCPServer`` (the 2.x name for
what older docs call FastMCP). Tools call the same extractor as the CLI.
"""

from __future__ import annotations

import logging
import os

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from structured_extract import __version__
from structured_extract.config import load_settings
from structured_extract.exceptions import StructuredExtractError
from structured_extract.extractor import extract as run_extract
from structured_extract.providers import build_provider
from structured_extract.schemas import available_schemas

logger = logging.getLogger(__name__)

server = MCPServer(
    name="structured-extract",
    version=__version__,
    log_level="ERROR",
    instructions=(
        "Extracts one validated JSON object from unstructured text. "
        "Call list_schemas, then extract with one of those names. "
        "The default provider is offline and only parses the labeled example texts. "
        "Provider failures and validation failures are tool errors, not empty objects."
    ),
)


@server.tool(structured_output=False)
def list_schemas() -> list[str]:
    """Return the names of the extraction schemas this server can fill."""

    return available_schemas()


@server.tool(structured_output=False)
def extract(schema: str, text: str) -> str:
    """Extract validated JSON for ``schema`` from ``text``.

    Uses STRUCTURED_EXTRACT_PROVIDER (default: fake). The fake provider only
    understands the labeled examples in this repository. Failures raise a
    tool error instead of returning placeholder JSON.
    """

    try:
        settings = load_settings()
        provider = build_provider(settings)
        model = run_extract(
            text,
            schema,
            provider,
            max_attempts=settings.max_attempts,
            timeout=settings.timeout,
        )
    except StructuredExtractError as exc:
        raise ToolError(str(exc)) from exc
    return model.model_dump_json(indent=2)


def main() -> None:
    """Serve over stdio until the client closes the pipe."""

    level_name = os.environ.get("STRUCTURED_EXTRACT_LOG_LEVEL", "WARNING").upper()
    level = logging.getLevelNamesMapping().get(level_name, logging.WARNING)
    logging.getLogger("structured_extract").setLevel(level)
    logger.info("starting structured-extract MCP server on stdio")
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
