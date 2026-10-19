"""Command-line entry point. Logs go to stderr. JSON goes to stdout."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from structured_extract.config import known_providers, load_settings
from structured_extract.exceptions import (
    ConfigError,
    SchemaNotFoundError,
    StructuredExtractError,
)
from structured_extract.extractor import extract
from structured_extract.providers import build_provider
from structured_extract.schemas import available_schemas


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code and does not call ``sys.exit``."""

    parser = argparse.ArgumentParser(
        prog="extract",
        description="Extract a validated JSON object from unstructured text.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="UTF-8 text file. Omit or pass '-' to read stdin.",
    )
    parser.add_argument("--schema", help="Schema name. See --list-schemas.")
    parser.add_argument(
        "--provider",
        choices=known_providers(),
        help="Override STRUCTURED_EXTRACT_PROVIDER. Default: fake.",
    )
    parser.add_argument("--timeout", type=float, help="Seconds per provider request.")
    parser.add_argument(
        "--max-attempts",
        type=int,
        help="Validation attempts, including the first. Default: 3.",
    )
    parser.add_argument("--list-schemas", action="store_true", help="Print schema names and exit.")
    parser.add_argument("--verbose", action="store_true", help="Log debug details to stderr.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if args.list_schemas:
        for name in available_schemas():
            print(name)
        return 0

    if not args.schema:
        parser.error("--schema is required unless --list-schemas is set")

    if args.max_attempts is not None and args.max_attempts < 1:
        print("--max-attempts must be at least 1", file=sys.stderr)
        return 2
    if args.timeout is not None and args.timeout <= 0:
        print("--timeout must be greater than 0", file=sys.stderr)
        return 2

    try:
        text = _read_text(args.path, parser)
        settings = load_settings()
        if args.provider is not None:
            settings = replace(settings, provider=args.provider)
        if args.timeout is not None:
            settings = replace(settings, timeout=args.timeout)
        if args.max_attempts is not None:
            settings = replace(settings, max_attempts=args.max_attempts)
        provider = build_provider(settings)
        result = extract(
            text,
            args.schema,
            provider,
            max_attempts=settings.max_attempts,
            timeout=settings.timeout,
        )
    except (SchemaNotFoundError, ConfigError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2
    except StructuredExtractError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(result.model_dump_json(indent=2))
    return 0


def _read_text(path: str | None, parser: argparse.ArgumentParser) -> str:
    if path is None or path == "-":
        if path is None and sys.stdin.isatty():
            parser.error("provide a text file or pipe text on stdin")
        return sys.stdin.read()
    file = Path(path)
    if not file.is_file():
        raise ConfigError(f"file not found: {path}")
    return file.read_text(encoding="utf-8")
