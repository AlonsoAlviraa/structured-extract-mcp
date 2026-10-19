"""CLI exit codes. The provider is forced to fake by the autouse fixture."""

from __future__ import annotations

import json
from io import StringIO

import pytest

from structured_extract.cli import main
from tests.conftest import EXAMPLES, EXPECTED_INVOICE, EXPECTED_JOB, EXPECTED_TICKET


def test_cli_prints_validated_json(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--schema", "invoice", str(EXAMPLES / "invoice.txt")])

    assert code == 0
    assert json.loads(capsys.readouterr().out) == EXPECTED_INVOICE


def test_cli_all_examples(capsys: pytest.CaptureFixture[str]) -> None:
    cases = [
        ("invoice", "invoice.txt", EXPECTED_INVOICE),
        ("job_posting", "job_posting.txt", EXPECTED_JOB),
        ("support_ticket", "support_ticket.txt", EXPECTED_TICKET),
    ]
    for schema, filename, expected in cases:
        code = main(["--schema", schema, str(EXAMPLES / filename)])
        assert code == 0
        assert json.loads(capsys.readouterr().out) == expected


def test_cli_list_schemas(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--list-schemas"]) == 0
    assert capsys.readouterr().out.split() == ["invoice", "job_posting", "support_ticket"]


def test_cli_unknown_schema(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--schema", "receipt", str(EXAMPLES / "invoice.txt")])
    assert code == 2
    assert "receipt" in capsys.readouterr().err


def test_cli_missing_file(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--schema", "invoice", "examples/does-not-exist.txt"])
    assert code == 2
    assert "file not found" in capsys.readouterr().err


def test_cli_stdin(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    text = (EXAMPLES / "invoice.txt").read_text(encoding="utf-8")
    monkeypatch.setattr("sys.stdin", StringIO(text))
    assert main(["--schema", "invoice", "-"]) == 0
    assert json.loads(capsys.readouterr().out)["invoice_number"] == "INV-2026-1042"


def test_cli_unrecognized_text_is_a_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.stdin", StringIO("not an invoice"))
    code = main(["--schema", "invoice", "-"])
    assert code == 1
    assert "missing label" in capsys.readouterr().err


def test_cli_openai_without_key_does_not_fall_back(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("STRUCTURED_EXTRACT_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code = main(["--schema", "invoice", str(EXAMPLES / "invoice.txt")])
    assert code == 2
    err = capsys.readouterr().err
    assert "OPENAI_API_KEY" in err
    assert "INV-2026-1042" not in err


def test_cli_requires_schema() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code == 2
