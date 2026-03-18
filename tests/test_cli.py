"""Tests for trust-check CLI."""

from __future__ import annotations

from cognilateral_trust.cli import main


class TestCLI:
    def test_act_returns_zero(self, capsys) -> None:
        code = main(["0.5"])
        assert code == 0
        assert "ACT" in capsys.readouterr().out

    def test_escalate_returns_two(self, capsys) -> None:
        code = main(["0.9", "--irreversible"])
        assert code == 2
        assert "ESCALATE" in capsys.readouterr().out

    def test_json_output(self, capsys) -> None:
        main(["0.7", "--json"])
        import json
        data = json.loads(capsys.readouterr().out)
        assert "tier" in data
        assert "verdict" in data
        assert "record_id" in data

    def test_low_confidence_basic(self, capsys) -> None:
        code = main(["0.2"])
        assert code == 0
        assert "basic" in capsys.readouterr().out

    def test_invalid_confidence(self, capsys) -> None:
        code = main(["1.5"])
        assert code == 1

    def test_context_flag(self, capsys) -> None:
        code = main(["0.5", "--context", "deploy to prod"])
        assert code == 0
