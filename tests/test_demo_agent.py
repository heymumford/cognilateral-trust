"""Sprint S5: Tests for the demo trust agent.

Validates that the demo agent:
- Runs without error
- Produces both ACT and ESCALATE decisions
- Shows non-zero calibration accuracy after completing
- Produces a summary with expected fields
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make examples directory importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))


class TestDemoTrustAgent:
    def test_demo_runs_without_error(self, tmp_path: Path) -> None:
        """The demo completes without raising any exception."""
        from demo_trust_agent import run_demo

        result = run_demo(persist_path=tmp_path, seed=42)
        assert result is not None

    def test_demo_produces_act_decisions(self, tmp_path: Path) -> None:
        from demo_trust_agent import run_demo

        result = run_demo(persist_path=tmp_path, seed=42)
        assert result["act_count"] > 0, "Demo should produce at least one ACT decision"

    def test_demo_produces_escalate_decisions(self, tmp_path: Path) -> None:
        from demo_trust_agent import run_demo

        result = run_demo(persist_path=tmp_path, seed=42)
        assert result["escalate_count"] > 0, "Demo should produce at least one ESCALATE decision"

    def test_calibration_accuracy_nonzero_after_demo(self, tmp_path: Path) -> None:
        from demo_trust_agent import run_demo

        result = run_demo(persist_path=tmp_path, seed=42)
        assert result["calibration_accuracy"] > 0.0, "Accuracy should be > 0 after outcomes recorded"

    def test_summary_has_required_fields(self, tmp_path: Path) -> None:
        from demo_trust_agent import run_demo

        result = run_demo(persist_path=tmp_path, seed=42)
        required = {"act_count", "escalate_count", "calibration_accuracy", "total_tasks"}
        assert required.issubset(result.keys()), f"Missing fields: {required - result.keys()}"

    def test_total_tasks_equals_act_plus_escalate(self, tmp_path: Path) -> None:
        from demo_trust_agent import run_demo

        result = run_demo(persist_path=tmp_path, seed=42)
        assert result["total_tasks"] == result["act_count"] + result["escalate_count"]

    def test_demo_runs_reproducibly_with_seed(self, tmp_path: Path) -> None:
        """Same seed → same results."""
        from demo_trust_agent import run_demo

        r1 = run_demo(persist_path=tmp_path / "run1", seed=99)
        r2 = run_demo(persist_path=tmp_path / "run2", seed=99)
        assert r1["act_count"] == r2["act_count"]
        assert r1["escalate_count"] == r2["escalate_count"]

    def test_persistence_files_created(self, tmp_path: Path) -> None:
        from demo_trust_agent import run_demo

        run_demo(persist_path=tmp_path, seed=42)
        jsonl_files = list(tmp_path.glob("*.jsonl"))
        assert len(jsonl_files) > 0, "Demo should create JSONL files"

    def test_demo_with_print_suppressed(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """Demo can run silently — no crash when stdout captured."""
        from demo_trust_agent import run_demo

        run_demo(persist_path=tmp_path, seed=42, quiet=True)
        # No assertion on output — just proving it doesn't crash when quiet
