"""Demo trust agent — end-to-end proof that the trust engine works.

Simulates an agent receiving tasks with varying confidence levels. Uses:
- CalibratedTrustEngine with JSONL persistence (state survives restarts)
- extract_confidence_from_text() to parse "LLM responses"
- @trust_gate decorator to protect critical actions
- Warrants for time-sensitive decisions

The agent processes tasks in three phases:
1. Parse simulated LLM response to extract confidence
2. Apply trust_gate — low-confidence tasks escalate automatically
3. Record actual outcomes to improve calibration over time

Run directly to see a live demo:
    uv run python examples/demo_trust_agent.py

Or import run_demo() for testing.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from cognilateral_trust.calibrated import CalibratedTrustEngine
from cognilateral_trust.extractors import extract_confidence_from_text
from cognilateral_trust.middleware import TrustEscalation, trust_gate
from cognilateral_trust.warrants import Warrant, WarrantStore, evaluate_warrant

# ---------------------------------------------------------------------------
# Simulated task pool
# ---------------------------------------------------------------------------

_TASKS: list[dict[str, Any]] = [
    # High confidence, routine tasks
    {
        "name": "retrieve database record",
        "llm_response": "Found the record. Confidence: 0.97",
        "correct": True,
        "is_reversible": True,
    },
    {
        "name": "calculate average",
        "llm_response": "The average is 42.5. I'm 95% confident in this computation.",
        "correct": True,
        "is_reversible": True,
    },
    {
        "name": "format output",
        "llm_response": "Formatted successfully. Confidence: 0.92",
        "correct": True,
        "is_reversible": True,
    },
    # Medium confidence
    {
        "name": "classify sentiment",
        "llm_response": "The sentiment appears positive. confidence_score: 0.72",
        "correct": True,
        "is_reversible": True,
    },
    {
        "name": "summarize document",
        "llm_response": "Here is a summary of the key points. I have moderate confidence in this.",
        "correct": True,
        "is_reversible": True,
    },
    # Low confidence — should escalate
    {
        "name": "predict future revenue",
        "llm_response": "Based on trends, revenue may increase. I'm quite uncertain about this.",
        "correct": False,
        "is_reversible": True,
    },
    {
        "name": "diagnose system issue",
        "llm_response": "The issue might be network-related. Confidence: 0.31",
        "correct": False,
        "is_reversible": True,
    },
    # Irreversible action requiring high confidence
    {
        "name": "send email to all users",
        "llm_response": "Ready to send notification. Confidence: 0.88",
        "correct": True,
        "is_reversible": False,
    },
    {
        "name": "delete archived records",
        "llm_response": "Identified records for deletion. Confidence: 0.65",
        "correct": True,
        "is_reversible": False,
    },
    # Warrant-backed tasks
    {
        "name": "execute trade order",
        "llm_response": "Market conditions confirmed. Confidence: 0.82",
        "correct": True,
        "is_reversible": False,
        "warrant_ttl": 60,  # 60-second warrant
    },
    # Edge cases
    {
        "name": "generate report",
        "llm_response": "The report is complete.",  # no confidence signal
        "correct": True,
        "is_reversible": True,
    },
    {
        "name": "parse ambiguous input",
        "llm_response": "Parsed with high confidence in the interpretation.",
        "correct": False,
        "is_reversible": True,
    },
]

_DEFAULT_GATE_THRESHOLD = 0.5
_WARRANT_STORE = WarrantStore()


# ---------------------------------------------------------------------------
# Trust-gated critical action
# ---------------------------------------------------------------------------


@trust_gate(min_confidence=_DEFAULT_GATE_THRESHOLD)
def execute_action(task_name: str, *, confidence: float) -> str:
    """Execute a trusted action. Decorated with @trust_gate."""
    return f"EXECUTED: {task_name}"


# ---------------------------------------------------------------------------
# Core demo logic
# ---------------------------------------------------------------------------


def _process_task(
    task: dict[str, Any],
    engine: CalibratedTrustEngine,
    *,
    quiet: bool = False,
) -> dict[str, Any]:
    """Process a single task through the full trust pipeline.

    Returns a dict with: name, confidence, decision, verdict, outcome_correct.
    """
    task_name = task["name"]
    llm_response = task["llm_response"]
    is_reversible = task.get("is_reversible", True)
    expected_correct = task.get("correct", True)

    # 1. Extract confidence from simulated LLM response
    confidence = extract_confidence_from_text(llm_response)
    if confidence is None:
        confidence = 0.5  # neutral default when no signal

    # 2. If task has a warrant, evaluate it (may reduce effective confidence)
    if "warrant_ttl" in task:
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        warrant = Warrant(
            evidence_source=f"llm-response:{task_name}",
            issued_at=now,
            expires_at=now + timedelta(seconds=task["warrant_ttl"]),
            confidence=confidence,
            decay_rate=0.2,
        )
        wid = _WARRANT_STORE.add(warrant)
        confidence = evaluate_warrant(warrant)
        _ = wid  # tracked but not needed for demo

    # 3. Evaluate trust through the engine (tracks prediction)
    result = engine.evaluate(confidence, is_reversible=is_reversible, context=task_name)

    # 4. Apply trust gate — raises TrustEscalation if below threshold or escalated
    verdict = "ACT"
    decision = "SKIP"
    try:
        if not result.should_proceed:
            raise TrustEscalation(confidence, _DEFAULT_GATE_THRESHOLD)
        output = execute_action(task_name, confidence=confidence)
        decision = output
        verdict = "ACT"
    except TrustEscalation:
        verdict = "ESCALATE"
        decision = f"ESCALATED: {task_name} (confidence={confidence:.2f})"

    # 5. Record outcome for calibration
    record_id = result.accountability_record.record_id
    actual_correct = expected_correct if verdict == "ACT" else False
    engine.record_outcome(record_id, correct=actual_correct)

    if not quiet:
        print(f"  [{verdict:8s}] {task_name!r:<40} conf={confidence:.2f}")

    return {
        "name": task_name,
        "confidence": confidence,
        "decision": decision,
        "verdict": verdict,
        "outcome_correct": actual_correct,
    }


def run_demo(
    *,
    persist_path: Path | str | None = None,
    seed: int | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    """Run the demo agent. Returns a summary dict.

    Args:
        persist_path: If set, predictions and audit records are persisted to JSONL.
        seed:         Random seed for reproducibility.
        quiet:        Suppress print output.

    Returns dict with:
        act_count, escalate_count, total_tasks, calibration_accuracy,
        decisions: list of per-task result dicts.
    """
    if seed is not None:
        random.seed(seed)

    if persist_path is not None:
        persist_path = Path(persist_path)
        persist_path.mkdir(parents=True, exist_ok=True)

    engine = CalibratedTrustEngine(persist_path=persist_path)

    if not quiet:
        print("\n=== Cognilateral Trust Agent Demo ===")
        print(f"Tasks: {len(_TASKS)} | Persistence: {'enabled' if persist_path else 'memory-only'}\n")

    decisions = []
    for task in _TASKS:
        task_result = _process_task(task, engine, quiet=quiet)
        decisions.append(task_result)

    act_count = sum(1 for d in decisions if d["verdict"] == "ACT")
    escalate_count = sum(1 for d in decisions if d["verdict"] == "ESCALATE")
    accuracy = engine.calibration_accuracy()

    if not quiet:
        print("\n--- Summary ---")
        print(f"  Total tasks:    {len(decisions)}")
        print(f"  ACT decisions:  {act_count}")
        print(f"  ESCALATE:       {escalate_count}")
        print(f"  Calibration:    {accuracy:.3f}")

    return {
        "total_tasks": len(decisions),
        "act_count": act_count,
        "escalate_count": escalate_count,
        "calibration_accuracy": accuracy,
        "decisions": decisions,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys
    from pathlib import Path

    persist = Path.home() / ".cognilateral-demo"
    result = run_demo(persist_path=persist)

    print(f"\nJSONL files saved to: {persist}")
    print("Run again to see calibration accumulate across restarts.")
    sys.exit(0)
