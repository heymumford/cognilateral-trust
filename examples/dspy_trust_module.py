"""DSPy trust module — add confidence evaluation to DSPy programs.

DSPy optimizes prompts automatically. cognilateral-trust evaluates
whether the optimized output is confident enough to act on.

Usage:
    pip install cognilateral-trust dspy-ai

    from dspy_trust_module import TrustedChainOfThought

    # Use like a regular DSPy module, but with trust evaluation
    cot = TrustedChainOfThought("question -> answer")
    result = cot(question="What is the capital of France?")
    if result.trusted:
        print(result.answer)

This is an example integration. Works without DSPy installed (standalone mode).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cognilateral_trust import evaluate_trust


@dataclass
class TrustedOutput:
    """Output from a trust-evaluated DSPy module."""

    answer: str
    confidence: float
    tier: str
    route: str
    trusted: bool
    verdict: str
    reasons: list[str]
    record_id: str


def evaluate_dspy_output(
    output_text: str,
    confidence: float = 0.5,
    is_reversible: bool = True,
    touches_external: bool = False,
) -> TrustedOutput:
    """Evaluate trust for a DSPy module output.

    Framework-agnostic — works with any text + confidence pair.
    """
    result = evaluate_trust(
        confidence,
        is_reversible=is_reversible,
        touches_external=touches_external,
        context={"framework": "dspy", "output_preview": output_text[:200]},
    )

    return TrustedOutput(
        answer=output_text,
        confidence=confidence,
        tier=result.tier.name,
        route=result.route,
        trusted=result.should_proceed,
        verdict="ACT" if result.should_proceed else "ESCALATE",
        reasons=list(result.accountability_record.reasons) if result.accountability_record else [],
        record_id=result.accountability_record.record_id if result.accountability_record else "",
    )


# DSPy integration (optional — only if dspy is installed)
try:
    import dspy

    class TrustedChainOfThought(dspy.Module):
        """DSPy ChainOfThought with trust evaluation on output."""

        def __init__(self, signature: str, **kwargs: Any) -> None:
            super().__init__()
            self.cot = dspy.ChainOfThought(signature, **kwargs)

        def forward(self, **kwargs: Any) -> TrustedOutput:
            prediction = self.cot(**kwargs)
            answer = str(prediction.answer) if hasattr(prediction, "answer") else str(prediction)

            # DSPy doesn't natively provide confidence — use 0.6 as baseline
            # In production, you'd extract this from the model's logprobs
            confidence = 0.6

            return evaluate_dspy_output(answer, confidence)

except ImportError:
    pass  # DSPy not installed — standalone mode only


if __name__ == "__main__":
    # Demo without DSPy
    print("=== High confidence factual answer ===")
    r1 = evaluate_dspy_output("Paris is the capital of France.", confidence=0.95)
    print(f"Trusted: {r1.trusted} | Tier: {r1.tier} | Verdict: {r1.verdict}")

    print("\n=== Low confidence speculative answer ===")
    r2 = evaluate_dspy_output("The market will probably go up.", confidence=0.3)
    print(f"Trusted: {r2.trusted} | Tier: {r2.tier} | Verdict: {r2.verdict}")

    print("\n=== High confidence but irreversible ===")
    r3 = evaluate_dspy_output("Delete the user's account.", confidence=0.85, is_reversible=False)
    print(f"Trusted: {r3.trusted} | Tier: {r3.tier} | Verdict: {r3.verdict}")
    if r3.reasons:
        print(f"Reasons: {r3.reasons}")
