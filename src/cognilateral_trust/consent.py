"""User consent — configurable trust sensitivity.

The end user controls how cautious the trust engine is. This is the
philosophical completion of the D-05 welfare constraint: the person
downstream gets to adjust the dial, not just the developer.

Sensitivity shifts the should_proceed threshold without changing the
tier mapping. C7 is always C7 — but at high sensitivity, C7 might
not be enough to act autonomously.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "Sensitivity",
    "apply_sensitivity",
    "parse_sensitivity",
    "CONSERVATIVE",
    "BALANCED",
    "AGGRESSIVE",
]


class Sensitivity(Enum):
    """User-configurable trust sensitivity level."""

    LOW = "low"
    DEFAULT = "default"
    HIGH = "high"


def parse_sensitivity(value: str | None) -> Sensitivity:
    """Parse a sensitivity string into the enum.

    Returns Sensitivity.DEFAULT for None or empty string.
    Raises ValueError for unrecognized values.
    """
    if value is None or value == "":
        return Sensitivity.DEFAULT
    try:
        return Sensitivity(value.lower())
    except ValueError:
        raise ValueError(f"Invalid sensitivity: {value!r}. Must be 'low', 'default', or 'high'.")


def apply_sensitivity(
    should_proceed: bool,
    sensitivity: Sensitivity,
    confidence: float = 0.5,
) -> bool:
    """Adjust the should_proceed decision based on user sensitivity.

    - DEFAULT: no change
    - HIGH: requires higher confidence to proceed (shifts threshold up by 0.2)
    - LOW: permits lower confidence to proceed (shifts threshold down by 0.2)

    The confidence parameter is used by HIGH/LOW to apply the threshold shift.
    The tier is never changed — only the proceed/escalate verdict.
    """
    if sensitivity == Sensitivity.DEFAULT:
        return should_proceed

    if sensitivity == Sensitivity.HIGH:
        # High sensitivity: escalate if confidence is below 0.7
        # (even if the default routing said proceed)
        if confidence < 0.7:
            return False
        return should_proceed

    if sensitivity == Sensitivity.LOW:
        # Low sensitivity: permit if confidence is above 0.3
        # (even if the default routing said escalate)
        if confidence >= 0.3:
            return True
        return should_proceed

    return should_proceed


# ---------------------------------------------------------------------------
# D75-W7: Named consent presets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsentPreset:
    """Named consent preset with a min_confidence threshold and sensitivity."""

    name: str
    sensitivity: Sensitivity
    min_confidence: float
    always_escalate_irreversible: bool = False

    def evaluate(self, confidence: float, *, is_reversible: bool = True) -> bool:
        """Return True if the action should proceed under this preset."""
        from cognilateral_trust.evaluate import evaluate_trust

        result = evaluate_trust(confidence, is_reversible=is_reversible)
        should = apply_sensitivity(result.should_proceed, self.sensitivity, confidence)
        if confidence < self.min_confidence:
            return False
        if self.always_escalate_irreversible and not is_reversible:
            return False
        return should


CONSERVATIVE = ConsentPreset(
    name="conservative",
    sensitivity=Sensitivity.HIGH,
    min_confidence=0.8,
    always_escalate_irreversible=True,
)

BALANCED = ConsentPreset(
    name="balanced",
    sensitivity=Sensitivity.DEFAULT,
    min_confidence=0.5,
    always_escalate_irreversible=True,
)

AGGRESSIVE = ConsentPreset(
    name="aggressive",
    sensitivity=Sensitivity.LOW,
    min_confidence=0.15,
    always_escalate_irreversible=False,
)
