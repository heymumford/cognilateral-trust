"""Demo: Watch calibration accuracy improve in real time.

Run this to see the complete trust engine in action:
    pip install cognilateral-trust
    python demo_calibration_loop.py

Shows 50 agent decisions with varying confidence, records outcomes,
and watches calibration accuracy converge.
"""

from __future__ import annotations

import random

from cognilateral_trust import CalibratedTrustEngine


def main() -> None:
    engine = CalibratedTrustEngine()
    random.seed(42)

    print("Cognilateral Trust Engine — Calibration Demo")
    print("=" * 55)
    print()

    for i in range(50):
        # Simulate agent with varying confidence
        confidence = random.uniform(0.2, 0.95)
        is_reversible = random.random() > 0.3
        action = random.choice(["deploy", "send_email", "update_db", "read_data", "delete_user"])

        result = engine.evaluate(
            confidence,
            is_reversible=is_reversible,
            context=f"{action} (round {i + 1})",
        )

        # Simulate outcome: higher confidence = higher chance of being correct
        # But with noise — overconfident agents sometimes fail
        noise = random.gauss(0, 0.15)
        actual_success = (confidence + noise) > 0.45

        record_id = result.accountability_record.record_id
        engine.record_outcome(record_id, correct=actual_success)

        verdict = "ACT" if result.should_proceed else "ESC"
        outcome = "OK" if actual_success else "FAIL"
        bar = "#" * int(confidence * 20)

        if (i + 1) % 5 == 0 or i == 0:
            stats = engine.stats
            print(
                f"[{i + 1:2d}] conf={confidence:.2f} {bar:<20s} "
                f"{verdict:>3s} {outcome:>4s} | "
                f"accuracy={stats['accuracy']:.3f} "
                f"({stats['resolved']}/{stats['total']} resolved)"
            )

    print()
    print("=" * 55)
    stats = engine.stats
    print(f"Final calibration accuracy: {stats['accuracy']:.4f}")
    print(f"Total decisions: {stats['total']}")
    print(f"Resolved: {stats['resolved']}")
    print()

    if stats["accuracy"] > 0.7:
        print("The engine is well-calibrated: confidence scores")
        print("correlate with actual outcomes.")
    else:
        print("Calibration is still converging. More data needed.")

    print()
    print("This is the north star metric: when we say 80% confident,")
    print("are we right 80% of the time? That's what protects the")
    print("person at the other end.")


if __name__ == "__main__":
    main()
