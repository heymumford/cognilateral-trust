"""Load simulator — generate realistic trust evaluation traffic.

Simulates multiple AI agents making decisions with varying confidence.
Useful for:
- Populating the live dashboard with data
- Stress testing the trust engine
- Demonstrating calibration accuracy convergence

Usage:
    pip install cognilateral-trust
    python load_simulator.py               # 100 decisions
    python load_simulator.py --count 500   # 500 decisions
    python load_simulator.py --api http://localhost:8000  # against API
"""

from __future__ import annotations

import argparse
import random
import time

from cognilateral_trust import CalibratedTrustEngine


AGENT_PROFILES = [
    {"name": "deploy-bot", "base_confidence": 0.85, "reversible_rate": 0.3, "accuracy": 0.88},
    {"name": "email-agent", "base_confidence": 0.7, "reversible_rate": 0.1, "accuracy": 0.75},
    {"name": "data-analyst", "base_confidence": 0.6, "reversible_rate": 0.9, "accuracy": 0.65},
    {"name": "customer-support", "base_confidence": 0.5, "reversible_rate": 0.5, "accuracy": 0.55},
    {"name": "research-bot", "base_confidence": 0.4, "reversible_rate": 0.95, "accuracy": 0.45},
]


def simulate(count: int = 100, seed: int = 42) -> None:
    engine = CalibratedTrustEngine()
    random.seed(seed)

    print(f"Simulating {count} agent decisions across {len(AGENT_PROFILES)} agent profiles")
    print("=" * 65)

    act_count = 0
    escalate_count = 0
    correct_count = 0

    for i in range(count):
        agent = random.choice(AGENT_PROFILES)
        noise = random.gauss(0, 0.1)
        confidence = max(0.05, min(0.99, agent["base_confidence"] + noise))
        is_reversible = random.random() < agent["reversible_rate"]

        result = engine.evaluate(
            confidence,
            is_reversible=is_reversible,
            context=f"{agent['name']} decision #{i+1}",
        )

        # Simulate outcome based on agent accuracy profile
        correct = random.random() < agent["accuracy"]
        engine.record_outcome(result.accountability_record.record_id, correct=correct)

        if result.should_proceed:
            act_count += 1
        else:
            escalate_count += 1
        if correct:
            correct_count += 1

        if (i + 1) % max(1, count // 10) == 0:
            stats = engine.stats
            print(
                f"  [{i+1:4d}/{count}] "
                f"agent={agent['name']:<18s} "
                f"conf={confidence:.2f} "
                f"{'ACT' if result.should_proceed else 'ESC':>3s} "
                f"{'OK' if correct else 'FAIL':>4s} "
                f"| accuracy={stats['accuracy']:.3f}"
            )

    print("=" * 65)
    stats = engine.stats
    print(f"Final calibration accuracy: {stats['accuracy']:.4f}")
    print(f"ACT: {act_count} | ESCALATE: {escalate_count} | ACT rate: {act_count/count:.1%}")
    print(f"Correct: {correct_count}/{count} ({correct_count/count:.1%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate trust evaluation traffic")
    parser.add_argument("--count", type=int, default=100, help="Number of decisions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    simulate(count=args.count, seed=args.seed)
