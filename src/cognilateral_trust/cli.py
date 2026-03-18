"""CLI for cognilateral-trust — evaluate confidence from the command line.

Usage:
    trust-check 0.7
    trust-check 0.7 --irreversible
    trust-check 0.3 --external
    trust-check 0.9 --json
"""

from __future__ import annotations

import argparse
import json
import sys

from cognilateral_trust import evaluate_trust


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="trust-check",
        description="Evaluate epistemic trust for an AI confidence score.",
    )
    parser.add_argument("confidence", type=float, help="Confidence score (0.0-1.0)")
    parser.add_argument("--irreversible", action="store_true", help="Action is irreversible")
    parser.add_argument("--external", action="store_true", help="Action touches external systems")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Output as JSON")
    parser.add_argument("--context", type=str, default=None, help="Context string for accountability record")

    args = parser.parse_args(argv)

    if not 0.0 <= args.confidence <= 1.0:
        print(f"Error: confidence must be 0.0-1.0, got {args.confidence}", file=sys.stderr)
        return 1

    ctx = {"source": "cli"}
    if args.context:
        ctx["description"] = args.context

    result = evaluate_trust(
        args.confidence,
        is_reversible=not args.irreversible,
        touches_external=args.external,
        context=ctx,
    )

    if args.as_json:
        output = {
            "confidence": result.confidence,
            "tier": result.tier.name,
            "route": result.route,
            "should_proceed": result.should_proceed,
            "verdict": "ACT" if result.should_proceed else "ESCALATE",
            "requires_warrant_check": result.requires_warrant_check,
            "requires_sovereignty_gate": result.requires_sovereignty_gate,
        }
        if result.accountability_record:
            output["record_id"] = result.accountability_record.record_id
            output["reasons"] = list(result.accountability_record.reasons)
        print(json.dumps(output, indent=2))
    else:
        verdict = "ACT" if result.should_proceed else "ESCALATE"
        tier = result.tier.name
        route = result.route

        if result.should_proceed:
            print(f"{verdict} — {tier} ({route})")
        else:
            reasons = ", ".join(result.accountability_record.reasons) if result.accountability_record else "unknown"
            print(f"{verdict} — {tier} ({route}): {reasons}")

    return 0 if result.should_proceed else 2


if __name__ == "__main__":
    raise SystemExit(main())
