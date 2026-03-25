"""TrustBench CLI — evaluate model calibration on epistemic honesty scenarios."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cognilateral_trust.bench.fingerprint import generate_fingerprint, fingerprint_to_dict
from cognilateral_trust.bench.scenarios import BenchScenario, load_scenarios
from cognilateral_trust.bench.scoring import BenchResult, BenchScore, DomainScore, score_results

__all__ = ["main"]


def run_benchmark(
    model_name: str,
    output_path: str | Path | None = None,
) -> dict:
    """Run benchmark on a model.

    For now, uses a mock model (always returns 0.75 confidence).
    Future: plug in actual model evaluation.

    Args:
        model_name: Name of the model
        output_path: Optional path to write results JSON

    Returns:
        Dict with benchmark score and result path (if written)
    """
    scenarios = load_scenarios()

    # Group scenarios by domain
    by_domain: dict[str, list[BenchScenario]] = {}
    for scenario in scenarios:
        by_domain.setdefault(scenario.domain, []).append(scenario)

    # Run mock evaluation on each scenario
    results_by_domain: dict[str, list[BenchResult]] = {}
    for domain, domain_scenarios in by_domain.items():
        results = []
        for scenario in domain_scenarios:
            # Mock model: always 0.75 confidence
            model_confidence = 0.75

            # Mock correctness: random but consistent with expected_confidence range
            # For now, assume 0.75 is within the expected range (simplified)
            correctness = (
                1.0
                if (scenario.expected_confidence_low <= model_confidence <= scenario.expected_confidence_high)
                else 0.0
            )

            results.append(
                BenchResult(
                    scenario_id=scenario.id,
                    model_confidence=model_confidence,
                    correctness=correctness,
                )
            )
        results_by_domain[domain] = results

    # Score the results
    bench_score = score_results(model_name, results_by_domain)

    result = {
        "model": bench_score.model,
        "overall_score": bench_score.overall_score,
        "domain_scores": [
            {
                "domain": ds.domain,
                "calibration_error": ds.calibration_error,
                "scenario_count": ds.scenario_count,
            }
            for ds in bench_score.domain_scores
        ],
    }

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

    return result


def generate_fingerprint_from_file(results_path: str | Path) -> dict:
    """Generate fingerprint from saved benchmark results.

    Args:
        results_path: Path to results JSON file

    Returns:
        Dict with fingerprint (model, spokes)
    """
    results_file = Path(results_path)
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_file, "r") as f:
        results_dict = json.load(f)

    # Reconstruct BenchScore from dict
    model = results_dict["model"]
    overall_score = results_dict["overall_score"]
    domain_scores = [
        DomainScore(
            domain=ds["domain"],
            calibration_error=ds["calibration_error"],
            scenario_count=ds["scenario_count"],
        )
        for ds in results_dict["domain_scores"]
    ]

    bench_score = BenchScore(
        model=model,
        overall_score=overall_score,
        domain_scores=tuple(domain_scores),
    )

    fp = generate_fingerprint(bench_score)
    return fingerprint_to_dict(fp)


def main(args: list[str] | None = None) -> int:
    """CLI entry point for TrustBench.

    Subcommands:
      bench run --model <name> [--output path]
      bench fingerprint --results path

    Args:
        args: Command-line arguments (default: sys.argv[1:])

    Returns:
        Exit code (0 = success)
    """
    parser = argparse.ArgumentParser(description="TrustBench — evaluate model epistemic honesty")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subcommand: run
    run_parser = subparsers.add_parser("run", help="Run benchmark on a model")
    run_parser.add_argument("--model", required=True, help="Model name or identifier")
    run_parser.add_argument(
        "--output",
        default=None,
        help="Output path for results JSON (optional)",
    )

    # Subcommand: fingerprint
    fp_parser = subparsers.add_parser(
        "fingerprint",
        help="Generate calibration fingerprint from results",
    )
    fp_parser.add_argument(
        "--results",
        required=True,
        help="Path to benchmark results JSON",
    )

    parsed = parser.parse_args(args)

    if not parsed.command:
        parser.print_help()
        return 1

    try:
        if parsed.command == "run":
            result = run_benchmark(parsed.model, parsed.output)
            print(json.dumps(result, indent=2))
            return 0

        elif parsed.command == "fingerprint":
            fp_result = generate_fingerprint_from_file(parsed.results)
            print(json.dumps(fp_result, indent=2))
            return 0

        else:
            parser.print_help()
            return 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
