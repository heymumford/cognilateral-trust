"""A-tests and B-tests for TrustBench CLI (Step 17)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from cognilateral_trust.bench.cli import main, run_benchmark, generate_fingerprint_from_file


class TestCLIArgumentParsing:
    """A-tests: CLI argument parsing."""

    def test_a1_run_subcommand_requires_model(self) -> None:
        """bench run without --model shows usage."""
        with pytest.raises(SystemExit):
            main(["run"])

    def test_a2_run_with_model_succeeds(self, capsys: pytest.CaptureFixture) -> None:
        """bench run --model <name> succeeds and outputs JSON."""
        exit_code = main(["run", "--model", "TestModel"])

        assert exit_code == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["model"] == "TestModel"

    def test_a3_run_with_output_writes_file(self) -> None:
        """bench run --model <name> --output <path> writes JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"
            exit_code = main(["run", "--model", "FileModel", "--output", str(output_path)])

            assert exit_code == 0
            assert output_path.exists()
            with open(output_path) as f:
                result = json.load(f)
            assert result["model"] == "FileModel"

    def test_a4_fingerprint_subcommand_requires_results(self) -> None:
        """bench fingerprint without --results shows usage."""
        with pytest.raises(SystemExit):
            main(["fingerprint"])

    def test_a5_fingerprint_with_results_succeeds(self, capsys: pytest.CaptureFixture) -> None:
        """bench fingerprint --results <path> succeeds and outputs JSON."""
        # First create a results file
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "bench_results.json"
            run_benchmark("FPModel", results_path)

            # Now test fingerprint command
            exit_code = main(["fingerprint", "--results", str(results_path)])

            assert exit_code == 0
            captured = capsys.readouterr()
            output = json.loads(captured.out)
            assert output["model"] == "FPModel"
            assert "spokes" in output

    def test_a6_no_command_shows_help(self, capsys: pytest.CaptureFixture) -> None:
        """main() with no subcommand shows help and returns non-zero (invalid usage)."""
        exit_code = main([])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "TrustBench" in captured.out or "usage" in captured.out.lower()


class TestRunBenchmark:
    """B-tests: Benchmark execution."""

    def test_b1_run_benchmark_returns_dict(self) -> None:
        """run_benchmark returns dict with model and scores."""
        result = run_benchmark("TestModel")

        assert isinstance(result, dict)
        assert "model" in result
        assert "overall_score" in result
        assert "domain_scores" in result

    def test_b2_run_benchmark_model_name_in_output(self) -> None:
        """Output model name matches input."""
        result = run_benchmark("CustomModelName")

        assert result["model"] == "CustomModelName"

    def test_b3_run_benchmark_scores_are_valid(self) -> None:
        """Output scores are in valid range [0.0, 1.0]."""
        result = run_benchmark("ValidModel")

        assert 0.0 <= result["overall_score"] <= 1.0
        for ds in result["domain_scores"]:
            assert 0.0 <= ds["calibration_error"] <= 1.0

    def test_b4_run_benchmark_has_all_domains(self) -> None:
        """Output includes all 5 domains."""
        result = run_benchmark("AllDomainsModel")

        domains = {ds["domain"] for ds in result["domain_scores"]}
        assert "factual" in domains
        assert "reasoning" in domains
        assert "ambiguous" in domains
        assert "out_of_distribution" in domains
        assert "adversarial" in domains

    def test_b5_run_benchmark_creates_parent_dirs(self) -> None:
        """Output path creation makes parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "a" / "b" / "c" / "results.json"
            run_benchmark("NestedModel", nested_path)

            assert nested_path.exists()


class TestFingerprintFromFile:
    """C-tests: Loading and generating fingerprints."""

    def test_c1_fingerprint_from_file_returns_dict(self) -> None:
        """generate_fingerprint_from_file returns dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "results.json"
            run_benchmark("FPModel", results_path)

            result = generate_fingerprint_from_file(results_path)

            assert isinstance(result, dict)
            assert "model" in result
            assert "spokes" in result

    def test_c2_fingerprint_from_file_missing_file(self) -> None:
        """generate_fingerprint_from_file raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            generate_fingerprint_from_file("/nonexistent/path/results.json")

    def test_c3_fingerprint_matches_model(self) -> None:
        """Fingerprint model name matches results model name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "results.json"
            run_benchmark("MatchModel", results_path)

            fp = generate_fingerprint_from_file(results_path)

            assert fp["model"] == "MatchModel"

    def test_c4_fingerprint_spokes_match_domains(self) -> None:
        """Fingerprint spokes include all domains from results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "results.json"
            run_benchmark("DomainsModel", results_path)

            fp = generate_fingerprint_from_file(results_path)
            fp_domains = {s["domain"] for s in fp["spokes"]}

            assert "factual" in fp_domains
            assert "reasoning" in fp_domains


class TestCLIErrorHandling:
    """D-tests: Error handling."""

    def test_d1_cli_handles_invalid_results_path(self, capsys: pytest.CaptureFixture) -> None:
        """CLI handles missing results file gracefully."""
        exit_code = main(["fingerprint", "--results", "/nonexistent/file.json"])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err or "not found" in captured.err.lower()

    def test_d2_cli_returns_zero_on_success(self) -> None:
        """CLI returns exit code 0 on success."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"
            exit_code = main(["run", "--model", "SuccessModel", "--output", str(output_path)])

            assert exit_code == 0
