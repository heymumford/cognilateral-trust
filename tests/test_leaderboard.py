"""A-tests and B-tests for leaderboard generator (Step 18)."""

from __future__ import annotations

from cognilateral_trust.bench.leaderboard import generate_leaderboard
from cognilateral_trust.bench.scoring import BenchScore, DomainScore


class TestLeaderboardHTMLGeneration:
    """A-tests: HTML generation and structure."""

    def test_a1_leaderboard_returns_html_string(self) -> None:
        """generate_leaderboard returns an HTML string."""
        domain_scores = (DomainScore("test", 0.1, 40),)
        score = BenchScore("Model1", 0.1, domain_scores)

        html = generate_leaderboard([score])

        assert isinstance(html, str)
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_a2_leaderboard_contains_table(self) -> None:
        """Generated HTML contains a table."""
        domain_scores = (DomainScore("test", 0.1, 40),)
        score = BenchScore("Model1", 0.1, domain_scores)

        html = generate_leaderboard([score])

        assert "<table>" in html
        assert "</table>" in html
        assert "<thead>" in html
        assert "<tbody>" in html

    def test_a3_leaderboard_includes_model_names(self) -> None:
        """HTML includes model names in table."""
        domain_scores = (DomainScore("test", 0.1, 40),)
        score = BenchScore("ModelA", 0.1, domain_scores)

        html = generate_leaderboard([score])

        assert "ModelA" in html

    def test_a4_leaderboard_includes_scores(self) -> None:
        """HTML includes overall scores."""
        domain_scores = (DomainScore("test", 0.1, 40),)
        score = BenchScore("Model1", 0.85, domain_scores)

        html = generate_leaderboard([score])

        # Score should be formatted as decimal (1.0 - calibration_error)
        assert "0.85" in html or "0.8500" in html

    def test_a5_leaderboard_includes_domain_headers(self) -> None:
        """HTML table headers include domain names."""
        domain_scores = (
            DomainScore("factual", 0.1, 40),
            DomainScore("reasoning", 0.2, 40),
        )
        score = BenchScore("Model1", 0.15, domain_scores)

        html = generate_leaderboard([score])

        assert "factual" in html
        assert "reasoning" in html

    def test_a6_leaderboard_no_external_resources(self) -> None:
        """HTML contains no external URLs (self-contained)."""
        domain_scores = (DomainScore("test", 0.1, 40),)
        score = BenchScore("Model1", 0.1, domain_scores)

        html = generate_leaderboard([score])

        # Should not have href to external URLs
        assert "http://" not in html and "https://" not in html

    def test_a7_leaderboard_has_inline_css(self) -> None:
        """HTML contains inline CSS (no external stylesheets)."""
        domain_scores = (DomainScore("test", 0.1, 40),)
        score = BenchScore("Model1", 0.1, domain_scores)

        html = generate_leaderboard([score])

        assert "<style>" in html
        assert "</style>" in html
        assert "background:" in html

    def test_a8_leaderboard_sorted_by_score_descending(self) -> None:
        """Models sorted by overall_score in descending order."""
        domain_scores = (DomainScore("test", 0.0, 40),)
        model1 = BenchScore("Model1", 0.95, domain_scores)
        model2 = BenchScore("Model2", 0.75, domain_scores)
        model3 = BenchScore("Model3", 0.85, domain_scores)

        html = generate_leaderboard([model1, model2, model3])

        # HTML should render in order: Model1 (0.95), Model3 (0.85), Model2 (0.75)
        pos1 = html.find("Model1")
        pos3 = html.find("Model3")
        pos2 = html.find("Model2")

        assert pos1 < pos3 < pos2

    def test_a9_leaderboard_includes_rank_numbers(self) -> None:
        """HTML includes rank column (1, 2, 3, ...)."""
        domain_scores = (DomainScore("test", 0.0, 40),)
        scores = [
            BenchScore("Model1", 0.95, domain_scores),
            BenchScore("Model2", 0.75, domain_scores),
            BenchScore("Model3", 0.85, domain_scores),
        ]

        html = generate_leaderboard(scores)

        # Should have ranks 1, 2, 3
        assert ">1<" in html
        assert ">2<" in html
        assert ">3<" in html


class TestLeaderboardWithMultipleDomains:
    """B-tests: Multi-domain leaderboards."""

    def test_b1_leaderboard_with_5_domains(self) -> None:
        """Leaderboard with 5 domains renders all columns."""
        domain_scores = (
            DomainScore("factual", 0.1, 40),
            DomainScore("reasoning", 0.15, 40),
            DomainScore("ambiguous", 0.2, 40),
            DomainScore("out_of_distribution", 0.25, 40),
            DomainScore("adversarial", 0.3, 40),
        )
        score = BenchScore("Full5DomainModel", 0.2, domain_scores)

        html = generate_leaderboard([score])

        # All 5 domains should be in headers
        for domain in ["factual", "reasoning", "ambiguous", "out_of_distribution", "adversarial"]:
            assert domain in html

    def test_b2_leaderboard_with_multiple_models(self) -> None:
        """Leaderboard with 3+ models shows all."""
        domain_scores = (DomainScore("test", 0.1, 40),)
        scores = [
            BenchScore("Model1", 0.9, domain_scores),
            BenchScore("Model2", 0.8, domain_scores),
            BenchScore("Model3", 0.7, domain_scores),
            BenchScore("Model4", 0.6, domain_scores),
        ]

        html = generate_leaderboard(scores)

        for i in range(1, 5):
            assert f"Model{i}" in html

    def test_b3_leaderboard_each_model_has_domain_scores(self) -> None:
        """Each model row includes per-domain score cells."""
        domain_scores = (
            DomainScore("domain1", 0.1, 20),
            DomainScore("domain2", 0.2, 20),
        )
        score = BenchScore("TestModel", 0.15, domain_scores)

        html = generate_leaderboard([score])

        # Should have table cells for domain scores
        # 0.9 = 1.0 - 0.1, 0.8 = 1.0 - 0.2
        assert '<td class="domain-score">0.9</td>' in html or "0.9" in html
        assert '<td class="domain-score">0.8</td>' in html or "0.8" in html


class TestLeaderboardEdgeCases:
    """C-tests: Edge cases."""

    def test_c1_empty_domain_scores(self) -> None:
        """Leaderboard with model having no domain_scores still renders."""
        score = BenchScore("NoDomainsModel", 0.5, ())

        html = generate_leaderboard([score])

        assert "NoDomainsModel" in html
        assert "<table>" in html

    def test_c2_single_model(self) -> None:
        """Leaderboard with single model renders as rank 1."""
        domain_scores = (DomainScore("test", 0.1, 40),)
        score = BenchScore("OnlyModel", 0.9, domain_scores)

        html = generate_leaderboard([score])

        assert ">1<" in html
        assert "OnlyModel" in html

    def test_c3_tuple_input_accepted(self) -> None:
        """generate_leaderboard accepts tuple of BenchScore."""
        domain_scores = (DomainScore("test", 0.1, 40),)
        score = BenchScore("TupleModel", 0.9, domain_scores)

        html = generate_leaderboard((score,))

        assert "TupleModel" in html

    def test_c4_high_precision_scores_formatted(self) -> None:
        """Scores with many decimals are formatted to reasonable precision."""
        domain_scores = (DomainScore("test", 0.123456789, 40),)
        score = BenchScore("PrecisionModel", 0.987654321, domain_scores)

        html = generate_leaderboard([score])

        # Should have at least one score (overall or domain)
        assert "0.98" in html or "0.9876" in html

    def test_c5_special_chars_in_model_name(self) -> None:
        """Model names with special chars render without HTML injection."""
        domain_scores = (DomainScore("test", 0.1, 40),)
        score = BenchScore("Model<script>alert('xss')</script>", 0.5, domain_scores)

        html = generate_leaderboard([score])

        # Script tags should be escaped, not present as executable HTML
        assert "<script>" not in html and "</script>" not in html
        assert "Model" in html


class TestLeaderboardFormatting:
    """D-tests: Visual formatting."""

    def test_d1_leaderboard_has_title(self) -> None:
        """HTML has a title element."""
        domain_scores = (DomainScore("test", 0.1, 40),)
        score = BenchScore("Model1", 0.5, domain_scores)

        html = generate_leaderboard([score])

        assert "<title>" in html
        assert "Leaderboard" in html

    def test_d2_leaderboard_has_header(self) -> None:
        """HTML has a visual header section."""
        domain_scores = (DomainScore("test", 0.1, 40),)
        score = BenchScore("Model1", 0.5, domain_scores)

        html = generate_leaderboard([score])

        assert "<h1>" in html or "<h2>" in html

    def test_d3_leaderboard_has_footer(self) -> None:
        """HTML has a footer with explanation."""
        domain_scores = (DomainScore("test", 0.1, 40),)
        score = BenchScore("Model1", 0.5, domain_scores)

        html = generate_leaderboard([score])

        assert "footer" in html.lower() or "calibration" in html.lower()
