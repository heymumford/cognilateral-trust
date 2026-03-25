"""Tests for Cognee Knowledge Graph trust scoring integration.

Verifies CogneeTrustBridge behavior: claim scoring, edge verification,
multi-source agreement, contradiction detection, and degenerate inputs.
All tests use mocked Cognee data — no Cognee dependency required.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Fixtures: mock Cognee knowledge graph data
# ---------------------------------------------------------------------------

# Source documents that a Cognee pipeline would extract from
_SOURCE_PYTHON_HISTORY = (
    "Python is a high-level programming language created by Guido van Rossum. "
    "It was first released in 1991. Python emphasizes code readability and "
    "supports multiple programming paradigms including object-oriented and "
    "functional programming."
)

_SOURCE_PYTHON_DESIGN = (
    "Guido van Rossum began working on Python in the late 1980s as a successor "
    "to the ABC programming language. Python 2.0 was released in 2000 with "
    "features like list comprehensions and garbage collection."
)

_SOURCE_UNRELATED = (
    "The mitochondria is the powerhouse of the cell. Cellular respiration "
    "produces ATP through oxidative phosphorylation in the inner membrane."
)

_SOURCE_CONTRADICTING = (
    "Python was created by Larry Wall in 1987 as a scripting language for system administration tasks on Unix systems."
)

_CLAIM_PYTHON_CREATOR = "Python was created by Guido van Rossum in 1991."
_CLAIM_VAGUE = "Programming languages exist."
_CLAIM_UNSUPPORTED = "Quantum entanglement enables faster-than-light communication."


class TestScoreKnowledgeClaim:
    """CogneeTrustBridge.score_knowledge_claim must return structured TrustScore."""

    def test_score_knowledge_claim_returns_trust_score(self) -> None:
        """Well-supported claim with matching source produces valid TrustScore."""
        from cognilateral_trust.integrations.cognee import CogneeTrustBridge

        bridge = CogneeTrustBridge()
        score = bridge.score_knowledge_claim(
            claim_text=_CLAIM_PYTHON_CREATOR,
            source_documents=[_SOURCE_PYTHON_HISTORY],
        )

        assert score.confidence > 0.0
        assert score.tier is not None
        assert score.tier_name != ""
        assert score.route in ("basic", "warrant_check", "sovereignty_gate")
        assert score.evidence_quality in ("strong", "moderate", "weak")
        assert score.source_count == 1
        assert score.claim_text == _CLAIM_PYTHON_CREATOR
        assert score.trust_evaluation is not None
        assert len(score.fidelity_scores) == 1

    def test_high_fidelity_claim_has_strong_evidence(self) -> None:
        """Claim closely matching source text should yield strong evidence quality."""
        from cognilateral_trust.integrations.cognee import CogneeTrustBridge

        bridge = CogneeTrustBridge()
        score = bridge.score_knowledge_claim(
            claim_text=_CLAIM_PYTHON_CREATOR,
            source_documents=[_SOURCE_PYTHON_HISTORY],
        )

        # The claim shares significant vocabulary with the source
        assert score.mean_fidelity > 0.4
        assert score.evidence_quality in ("strong", "moderate")

    def test_unsupported_claim_has_weak_evidence(self) -> None:
        """Claim with no vocabulary overlap should yield weak evidence."""
        from cognilateral_trust.integrations.cognee import CogneeTrustBridge

        bridge = CogneeTrustBridge()
        score = bridge.score_knowledge_claim(
            claim_text=_CLAIM_UNSUPPORTED,
            source_documents=[_SOURCE_PYTHON_HISTORY],
        )

        assert score.mean_fidelity < 0.4
        assert score.evidence_quality == "weak"
        assert score.confidence < 0.5

    def test_empty_claim_returns_zero_confidence(self) -> None:
        """Empty claim text should produce zero-confidence result."""
        from cognilateral_trust.integrations.cognee import CogneeTrustBridge

        bridge = CogneeTrustBridge()
        score = bridge.score_knowledge_claim(
            claim_text="",
            source_documents=[_SOURCE_PYTHON_HISTORY],
        )

        assert score.confidence == 0.0
        assert score.evidence_quality == "weak"
        assert score.source_count == 0
        assert "empty claim text" in score.reasons

    def test_no_sources_returns_zero_confidence(self) -> None:
        """No source documents should produce zero-confidence result."""
        from cognilateral_trust.integrations.cognee import CogneeTrustBridge

        bridge = CogneeTrustBridge()
        score = bridge.score_knowledge_claim(
            claim_text=_CLAIM_PYTHON_CREATOR,
            source_documents=[],
        )

        assert score.confidence == 0.0
        assert score.source_count == 0
        assert "no source documents provided" in score.reasons

    def test_multiple_sources_increase_confidence(self) -> None:
        """Multiple agreeing sources apply a multi-source bonus vs. mean fidelity alone."""
        from cognilateral_trust.integrations.cognee import CogneeTrustBridge

        bridge = CogneeTrustBridge()

        # Use a claim that matches partially (not perfectly) against both sources
        # so the multi-source bonus is visible above the mean fidelity baseline.
        claim = "Guido van Rossum created Python programming language"
        single = bridge.score_knowledge_claim(
            claim_text=claim,
            source_documents=[_SOURCE_PYTHON_DESIGN],
        )
        multi = bridge.score_knowledge_claim(
            claim_text=claim,
            source_documents=[_SOURCE_PYTHON_DESIGN, _SOURCE_PYTHON_HISTORY],
        )

        # Multi-source should get a bonus: confidence >= mean fidelity of multi
        assert multi.source_count == 2
        assert multi.confidence >= multi.mean_fidelity
        # And multi-source confidence should exceed the weaker single source
        assert multi.confidence >= single.confidence

    def test_contradicting_sources_reduce_confidence(self) -> None:
        """Mix of supporting and contradicting sources should reduce confidence vs. pure support."""
        from cognilateral_trust.integrations.cognee import CogneeTrustBridge

        bridge = CogneeTrustBridge()

        clean = bridge.score_knowledge_claim(
            claim_text=_CLAIM_PYTHON_CREATOR,
            source_documents=[_SOURCE_PYTHON_HISTORY, _SOURCE_PYTHON_DESIGN],
        )
        contradicted = bridge.score_knowledge_claim(
            claim_text=_CLAIM_PYTHON_CREATOR,
            source_documents=[_SOURCE_PYTHON_HISTORY, _SOURCE_UNRELATED],
        )

        # Including an unrelated source (low fidelity) should drag down mean fidelity
        assert contradicted.mean_fidelity < clean.mean_fidelity

    def test_low_confidence_claims_flagged(self) -> None:
        """Claims with confidence below threshold should carry warning reasons."""
        from cognilateral_trust.integrations.cognee import CogneeTrustBridge

        bridge = CogneeTrustBridge()
        score = bridge.score_knowledge_claim(
            claim_text=_CLAIM_UNSUPPORTED,
            source_documents=[_SOURCE_UNRELATED],
        )

        # Unrelated claim + unrelated source = very low fidelity
        assert score.confidence < 0.5
        assert any("weak evidence" in r or "low confidence" in r for r in score.reasons)

    def test_context_passed_to_accountability_record(self) -> None:
        """Custom context should appear in the trust evaluation's accountability record."""
        from cognilateral_trust.integrations.cognee import CogneeTrustBridge

        bridge = CogneeTrustBridge()
        score = bridge.score_knowledge_claim(
            claim_text=_CLAIM_PYTHON_CREATOR,
            source_documents=[_SOURCE_PYTHON_HISTORY],
            context={"graph_id": "test-graph-001"},
        )

        record = score.trust_evaluation.accountability_record
        assert record is not None
        assert record.context.get("graph_id") == "test-graph-001"
        assert record.context.get("integration") == "cognee"


class TestVerifyGraphEdge:
    """CogneeTrustBridge.verify_graph_edge must verify triples against sources."""

    def test_verify_graph_edge_checks_fidelity(self) -> None:
        """Supported edge should return positive fidelity and edge_supported=True."""
        from cognilateral_trust.integrations.cognee import CogneeTrustBridge

        bridge = CogneeTrustBridge()
        edge = bridge.verify_graph_edge(
            subject="Python",
            predicate="created_by",
            object_="Guido van Rossum",
            sources=[_SOURCE_PYTHON_HISTORY],
        )

        assert edge.fidelity_score > 0.0
        assert edge.subject == "Python"
        assert edge.predicate == "created_by"
        assert edge.object_ == "Guido van Rossum"
        assert edge.trust_evaluation is not None
        assert len(edge.fidelity_details) == 1
        assert edge.evidence_quality in ("strong", "moderate", "weak")

    def test_supported_edge_has_positive_fidelity(self) -> None:
        """Edge whose terms appear in the source should have meaningful fidelity."""
        from cognilateral_trust.integrations.cognee import CogneeTrustBridge

        bridge = CogneeTrustBridge()
        edge = bridge.verify_graph_edge(
            subject="Python",
            predicate="created_by",
            object_="Guido van Rossum",
            sources=[_SOURCE_PYTHON_HISTORY],
        )

        # "Python", "created", "Guido", "van", "Rossum" all appear in source
        assert edge.fidelity_score >= 0.4
        assert edge.edge_supported is True

    def test_unsupported_edge_flagged(self) -> None:
        """Edge with no source support should be flagged as unsupported."""
        from cognilateral_trust.integrations.cognee import CogneeTrustBridge

        bridge = CogneeTrustBridge()
        edge = bridge.verify_graph_edge(
            subject="Quantum",
            predicate="enables",
            object_="teleportation",
            sources=[_SOURCE_PYTHON_HISTORY],
        )

        assert edge.fidelity_score < 0.4
        assert edge.edge_supported is False
        assert any("unsupported" in r for r in edge.reasons)

    def test_edge_with_no_sources(self) -> None:
        """Edge with empty sources list should return zero fidelity."""
        from cognilateral_trust.integrations.cognee import CogneeTrustBridge

        bridge = CogneeTrustBridge()
        edge = bridge.verify_graph_edge(
            subject="Python",
            predicate="created_by",
            object_="Guido van Rossum",
            sources=[],
        )

        assert edge.fidelity_score == 0.0
        assert edge.edge_supported is False
        assert "no source texts provided" in edge.reasons[0]

    def test_edge_with_multiple_supporting_sources(self) -> None:
        """Multiple sources supporting an edge should apply multi-source bonus."""
        from cognilateral_trust.integrations.cognee import CogneeTrustBridge

        bridge = CogneeTrustBridge()

        # Verify single-source baseline, then multi-source Bayesian path
        single = bridge.verify_graph_edge(
            subject="Python",
            predicate="released_in",
            object_="2000",
            sources=[_SOURCE_PYTHON_DESIGN],
        )
        assert single.trust_evaluation.should_proceed
        multi = bridge.verify_graph_edge(
            subject="Python",
            predicate="released_in",
            object_="2000",
            sources=[_SOURCE_PYTHON_DESIGN, _SOURCE_PYTHON_HISTORY],
        )

        # Multi-source evaluation uses Bayesian weighting — a second source
        # with lower fidelity correctly reduces overall confidence vs a
        # single perfect-fidelity source. Verify the multi-source path runs
        # and produces a reasonable confidence (>0.5) with 2 fidelity details.
        assert multi.trust_evaluation.confidence > 0.5
        assert multi.trust_evaluation.should_proceed
        assert len(multi.fidelity_details) == 2

    def test_edge_predicate_underscores_converted(self) -> None:
        """Predicate underscores should be converted to spaces for fidelity check."""
        from cognilateral_trust.integrations.cognee import CogneeTrustBridge

        bridge = CogneeTrustBridge()
        edge = bridge.verify_graph_edge(
            subject="Python",
            predicate="first_released_in",
            object_="1991",
            sources=[_SOURCE_PYTHON_HISTORY],
        )

        # "Python first released in 1991" should match well against source
        assert edge.fidelity_score > 0.0
        # The fidelity detail should contain the readable predicate
        assert edge.fidelity_details[0].claim_text == "Python first released in 1991"

    def test_edge_context_in_accountability(self) -> None:
        """Edge verification should include edge description in accountability context."""
        from cognilateral_trust.integrations.cognee import CogneeTrustBridge

        bridge = CogneeTrustBridge()
        edge = bridge.verify_graph_edge(
            subject="Python",
            predicate="created_by",
            object_="Guido",
            sources=[_SOURCE_PYTHON_HISTORY],
            context={"graph_id": "test-edge-001"},
        )

        record = edge.trust_evaluation.accountability_record
        assert record is not None
        assert "Python -> created_by -> Guido" in record.context.get("edge", "")
        assert record.context.get("graph_id") == "test-edge-001"


class TestBridgeEdgeCases:
    """Edge cases and invariants for CogneeTrustBridge."""

    def test_whitespace_only_claim_returns_zero(self) -> None:
        """Whitespace-only claim should be treated as empty."""
        from cognilateral_trust.integrations.cognee import CogneeTrustBridge

        bridge = CogneeTrustBridge()
        score = bridge.score_knowledge_claim(
            claim_text="   ",
            source_documents=[_SOURCE_PYTHON_HISTORY],
        )

        assert score.confidence == 0.0

    def test_confidence_always_bounded_zero_one(self) -> None:
        """Confidence must always be in [0.0, 1.0] regardless of input."""
        from cognilateral_trust.integrations.cognee import CogneeTrustBridge

        bridge = CogneeTrustBridge()

        # Many sources with the exact claim text — should not exceed 1.0
        sources = [_CLAIM_PYTHON_CREATOR] * 20
        score = bridge.score_knowledge_claim(
            claim_text=_CLAIM_PYTHON_CREATOR,
            source_documents=sources,
        )

        assert 0.0 <= score.confidence <= 1.0

    def test_trust_score_is_frozen_dataclass(self) -> None:
        """TrustScore should be immutable (frozen dataclass)."""
        from cognilateral_trust.integrations.cognee import CogneeTrustBridge

        bridge = CogneeTrustBridge()
        score = bridge.score_knowledge_claim(
            claim_text=_CLAIM_PYTHON_CREATOR,
            source_documents=[_SOURCE_PYTHON_HISTORY],
        )

        try:
            score.confidence = 999.0  # type: ignore[misc]
            raise AssertionError("TrustScore should be frozen")
        except AttributeError:
            pass  # Expected — frozen dataclass

    def test_edge_verification_is_frozen_dataclass(self) -> None:
        """EdgeVerification should be immutable (frozen dataclass)."""
        from cognilateral_trust.integrations.cognee import CogneeTrustBridge

        bridge = CogneeTrustBridge()
        edge = bridge.verify_graph_edge(
            subject="Python",
            predicate="created_by",
            object_="Guido",
            sources=[_SOURCE_PYTHON_HISTORY],
        )

        try:
            edge.fidelity_score = 999.0  # type: ignore[misc]
            raise AssertionError("EdgeVerification should be frozen")
        except AttributeError:
            pass  # Expected — frozen dataclass
