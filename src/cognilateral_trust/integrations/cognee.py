"""Cognee Knowledge Graph integration — trust scoring for knowledge claims.

Cognee builds knowledge graphs from documents. Cognilateral adds epistemic
trust scoring to claims extracted from those graphs. Together: verified
knowledge with confidence.

Usage:
    from cognilateral_trust.integrations.cognee import CogneeTrustBridge

    bridge = CogneeTrustBridge()

    # Score a knowledge claim against its source documents
    score = bridge.score_knowledge_claim(
        claim_text="Python was created by Guido van Rossum in 1991.",
        source_documents=["Python is a programming language created by Guido van Rossum..."],
    )
    print(score.confidence)  # 0.0-1.0
    print(score.tier.name)   # e.g., "C7"
    print(score.evidence_quality)  # e.g., "strong"

    # Verify a knowledge graph edge (subject-predicate-object triple)
    edge = bridge.verify_graph_edge(
        subject="Python",
        predicate="created_by",
        object_="Guido van Rossum",
        sources=["Python was created by Guido van Rossum in 1991."],
    )
    print(edge.fidelity_score)      # 0.0-1.0
    print(edge.edge_supported)      # True/False
    print(edge.trust_evaluation)    # TrustEvaluation

Zero dependencies beyond cognilateral-trust itself. Cognee is only
needed at runtime when consuming these results in a Cognee pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cognilateral_trust.core import ConfidenceTier
from cognilateral_trust.evaluate import TrustEvaluation, evaluate_trust
from cognilateral_trust.fidelity import FidelityResult, verify_fidelity

__all__ = [
    "CogneeTrustBridge",
    "EdgeVerification",
    "TrustScore",
]

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_FIDELITY_STRONG_THRESHOLD = 0.7
_FIDELITY_MODERATE_THRESHOLD = 0.4
_LOW_CONFIDENCE_THRESHOLD = 0.3
_CONTRADICTION_PENALTY = 0.15
_MULTI_SOURCE_BONUS_PER_SOURCE = 0.05
_MULTI_SOURCE_BONUS_CAP = 0.2
_EDGE_SUPPORT_THRESHOLD = 0.4


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrustScore:
    """Trust evaluation result for a knowledge claim from a Cognee graph.

    Designed for display in Cognee's UI: confidence gauge, tier badge,
    evidence quality indicator, and supporting fidelity detail.
    """

    confidence: float
    tier: ConfidenceTier
    tier_name: str
    route: str
    evidence_quality: str
    should_proceed: bool
    fidelity_scores: tuple[float, ...]
    mean_fidelity: float
    source_count: int
    claim_text: str
    reasons: tuple[str, ...]
    trust_evaluation: TrustEvaluation


@dataclass(frozen=True)
class EdgeVerification:
    """Verification result for a single knowledge graph edge (triple).

    subject --predicate--> object, verified against source texts.
    """

    subject: str
    predicate: str
    object_: str
    fidelity_score: float
    edge_supported: bool
    trust_evaluation: TrustEvaluation
    fidelity_details: tuple[FidelityResult, ...]
    evidence_quality: str
    reasons: tuple[str, ...]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _classify_evidence_quality(mean_fidelity: float) -> str:
    """Classify evidence quality from mean fidelity score."""
    if mean_fidelity >= _FIDELITY_STRONG_THRESHOLD:
        return "strong"
    if mean_fidelity >= _FIDELITY_MODERATE_THRESHOLD:
        return "moderate"
    return "weak"


def _compute_source_agreement(fidelity_scores: list[float]) -> float:
    """Compute agreement factor across sources.

    When sources agree (all high or all low), agreement is high.
    When sources contradict (mix of high and low), agreement drops.
    Returns a value in [0.0, 1.0].
    """
    if len(fidelity_scores) <= 1:
        return 1.0
    mean = sum(fidelity_scores) / len(fidelity_scores)
    variance = sum((s - mean) ** 2 for s in fidelity_scores) / len(fidelity_scores)
    # Normalize variance: max possible variance for [0,1] scores is 0.25
    # High variance = low agreement
    return max(0.0, 1.0 - (variance / 0.25))


def _compute_confidence(
    mean_fidelity: float,
    source_count: int,
    agreement: float,
) -> float:
    """Compute overall confidence from fidelity, source count, and agreement.

    Base confidence is the mean fidelity score, boosted by multiple
    agreeing sources and penalized by disagreement.
    """
    base = mean_fidelity

    # Multi-source bonus: each additional source adds a small bonus, capped
    extra_sources = max(0, source_count - 1)
    multi_source_bonus = min(
        extra_sources * _MULTI_SOURCE_BONUS_PER_SOURCE,
        _MULTI_SOURCE_BONUS_CAP,
    )
    base += multi_source_bonus

    # Agreement penalty: low agreement (contradicting sources) reduces confidence
    if agreement < 0.7:
        penalty = _CONTRADICTION_PENALTY * (1.0 - agreement)
        base -= penalty

    return max(0.0, min(1.0, round(base, 4)))


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------


class CogneeTrustBridge:
    """Bridge between Cognee knowledge graphs and Cognilateral trust evaluation.

    Scores knowledge claims and verifies graph edges using fidelity
    verification and trust evaluation. Stateless and thread-safe.
    """

    def score_knowledge_claim(
        self,
        claim_text: str,
        source_documents: list[str],
        *,
        context: dict[str, Any] | None = None,
    ) -> TrustScore:
        """Score a knowledge claim against its source documents.

        Verifies fidelity of the claim against each source document,
        computes aggregate confidence, and evaluates trust tier routing.

        Args:
            claim_text: The knowledge claim text to evaluate.
            source_documents: Source document texts that should support the claim.
            context: Optional context for the accountability record.

        Returns:
            TrustScore with confidence, tier, evidence quality, and fidelity detail.
        """
        if not claim_text or not claim_text.strip():
            return self._empty_trust_score(claim_text, "empty claim text")

        if not source_documents:
            return self._empty_trust_score(claim_text, "no source documents provided")

        # Verify fidelity against each source
        fidelity_results = [verify_fidelity(claim_text, doc) for doc in source_documents]
        fidelity_scores = [r.score for r in fidelity_results]
        mean_fidelity = sum(fidelity_scores) / len(fidelity_scores)
        agreement = _compute_source_agreement(fidelity_scores)

        confidence = _compute_confidence(
            mean_fidelity=mean_fidelity,
            source_count=len(source_documents),
            agreement=agreement,
        )

        evidence_quality = _classify_evidence_quality(mean_fidelity)

        # Build reasons
        reasons: list[str] = []
        if confidence < _LOW_CONFIDENCE_THRESHOLD:
            reasons.append(f"low confidence ({confidence:.2f}) — claim may be unsupported")
        if evidence_quality == "weak":
            reasons.append("weak evidence — low fidelity across source documents")
        if agreement < 0.7:
            reasons.append(f"source disagreement (agreement={agreement:.2f})")
        if len(source_documents) > 1 and evidence_quality == "strong":
            reasons.append(f"corroborated across {len(source_documents)} sources")

        eval_context = {
            "integration": "cognee",
            "claim_text": claim_text[:200],
            "source_count": len(source_documents),
            "mean_fidelity": round(mean_fidelity, 4),
            "agreement": round(agreement, 4),
            **(context or {}),
        }

        trust_eval = evaluate_trust(
            confidence,
            is_reversible=True,
            touches_external=False,
            context=eval_context,
        )

        return TrustScore(
            confidence=confidence,
            tier=trust_eval.tier,
            tier_name=trust_eval.tier.name,
            route=trust_eval.route,
            evidence_quality=evidence_quality,
            should_proceed=trust_eval.should_proceed,
            fidelity_scores=tuple(fidelity_scores),
            mean_fidelity=round(mean_fidelity, 4),
            source_count=len(source_documents),
            claim_text=claim_text,
            reasons=tuple(reasons),
            trust_evaluation=trust_eval,
        )

    def verify_graph_edge(
        self,
        subject: str,
        predicate: str,
        object_: str,
        sources: list[str],
        *,
        context: dict[str, Any] | None = None,
    ) -> EdgeVerification:
        """Verify a knowledge graph edge (subject-predicate-object triple).

        Reconstructs the edge as a natural-language claim, then verifies
        fidelity against each source text and evaluates trust.

        Args:
            subject: The subject entity of the triple.
            predicate: The predicate/relationship of the triple.
            object_: The object entity of the triple.
            sources: Source texts that should support this edge.
            context: Optional context for the accountability record.

        Returns:
            EdgeVerification with fidelity score, support verdict, and trust detail.
        """
        # Reconstruct a natural-language claim from the triple
        predicate_readable = predicate.replace("_", " ")
        claim_text = f"{subject} {predicate_readable} {object_}"

        if not sources:
            fidelity_details: tuple[FidelityResult, ...] = ()
            fidelity_score = 0.0
            evidence_quality = "weak"
            reasons = ("no source texts provided for edge verification",)
            trust_eval = evaluate_trust(
                0.0,
                is_reversible=True,
                touches_external=False,
                context={
                    "integration": "cognee",
                    "edge": f"{subject} -> {predicate} -> {object_}",
                    **(context or {}),
                },
            )
            return EdgeVerification(
                subject=subject,
                predicate=predicate,
                object_=object_,
                fidelity_score=fidelity_score,
                edge_supported=False,
                trust_evaluation=trust_eval,
                fidelity_details=fidelity_details,
                evidence_quality=evidence_quality,
                reasons=reasons,
            )

        # Verify fidelity of the reconstructed claim against each source
        fidelity_results = [verify_fidelity(claim_text, src) for src in sources]
        fidelity_scores = [r.score for r in fidelity_results]

        # Edge fidelity: use the best source match (max) since any one
        # source supporting the edge is sufficient
        fidelity_score = max(fidelity_scores)
        mean_fidelity = sum(fidelity_scores) / len(fidelity_scores)

        edge_supported = fidelity_score >= _EDGE_SUPPORT_THRESHOLD
        evidence_quality = _classify_evidence_quality(mean_fidelity)

        # Confidence for the edge: best fidelity with multi-source consideration
        agreement = _compute_source_agreement(fidelity_scores)
        confidence = _compute_confidence(
            mean_fidelity=mean_fidelity,
            source_count=len(sources),
            agreement=agreement,
        )

        reasons_list: list[str] = []
        if not edge_supported:
            reasons_list.append(
                f"edge unsupported — best fidelity {fidelity_score:.2f} below threshold {_EDGE_SUPPORT_THRESHOLD}"
            )
        if edge_supported and fidelity_score >= _FIDELITY_STRONG_THRESHOLD:
            reasons_list.append("edge strongly supported by source evidence")
        elif edge_supported:
            reasons_list.append("edge moderately supported by source evidence")
        if agreement < 0.7:
            reasons_list.append(f"sources show disagreement (agreement={agreement:.2f})")

        eval_context = {
            "integration": "cognee",
            "edge": f"{subject} -> {predicate} -> {object_}",
            "best_fidelity": round(fidelity_score, 4),
            "source_count": len(sources),
            **(context or {}),
        }

        trust_eval = evaluate_trust(
            confidence,
            is_reversible=True,
            touches_external=False,
            context=eval_context,
        )

        return EdgeVerification(
            subject=subject,
            predicate=predicate,
            object_=object_,
            fidelity_score=round(fidelity_score, 4),
            edge_supported=edge_supported,
            trust_evaluation=trust_eval,
            fidelity_details=tuple(fidelity_results),
            evidence_quality=evidence_quality,
            reasons=tuple(reasons_list),
        )

    def _empty_trust_score(self, claim_text: str, reason: str) -> TrustScore:
        """Return a zero-confidence TrustScore for degenerate inputs."""
        trust_eval = evaluate_trust(
            0.0,
            is_reversible=True,
            touches_external=False,
            context={"integration": "cognee", "reason": reason},
        )
        return TrustScore(
            confidence=0.0,
            tier=trust_eval.tier,
            tier_name=trust_eval.tier.name,
            route=trust_eval.route,
            evidence_quality="weak",
            should_proceed=trust_eval.should_proceed,
            fidelity_scores=(),
            mean_fidelity=0.0,
            source_count=0,
            claim_text=claim_text or "",
            reasons=(reason,),
            trust_evaluation=trust_eval,
        )
