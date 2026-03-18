"""Cognee trust layer — confidence gates for knowledge graph operations.

Cognee builds knowledge graphs from text and lets agents search and reason
over them. cognilateral-trust gates two critical operations:

1. **Ingestion** — before adding facts to the graph, verify confidence is
   high enough that you won't pollute the knowledge base.

2. **Retrieval** — before acting on retrieved knowledge, verify the source
   confidence warrants the intended action.

Usage:
    pip install cognilateral-trust cognee

    from cognee_trust_layer import TrustedCognee

    tc = TrustedCognee()

    # Only adds fact to graph if confidence >= C4
    result = await tc.add(
        text="The deploy pipeline requires two approvals",
        confidence=0.85,
    )

    # Retrieves and gates before returning
    facts = await tc.search("deploy pipeline", min_confidence=0.7)
    for fact in facts["trusted"]:
        print(fact["text"])  # Only facts that clear the trust bar

This is an example integration. Adapt to your cognee configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cognilateral_trust import evaluate_trust
from cognilateral_trust.core import ConfidenceTier

# cognee is optional — example runs standalone without it
try:
    import cognee

    HAS_COGNEE = True
except ImportError:
    HAS_COGNEE = False


# Minimum tier required before writing to the knowledge graph.
# C4 (Tested) is the floor — unverified facts pollute the graph.
_DEFAULT_INGEST_TIER = ConfidenceTier.C4

# Minimum tier for acting on retrieved facts.
# C3 (Measured) — we accept lower confidence for reads than writes.
_DEFAULT_RETRIEVAL_TIER = ConfidenceTier.C3


@dataclass
class IngestionResult:
    """Result of a trust-gated knowledge ingestion attempt."""

    text: str
    confidence: float
    tier: str
    route: str
    ingested: bool
    reasons: list[str] = field(default_factory=list)
    record_id: str = ""


@dataclass
class RetrievalResult:
    """Trust-scored retrieval result from the knowledge graph."""

    text: str
    confidence: float
    tier: str
    route: str
    trusted: bool
    reasons: list[str] = field(default_factory=list)
    record_id: str = ""


def _source_confidence(result: dict[str, Any]) -> float:
    """Extract confidence from a Cognee search result.

    Cognee returns results with varying metadata depending on version.
    We prefer explicit confidence scores, then fall back to similarity.
    """
    if "confidence" in result:
        return float(result["confidence"])
    if "score" in result:
        # Normalize similarity score (typically 0-1, sometimes higher)
        return min(1.0, float(result["score"]))
    # Unknown provenance — treat as low confidence
    return 0.3


def evaluate_ingestion(
    text: str,
    confidence: float,
    min_tier: ConfidenceTier = _DEFAULT_INGEST_TIER,
    context: dict[str, Any] | None = None,
) -> IngestionResult:
    """Evaluate whether a fact should be ingested into the knowledge graph.

    Knowledge graphs are hard to audit once polluted. The threshold is
    intentionally strict: C4 (Tested) minimum before any fact enters the graph.

    Args:
        text: The text/fact to evaluate.
        confidence: Source confidence in this fact (0.0-1.0).
        min_tier: Minimum confidence tier to permit ingestion. Default: C4.
        context: Optional metadata to attach to the accountability record.

    Returns:
        IngestionResult with verdict and accountability record.
    """
    result = evaluate_trust(
        confidence,
        is_reversible=True,  # Facts can be retracted; the tier floor is the primary gate
        touches_external=False,  # Trust engine context: evaluate confidence quality, not action risk
        context={
            **(context or {}),
            "operation": "cognee_ingest",
            "text_preview": text[:100],
        },
    )

    # Enforce minimum tier — even if trust engine says "proceed",
    # we refuse if the tier is below the ingestion floor.
    tier_value = result.tier.value if hasattr(result.tier, "value") else int(result.tier.name[1:])
    min_tier_value = min_tier.value if hasattr(min_tier, "value") else int(min_tier.name[1:])
    meets_floor = tier_value >= min_tier_value

    ingested = result.should_proceed and meets_floor

    reasons = list(result.accountability_record.reasons) if result.accountability_record else []
    if not meets_floor:
        reasons.append(f"tier {result.tier.name} below ingestion floor {min_tier.name}")

    return IngestionResult(
        text=text,
        confidence=confidence,
        tier=result.tier.name,
        route=result.route,
        ingested=ingested,
        reasons=reasons,
        record_id=result.accountability_record.record_id if result.accountability_record else "",
    )


def evaluate_retrieval(
    result: dict[str, Any],
    intended_action: str = "use",
    is_reversible: bool = True,
    min_tier: ConfidenceTier = _DEFAULT_RETRIEVAL_TIER,
    context: dict[str, Any] | None = None,
) -> RetrievalResult:
    """Evaluate whether a retrieved fact can be acted upon.

    Args:
        result: A Cognee search result dict.
        intended_action: Description of how you plan to use the fact.
        is_reversible: Whether the action using this fact is reversible.
        min_tier: Minimum tier to proceed. Default: C3.
        context: Optional metadata.

    Returns:
        RetrievalResult with trust verdict.
    """
    text = result.get("text", result.get("content", str(result)))
    confidence = _source_confidence(result)

    trust = evaluate_trust(
        confidence,
        is_reversible=is_reversible,
        context={
            **(context or {}),
            "operation": "cognee_retrieval",
            "intended_action": intended_action,
            "text_preview": text[:100],
        },
    )

    tier_value = trust.tier.value if hasattr(trust.tier, "value") else int(trust.tier.name[1:])
    min_tier_value = min_tier.value if hasattr(min_tier, "value") else int(min_tier.name[1:])
    meets_floor = tier_value >= min_tier_value

    trusted = trust.should_proceed and meets_floor

    reasons = list(trust.accountability_record.reasons) if trust.accountability_record else []
    if not meets_floor:
        reasons.append(f"retrieved fact tier {trust.tier.name} below required {min_tier.name}")

    return RetrievalResult(
        text=text,
        confidence=confidence,
        tier=trust.tier.name,
        route=trust.route,
        trusted=trusted,
        reasons=reasons,
        record_id=trust.accountability_record.record_id if trust.accountability_record else "",
    )


class TrustedCognee:
    """Cognee wrapper with trust gates on ingest and retrieval.

    All knowledge flowing through TrustedCognee is evaluated for confidence
    before entering or leaving the graph. Low-confidence facts are blocked at
    ingestion. Low-confidence retrievals are quarantined, not silently dropped.

    Works without Cognee installed (standalone mode for testing).
    """

    def __init__(
        self,
        ingest_min_tier: ConfidenceTier = _DEFAULT_INGEST_TIER,
        retrieval_min_tier: ConfidenceTier = _DEFAULT_RETRIEVAL_TIER,
    ) -> None:
        self.ingest_min_tier = ingest_min_tier
        self.retrieval_min_tier = retrieval_min_tier
        self._blocked_count = 0
        self._ingested_count = 0

    async def add(
        self,
        text: str,
        confidence: float,
        dataset_name: str = "default",
        context: dict[str, Any] | None = None,
    ) -> IngestionResult:
        """Add a fact to the knowledge graph, gated by trust evaluation.

        Facts below the ingestion floor are rejected with a full accountability
        record — you can audit why each fact was blocked.

        Args:
            text: The fact to add.
            confidence: Your confidence in this fact (0.0-1.0).
            dataset_name: Cognee dataset to write to.
            context: Optional metadata for the accountability record.

        Returns:
            IngestionResult. Check `.ingested` to see if it was accepted.
        """
        assessment = evaluate_ingestion(text, confidence, self.ingest_min_tier, context)

        if not assessment.ingested:
            self._blocked_count += 1
            return assessment

        if HAS_COGNEE:
            await cognee.add(text, dataset_name=dataset_name)
            await cognee.cognify()

        self._ingested_count += 1
        return assessment

    async def search(
        self,
        query: str,
        intended_action: str = "use",
        is_reversible: bool = True,
        min_confidence: float | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, list[RetrievalResult]]:
        """Search the knowledge graph with trust evaluation on results.

        Returns results split into two lists:
        - `trusted`: facts that clear the trust bar for `intended_action`
        - `quarantined`: facts returned by Cognee but below the trust threshold

        Quarantined facts are visible — the caller decides whether to
        surface them to a human for review, discard them, or lower the bar.

        Args:
            query: The search query.
            intended_action: How you plan to use the results.
            is_reversible: Whether the intended action is reversible.
            min_confidence: Override minimum confidence (0.0-1.0).
            context: Optional metadata.

        Returns:
            Dict with "trusted" and "quarantined" lists of RetrievalResult.
        """
        if min_confidence is not None:
            # Compute effective min_tier from min_confidence
            tier_index = min(9, max(0, int(min_confidence * 10)))
            try:
                effective_min_tier = ConfidenceTier[f"C{tier_index}"]
            except KeyError:
                effective_min_tier = self.retrieval_min_tier
        else:
            effective_min_tier = self.retrieval_min_tier

        raw_results: list[dict[str, Any]] = []
        if HAS_COGNEE:
            raw_results = await cognee.search(query, type="insights")
        else:
            # Standalone demo: simulate two results with different confidence
            raw_results = [
                {"text": f"[simulated] {query} — high confidence result", "score": 0.87},
                {"text": f"[simulated] {query} — low confidence result", "score": 0.25},
            ]

        trusted: list[RetrievalResult] = []
        quarantined: list[RetrievalResult] = []

        for raw in raw_results:
            evaluation = evaluate_retrieval(
                raw,
                intended_action=intended_action,
                is_reversible=is_reversible,
                min_tier=effective_min_tier,
                context=context,
            )
            if evaluation.trusted:
                trusted.append(evaluation)
            else:
                quarantined.append(evaluation)

        return {"trusted": trusted, "quarantined": quarantined}

    @property
    def stats(self) -> dict[str, int]:
        """Return ingestion statistics."""
        total = self._ingested_count + self._blocked_count
        return {
            "ingested": self._ingested_count,
            "blocked": self._blocked_count,
            "total_attempted": total,
        }


# --- Standalone demo ---


async def _demo() -> None:
    """Run without cognee installed to show the trust layer in action."""

    tc = TrustedCognee()

    print("=== Ingestion gate ===")
    facts = [
        ("Our API rate limit is 100 req/s", 0.95),  # C9 — high confidence
        ("Deploys usually take 10 minutes", 0.65),  # C6 — mid confidence
        ("I think the schema might have changed", 0.3),  # C3 — below floor
    ]

    for text, conf in facts:
        result = await tc.add(text, confidence=conf, context={"source": "demo"})
        status = "INGESTED" if result.ingested else "BLOCKED"
        print(f"  [{status}] tier={result.tier} conf={result.confidence:.2f} | {text[:50]}")
        if result.reasons:
            print(f"           reason: {result.reasons[0]}")

    print(f"\n  Stats: {tc.stats}")

    print("\n=== Retrieval gate — reversible action (read/display) ===")
    results = await tc.search("rate limit", intended_action="display-in-ui", is_reversible=True)

    print(f"  Trusted ({len(results['trusted'])}):")
    for r in results["trusted"]:
        print(f"    tier={r.tier} conf={r.confidence:.2f} | {r.text[:60]}")

    print(f"  Quarantined ({len(results['quarantined'])}):")
    for r in results["quarantined"]:
        print(f"    tier={r.tier} conf={r.confidence:.2f} | {r.text[:60]}")
        print(f"      reason: {r.reasons[0] if r.reasons else 'unknown'}")

    print("\n=== Retrieval gate — irreversible action (auto-scale infra) ===")
    results2 = await tc.search("rate limit", intended_action="auto-scale", is_reversible=False)

    print(f"  Trusted ({len(results2['trusted'])}):")
    for r in results2["trusted"]:
        print(f"    tier={r.tier} conf={r.confidence:.2f} | {r.text[:60]}")

    print(f"  Quarantined ({len(results2['quarantined'])}):")
    for r in results2["quarantined"]:
        print(f"    tier={r.tier} conf={r.confidence:.2f} | {r.text[:60]}")
        print(f"      reason: {r.reasons[0] if r.reasons else 'unknown'}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(_demo())
