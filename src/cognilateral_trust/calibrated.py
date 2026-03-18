"""Calibrated trust — evaluation + prediction tracking in one call.

The simplest way to use cognilateral-trust with calibration:

    engine = CalibratedTrustEngine()
    result = engine.evaluate(0.8, context="deploy")
    # ... agent acts ...
    engine.record_outcome(result.record_id, correct=True)
    print(engine.accuracy)  # improves over time

With persistence (survives restarts):

    engine = CalibratedTrustEngine(persist_path="/tmp/my-agent/trust")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cognilateral_trust.accountability import AccountabilityRecord, AccountabilityStore
from cognilateral_trust.evaluate import TrustEvaluation, evaluate_trust
from cognilateral_trust.prediction_store import PredictionStore


class CalibratedTrustEngine:
    """Trust engine with built-in calibration tracking.

    Wraps evaluate_trust() + PredictionStore + AccountabilityStore into a
    single interface. Every evaluation is automatically tracked. Feed outcomes
    back to build calibration accuracy over time.

    Args:
        max_predictions: Maximum predictions to retain in memory.
        persist_path:    Directory path for JSONL persistence. If None (default),
                         everything is in-memory only. If provided, prediction
                         records and accountability records survive restarts.
                         Accepts Path or str.
    """

    def __init__(
        self,
        max_predictions: int = 10000,
        *,
        persist_path: Path | str | None = None,
    ) -> None:
        if persist_path is not None:
            from cognilateral_trust.persistence import (
                JSONLAccountabilityStore,
                JSONLPredictionStore,
            )

            persist = Path(persist_path)
            self._store: PredictionStore = JSONLPredictionStore(
                persist / "predictions.jsonl",
                max_records=max_predictions,
            )
            self._accountability: AccountabilityStore = JSONLAccountabilityStore(
                persist / "accountability.jsonl",
                max_records=max_predictions,
            )
        else:
            self._store = PredictionStore(max_records=max_predictions)
            self._accountability = AccountabilityStore(max_records=max_predictions)

    def evaluate(
        self,
        confidence: float,
        *,
        is_reversible: bool = True,
        touches_external: bool = False,
        context: str = "",
    ) -> TrustEvaluation:
        """Evaluate trust and auto-track the prediction."""
        result = evaluate_trust(
            confidence,
            is_reversible=is_reversible,
            touches_external=touches_external,
            context={"calibrated": True, "description": context},
        )
        if result.accountability_record:
            self._store.record_prediction(
                result.accountability_record.record_id,
                confidence,
                context=context,
            )
            self._accountability.append(result.accountability_record)
        return result

    def record_outcome(self, record_id: str, correct: bool) -> bool:
        """Record whether a prediction was correct. Returns True if found."""
        return self._store.record_outcome(record_id, correct) is not None

    def get_record(self, record_id: str) -> AccountabilityRecord | None:
        """Retrieve an accountability record by ID."""
        return self._accountability.get(record_id)

    def calibration_accuracy(self, n_bins: int = 5) -> float:
        """Current calibration accuracy (0.0-1.0)."""
        return self._store.calibration_accuracy(n_bins=n_bins)

    @property
    def accuracy(self) -> float:
        """Current calibration accuracy (0.0-1.0)."""
        return self._store.calibration_accuracy()

    @property
    def stats(self) -> dict[str, Any]:
        """Summary statistics."""
        return {
            "total": self._store.total,
            "pending": self._store.pending,
            "resolved": self._store.resolved,
            "accuracy": round(self.accuracy, 4),
        }
