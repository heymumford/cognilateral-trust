"""Prediction store — tracks confidence predictions and outcomes for calibration.

The calibration accuracy metric measures: when we say 90% confident, are we
right 90% of the time? This requires storing (prediction, outcome) pairs.

Thread-safe, bounded, with rolling window support.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PredictionRecord:
    """A single prediction with its outcome."""

    record_id: str
    timestamp: float
    predicted_confidence: float
    outcome: Literal["correct", "incorrect", "pending"]
    context: str


class PredictionStore:
    """Thread-safe bounded store for calibration tracking."""

    def __init__(self, max_records: int = 10000) -> None:
        self._lock = threading.Lock()
        self._records: list[PredictionRecord] = []
        self._max = max_records

    def record_prediction(
        self,
        record_id: str,
        predicted_confidence: float,
        context: str = "",
    ) -> PredictionRecord:
        """Record a new prediction (outcome pending)."""
        rec = PredictionRecord(
            record_id=record_id,
            timestamp=time.time(),
            predicted_confidence=predicted_confidence,
            outcome="pending",
            context=context,
        )
        with self._lock:
            self._records.append(rec)
            if len(self._records) > self._max:
                self._records = self._records[-self._max :]
        return rec

    def record_outcome(self, record_id: str, correct: bool) -> PredictionRecord | None:
        """Update a prediction with its actual outcome."""
        outcome: Literal["correct", "incorrect"] = "correct" if correct else "incorrect"
        with self._lock:
            for i, rec in enumerate(self._records):
                if rec.record_id == record_id and rec.outcome == "pending":
                    updated = PredictionRecord(
                        record_id=rec.record_id,
                        timestamp=rec.timestamp,
                        predicted_confidence=rec.predicted_confidence,
                        outcome=outcome,
                        context=rec.context,
                    )
                    self._records[i] = updated
                    return updated
        return None

    def get_resolved(self, window_seconds: float | None = None) -> list[PredictionRecord]:
        """Get all resolved (non-pending) predictions, optionally within a time window."""
        now = time.time()
        with self._lock:
            results = [r for r in self._records if r.outcome != "pending"]
            if window_seconds is not None:
                cutoff = now - window_seconds
                results = [r for r in results if r.timestamp >= cutoff]
            return results

    def calibration_accuracy(self, n_bins: int = 5, window_seconds: float | None = None) -> float:
        """Compute calibration accuracy: 1.0 - mean absolute error across bins.

        Groups predictions into bins, computes actual success rate per bin,
        measures deviation from predicted confidence. Returns 0.0 if no data.
        """
        resolved = self.get_resolved(window_seconds)
        if not resolved:
            return 0.0

        bin_predicted: list[float] = [0.0] * n_bins
        bin_actual: list[float] = [0.0] * n_bins
        bin_counts: list[int] = [0] * n_bins

        for rec in resolved:
            idx = min(int(rec.predicted_confidence * n_bins), n_bins - 1)
            bin_predicted[idx] += rec.predicted_confidence
            bin_actual[idx] += 1.0 if rec.outcome == "correct" else 0.0
            bin_counts[idx] += 1

        total_error = 0.0
        populated = 0
        for i in range(n_bins):
            if bin_counts[i] == 0:
                continue
            avg_pred = bin_predicted[i] / bin_counts[i]
            actual_rate = bin_actual[i] / bin_counts[i]
            total_error += abs(avg_pred - actual_rate)
            populated += 1

        if populated == 0:
            return 0.0
        return max(0.0, 1.0 - total_error / populated)

    @property
    def total(self) -> int:
        with self._lock:
            return len(self._records)

    @property
    def pending(self) -> int:
        with self._lock:
            return sum(1 for r in self._records if r.outcome == "pending")

    @property
    def resolved(self) -> int:
        with self._lock:
            return sum(1 for r in self._records if r.outcome != "pending")
