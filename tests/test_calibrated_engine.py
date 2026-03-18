"""Tests for CalibratedTrustEngine."""

from __future__ import annotations

from cognilateral_trust.calibrated import CalibratedTrustEngine


class TestCalibratedTrustEngine:
    def test_evaluate_tracks_prediction(self) -> None:
        engine = CalibratedTrustEngine()
        result = engine.evaluate(0.8)
        assert engine.stats["total"] == 1
        assert engine.stats["pending"] == 1

    def test_record_outcome_resolves(self) -> None:
        engine = CalibratedTrustEngine()
        result = engine.evaluate(0.8)
        rid = result.accountability_record.record_id
        assert engine.record_outcome(rid, correct=True)
        assert engine.stats["resolved"] == 1
        assert engine.stats["pending"] == 0

    def test_accuracy_improves(self) -> None:
        engine = CalibratedTrustEngine()
        # 10 decisions at 0.8 confidence, 8 correct = well-calibrated
        for i in range(10):
            r = engine.evaluate(0.8, context=f"decision-{i}")
            engine.record_outcome(r.accountability_record.record_id, correct=i < 8)
        assert engine.accuracy > 0.8

    def test_overconfidence_detected(self) -> None:
        engine = CalibratedTrustEngine()
        # Says 0.9 but only 50% correct = overconfident
        for i in range(20):
            r = engine.evaluate(0.9)
            engine.record_outcome(r.accountability_record.record_id, correct=i < 10)
        assert engine.accuracy < 0.7  # poor calibration

    def test_stats_structure(self) -> None:
        engine = CalibratedTrustEngine()
        s = engine.stats
        assert "total" in s
        assert "pending" in s
        assert "resolved" in s
        assert "accuracy" in s

    def test_unfound_outcome_returns_false(self) -> None:
        engine = CalibratedTrustEngine()
        assert engine.record_outcome("nonexistent", correct=True) is False
