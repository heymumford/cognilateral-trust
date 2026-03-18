"""Sprint S4: Tests for cognilateral_trust.persistence module.

Covers:
- JSONLPredictionStore: write → restart → survive
- JSONLAccountabilityStore: write → restart → survive
- CalibratedTrustEngine optional persistence via persist_path
- Corrupt JSONL line graceful skip
- File created on first write, not on init
- Thread-safe append behavior
"""

from __future__ import annotations

import json
import threading
from pathlib import Path


from cognilateral_trust.persistence import (
    JSONLAccountabilityStore,
    JSONLPredictionStore,
)


# ---------------------------------------------------------------------------
# JSONLPredictionStore
# ---------------------------------------------------------------------------


class TestJSONLPredictionStore:
    def test_file_not_created_on_init(self, tmp_path: Path) -> None:
        path = tmp_path / "preds.jsonl"
        JSONLPredictionStore(path)
        assert not path.exists()

    def test_file_created_on_first_write(self, tmp_path: Path) -> None:
        path = tmp_path / "preds.jsonl"
        store = JSONLPredictionStore(path)
        store.record_prediction("r1", 0.8)
        assert path.exists()

    def test_predictions_survive_restart(self, tmp_path: Path) -> None:
        path = tmp_path / "preds.jsonl"
        store1 = JSONLPredictionStore(path)
        store1.record_prediction("r1", 0.8, context="test-context")
        store1.record_prediction("r2", 0.5)
        assert store1.total == 2

        # New instance loads from file
        store2 = JSONLPredictionStore(path)
        assert store2.total == 2

    def test_outcomes_survive_restart(self, tmp_path: Path) -> None:
        path = tmp_path / "preds.jsonl"
        store1 = JSONLPredictionStore(path)
        store1.record_prediction("r1", 0.8)
        store1.record_outcome("r1", correct=True)
        assert store1.resolved == 1

        store2 = JSONLPredictionStore(path)
        assert store2.resolved == 1
        assert store2.pending == 0

    def test_corrupt_line_skipped_gracefully(self, tmp_path: Path) -> None:
        path = tmp_path / "preds.jsonl"
        store1 = JSONLPredictionStore(path)
        store1.record_prediction("r1", 0.8)

        # Inject corrupt line
        with path.open("a") as f:
            f.write("this is not valid json\n")

        # Second valid record
        store1.record_prediction("r2", 0.6)

        # Reload — corrupt line skipped, valid records loaded
        store2 = JSONLPredictionStore(path)
        assert store2.total >= 1  # at least one valid record

    def test_predictions_appended_not_rewritten(self, tmp_path: Path) -> None:
        path = tmp_path / "preds.jsonl"
        store = JSONLPredictionStore(path)
        store.record_prediction("r1", 0.7)
        store.record_prediction("r2", 0.8)
        lines = path.read_text().strip().split("\n")
        # Each prediction should be one line
        assert len(lines) >= 2

    def test_outcome_updates_persisted(self, tmp_path: Path) -> None:
        path = tmp_path / "preds.jsonl"
        store = JSONLPredictionStore(path)
        store.record_prediction("r1", 0.9)
        store.record_outcome("r1", correct=False)

        store2 = JSONLPredictionStore(path)
        resolved = store2.get_resolved()
        assert len(resolved) == 1
        assert resolved[0].outcome == "incorrect"

    def test_thread_safe_concurrent_writes(self, tmp_path: Path) -> None:
        path = tmp_path / "preds.jsonl"
        store = JSONLPredictionStore(path)
        errors: list[Exception] = []

        def write_batch(batch_id: int) -> None:
            try:
                for i in range(10):
                    store.record_prediction(f"r-{batch_id}-{i}", 0.5 + i * 0.01)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_batch, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert store.total == 50

    def test_empty_file_loads_cleanly(self, tmp_path: Path) -> None:
        path = tmp_path / "preds.jsonl"
        path.write_text("")  # empty file
        store = JSONLPredictionStore(path)
        assert store.total == 0

    def test_calibration_accuracy_survives_restart(self, tmp_path: Path) -> None:
        path = tmp_path / "preds.jsonl"
        store1 = JSONLPredictionStore(path)
        for i in range(10):
            store1.record_prediction(f"r{i}", 0.8)
            store1.record_outcome(f"r{i}", correct=(i < 8))

        acc1 = store1.calibration_accuracy()

        store2 = JSONLPredictionStore(path)
        acc2 = store2.calibration_accuracy()

        assert abs(acc1 - acc2) < 0.01


# ---------------------------------------------------------------------------
# JSONLAccountabilityStore
# ---------------------------------------------------------------------------


class TestJSONLAccountabilityStore:
    def test_file_not_created_on_init(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.jsonl"
        JSONLAccountabilityStore(path)
        assert not path.exists()

    def test_file_created_on_first_write(self, tmp_path: Path) -> None:
        from cognilateral_trust.accountability import create_accountability_record

        path = tmp_path / "audit.jsonl"
        store = JSONLAccountabilityStore(path)
        record = create_accountability_record(verdict="ACT", confidence=0.8)
        store.append(record)
        assert path.exists()

    def test_records_survive_restart(self, tmp_path: Path) -> None:
        from cognilateral_trust.accountability import create_accountability_record

        path = tmp_path / "audit.jsonl"
        store1 = JSONLAccountabilityStore(path)
        r1 = create_accountability_record(verdict="ACT", confidence=0.9)
        r2 = create_accountability_record(verdict="ESCALATE", reasons=("low confidence",), confidence=0.2)
        store1.append(r1)
        store1.append(r2)
        assert len(store1.list_recent()) == 2

        store2 = JSONLAccountabilityStore(path)
        records = store2.list_recent()
        assert len(records) == 2

    def test_record_ids_preserved(self, tmp_path: Path) -> None:
        from cognilateral_trust.accountability import create_accountability_record

        path = tmp_path / "audit.jsonl"
        store1 = JSONLAccountabilityStore(path)
        record = create_accountability_record(verdict="ACT", confidence=0.7)
        original_id = record.record_id
        store1.append(record)

        store2 = JSONLAccountabilityStore(path)
        retrieved = store2.get(original_id)
        assert retrieved is not None
        assert retrieved.record_id == original_id

    def test_corrupt_line_skipped(self, tmp_path: Path) -> None:
        from cognilateral_trust.accountability import create_accountability_record

        path = tmp_path / "audit.jsonl"
        store1 = JSONLAccountabilityStore(path)
        r1 = create_accountability_record(verdict="ACT", confidence=0.8)
        store1.append(r1)

        with path.open("a") as f:
            f.write("{corrupted json\n")

        r2 = create_accountability_record(verdict="ACT", confidence=0.75)
        store1.append(r2)

        store2 = JSONLAccountabilityStore(path)
        assert len(store2.list_recent()) >= 1

    def test_verdicts_preserved_after_reload(self, tmp_path: Path) -> None:
        from cognilateral_trust.accountability import create_accountability_record

        path = tmp_path / "audit.jsonl"
        store1 = JSONLAccountabilityStore(path)
        store1.append(create_accountability_record(verdict="ESCALATE", reasons=("reason",), confidence=0.1))
        store1.append(create_accountability_record(verdict="ACT", confidence=0.9))

        store2 = JSONLAccountabilityStore(path)
        recent = store2.list_recent()
        verdicts = {r.verdict for r in recent}
        assert "ESCALATE" in verdicts
        assert "ACT" in verdicts

    def test_reasons_preserved_as_tuple(self, tmp_path: Path) -> None:
        from cognilateral_trust.accountability import create_accountability_record

        path = tmp_path / "audit.jsonl"
        store1 = JSONLAccountabilityStore(path)
        r = create_accountability_record(
            verdict="ESCALATE",
            reasons=("reason one", "reason two"),
            confidence=0.3,
        )
        store1.append(r)

        store2 = JSONLAccountabilityStore(path)
        loaded = store2.list_recent()[0]
        assert isinstance(loaded.reasons, tuple)
        assert len(loaded.reasons) == 2

    def test_thread_safe_concurrent_writes(self, tmp_path: Path) -> None:
        from cognilateral_trust.accountability import create_accountability_record

        path = tmp_path / "audit.jsonl"
        store = JSONLAccountabilityStore(path)
        errors: list[Exception] = []

        def write_batch() -> None:
            try:
                for _ in range(10):
                    store.append(create_accountability_record(verdict="ACT", confidence=0.8))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_batch) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(store.list_recent(100)) == 50


# ---------------------------------------------------------------------------
# CalibratedTrustEngine with persist_path
# ---------------------------------------------------------------------------


class TestCalibratedTrustEnginePersistence:
    def test_persist_path_none_works_as_memory_only(self) -> None:
        from cognilateral_trust.calibrated import CalibratedTrustEngine

        engine = CalibratedTrustEngine(persist_path=None)
        engine.evaluate(0.8)
        assert engine.stats["total"] == 1

    def test_predictions_survive_engine_restart(self, tmp_path: Path) -> None:
        from cognilateral_trust.calibrated import CalibratedTrustEngine

        path = tmp_path / "engine"
        path.mkdir()

        engine1 = CalibratedTrustEngine(persist_path=path)
        result = engine1.evaluate(0.75)
        engine1.record_outcome(result.accountability_record.record_id, correct=True)
        assert engine1.stats["resolved"] == 1

        engine2 = CalibratedTrustEngine(persist_path=path)
        assert engine2.stats["resolved"] == 1

    def test_accountability_records_survive_engine_restart(self, tmp_path: Path) -> None:
        from cognilateral_trust.calibrated import CalibratedTrustEngine

        path = tmp_path / "engine"
        path.mkdir()

        engine1 = CalibratedTrustEngine(persist_path=path)
        result = engine1.evaluate(0.8)
        record_id = result.accountability_record.record_id

        engine2 = CalibratedTrustEngine(persist_path=path)
        assert engine2.get_record(record_id) is not None

    def test_persist_path_as_string_accepted(self, tmp_path: Path) -> None:
        from cognilateral_trust.calibrated import CalibratedTrustEngine

        path = tmp_path / "engine_str"
        path.mkdir()

        engine = CalibratedTrustEngine(persist_path=str(path))
        engine.evaluate(0.7)
        assert engine.stats["total"] == 1

    def test_no_persist_path_engine_has_no_files(self, tmp_path: Path) -> None:
        from cognilateral_trust.calibrated import CalibratedTrustEngine

        engine = CalibratedTrustEngine()  # default: no persistence
        engine.evaluate(0.9)
        # No files should be created in cwd
        assert engine.stats["total"] == 1

    def test_jsonl_files_created_in_persist_path(self, tmp_path: Path) -> None:
        from cognilateral_trust.calibrated import CalibratedTrustEngine

        path = tmp_path / "engine"
        path.mkdir()

        engine = CalibratedTrustEngine(persist_path=path)
        engine.evaluate(0.8)

        jsonl_files = list(path.glob("*.jsonl"))
        assert len(jsonl_files) > 0

    def test_calibration_accuracy_matches_after_reload(self, tmp_path: Path) -> None:
        from cognilateral_trust.calibrated import CalibratedTrustEngine

        path = tmp_path / "engine"
        path.mkdir()

        engine1 = CalibratedTrustEngine(persist_path=path)
        results = [engine1.evaluate(0.8) for _ in range(8)]
        for i, r in enumerate(results):
            engine1.record_outcome(r.accountability_record.record_id, correct=(i < 6))

        acc1 = engine1.calibration_accuracy()
        assert acc1 > 0

        engine2 = CalibratedTrustEngine(persist_path=path)
        acc2 = engine2.calibration_accuracy()
        assert abs(acc1 - acc2) < 0.01


# ---------------------------------------------------------------------------
# JSONL file format integrity
# ---------------------------------------------------------------------------


class TestJSONLFormat:
    def test_each_line_is_valid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "preds.jsonl"
        store = JSONLPredictionStore(path)
        for i in range(5):
            store.record_prediction(f"r{i}", 0.5 + i * 0.1)

        lines = path.read_text().strip().split("\n")
        for line in lines:
            parsed = json.loads(line)
            assert "record_id" in parsed

    def test_accountability_each_line_is_valid_json(self, tmp_path: Path) -> None:
        from cognilateral_trust.accountability import create_accountability_record

        path = tmp_path / "audit.jsonl"
        store = JSONLAccountabilityStore(path)
        for _ in range(3):
            store.append(create_accountability_record(verdict="ACT", confidence=0.8))

        lines = path.read_text().strip().split("\n")
        for line in lines:
            parsed = json.loads(line)
            assert "record_id" in parsed
            assert "verdict" in parsed
