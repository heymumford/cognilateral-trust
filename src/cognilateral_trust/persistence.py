"""JSONL persistence for PredictionStore and AccountabilityStore.

Provides drop-in replacements that auto-persist to JSONL files:
- JSONLPredictionStore(path) — predictions survive restarts
- JSONLAccountabilityStore(path) — accountability records survive restarts

Design:
- Load-on-init: all valid lines loaded when instance created
- Append-on-write: new records appended atomically
- Corrupt lines skipped gracefully (logged to stderr)
- Thread-safe: file writes protected by the parent class lock
- File created on first write (not on init)

JSONL format: one JSON object per line, UTF-8, LF newlines.

Public API:
    JSONLPredictionStore(path: Path | str)
    JSONLAccountabilityStore(path: Path | str)
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any

from cognilateral_trust.accountability import AccountabilityRecord, AccountabilityStore
from cognilateral_trust.prediction_store import PredictionRecord, PredictionStore

__all__ = [
    "JSONLAccountabilityStore",
    "JSONLPredictionStore",
]


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load all valid JSON lines from a JSONL file. Skips corrupt lines."""
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[cognilateral-trust] Skipping corrupt JSONL line {lineno}: {exc}", file=sys.stderr)
    return records


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON object to a JSONL file. Creates file if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


# ---------------------------------------------------------------------------
# JSONLPredictionStore
# ---------------------------------------------------------------------------


def _prediction_to_dict(rec: PredictionRecord) -> dict[str, Any]:
    return {
        "record_id": rec.record_id,
        "timestamp": rec.timestamp,
        "predicted_confidence": rec.predicted_confidence,
        "outcome": rec.outcome,
        "context": rec.context,
    }


def _prediction_from_dict(d: dict[str, Any]) -> PredictionRecord | None:
    try:
        return PredictionRecord(
            record_id=str(d["record_id"]),
            timestamp=float(d["timestamp"]),
            predicted_confidence=float(d["predicted_confidence"]),
            outcome=d["outcome"],
            context=str(d.get("context", "")),
        )
    except (KeyError, ValueError, TypeError):
        return None


class JSONLPredictionStore(PredictionStore):
    """PredictionStore that auto-persists to a JSONL file.

    Load-on-init: existing records loaded from file.
    Append-on-write: each new prediction or outcome update appended.

    Outcome updates are written as a full record snapshot — on reload
    the last record for each record_id wins (outcome overwrites pending).
    """

    def __init__(self, path: Path | str, max_records: int = 10000) -> None:
        super().__init__(max_records=max_records)
        self._path = Path(path)
        self._file_lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        """Load records from JSONL on init."""
        rows = _load_jsonl(self._path)
        # Build a dict keyed by record_id — last write for each id wins
        by_id: dict[str, PredictionRecord] = {}
        for d in rows:
            rec = _prediction_from_dict(d)
            if rec is not None:
                by_id[rec.record_id] = rec
        with self._lock:
            self._records = list(by_id.values())

    def record_prediction(
        self,
        record_id: str,
        predicted_confidence: float,
        context: str = "",
    ) -> PredictionRecord:
        rec = super().record_prediction(record_id, predicted_confidence, context=context)
        with self._file_lock:
            _append_jsonl(self._path, _prediction_to_dict(rec))
        return rec

    def record_outcome(self, record_id: str, correct: bool) -> PredictionRecord | None:
        rec = super().record_outcome(record_id, correct)
        if rec is not None:
            with self._file_lock:
                _append_jsonl(self._path, _prediction_to_dict(rec))
        return rec


# ---------------------------------------------------------------------------
# JSONLAccountabilityStore
# ---------------------------------------------------------------------------


def _record_to_dict(rec: AccountabilityRecord) -> dict[str, Any]:
    return {
        "record_id": rec.record_id,
        "timestamp": rec.timestamp,
        "verdict": rec.verdict,
        "reasons": list(rec.reasons),
        "context": rec.context,
        "confidence": rec.confidence,
        "confidence_tier": rec.confidence_tier,
    }


def _record_from_dict(d: dict[str, Any]) -> AccountabilityRecord | None:
    try:
        return AccountabilityRecord(
            record_id=str(d["record_id"]),
            timestamp=float(d["timestamp"]),
            verdict=d["verdict"],
            reasons=tuple(d.get("reasons", [])),
            context=d.get("context") or {},
            confidence=float(d.get("confidence", 1.0)),
            confidence_tier=d.get("confidence_tier"),
        )
    except (KeyError, ValueError, TypeError):
        return None


class JSONLAccountabilityStore(AccountabilityStore):
    """AccountabilityStore that auto-persists to a JSONL file.

    Load-on-init: existing records loaded from file.
    Append-on-write: each new record appended immediately.
    """

    def __init__(self, path: Path | str, max_records: int = 10000) -> None:
        super().__init__(max_records=max_records)
        self._path = Path(path)
        self._file_lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        """Load records from JSONL on init."""
        rows = _load_jsonl(self._path)
        loaded = []
        for d in rows:
            rec = _record_from_dict(d)
            if rec is not None:
                loaded.append(rec)
        with self._lock:
            self._records = loaded

    def append(self, record: AccountabilityRecord) -> None:
        super().append(record)
        with self._file_lock:
            _append_jsonl(self._path, _record_to_dict(record))
