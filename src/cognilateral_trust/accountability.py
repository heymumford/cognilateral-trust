"""Accountability records — immutable audit trail for trust decisions."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class AccountabilityRecord:
    """Immutable record of a trust decision."""

    record_id: str
    timestamp: float
    verdict: Literal["ACT", "ESCALATE"]
    reasons: tuple[str, ...]
    context: dict
    confidence: float
    confidence_tier: int | None


def create_accountability_record(
    *,
    verdict: str,
    reasons: tuple[str, ...] = (),
    context: dict | None = None,
    confidence: float = 1.0,
    confidence_tier: int | None = None,
) -> AccountabilityRecord:
    """Create an accountability record with auto-generated ID and timestamp."""
    return AccountabilityRecord(
        record_id=str(uuid.uuid4()),
        timestamp=time.time(),
        verdict=verdict,
        reasons=reasons,
        context=context or {},
        confidence=confidence,
        confidence_tier=confidence_tier,
    )


class AccountabilityStore:
    """Thread-safe in-memory store for accountability records."""

    def __init__(self, max_records: int = 10000) -> None:
        self._lock = threading.Lock()
        self._records: list[AccountabilityRecord] = []
        self._max = max_records

    def append(self, record: AccountabilityRecord) -> None:
        with self._lock:
            self._records.append(record)
            if len(self._records) > self._max:
                self._records = self._records[-self._max :]

    def list_recent(self, n: int = 100) -> list[AccountabilityRecord]:
        with self._lock:
            return list(self._records[-n:])

    def get(self, record_id: str) -> AccountabilityRecord | None:
        with self._lock:
            for r in reversed(self._records):
                if r.record_id == record_id:
                    return r
            return None
