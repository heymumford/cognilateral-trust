"""Provenance chain — cryptographic accountability."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class ChainEntry:
    """Immutable link in the provenance chain."""

    entry_id: str
    previous_hash: str
    agent_id: str
    action: str
    confidence: float
    evidence: tuple[str, ...] = ()
    timestamp: str = ""
    affected_party_id: str | None = None
    hash: str = ""


class ProvenanceChain:
    """SHA-256 cryptographic accountability ledger."""

    def __init__(self) -> None:
        """Initialize empty chain."""
        self._entries: list[ChainEntry] = []

    def append(
        self,
        agent_id: str,
        action: str,
        confidence: float,
        evidence: tuple[str, ...] = (),
        affected_party_id: str | None = None,
    ) -> ChainEntry:
        """Append an entry to the chain."""
        entry_id = str(uuid.uuid4())
        previous_hash = self._entries[-1].hash if self._entries else "0" * 64
        timestamp = datetime.now(timezone.utc).isoformat()

        entry_data = {
            "entry_id": entry_id,
            "previous_hash": previous_hash,
            "agent_id": agent_id,
            "action": action,
            "confidence": confidence,
            "evidence": evidence,
            "timestamp": timestamp,
            "affected_party_id": affected_party_id,
        }
        entry_json = json.dumps(entry_data, sort_keys=True, default=str)
        entry_hash = hashlib.sha256(entry_json.encode()).hexdigest()
        entry = ChainEntry(
            entry_id=entry_id,
            previous_hash=previous_hash,
            agent_id=agent_id,
            action=action,
            confidence=confidence,
            evidence=evidence,
            timestamp=timestamp,
            affected_party_id=affected_party_id,
            hash=entry_hash,
        )
        self._entries.append(entry)
        return entry

    def verify(self) -> bool:
        """Verify chain integrity."""
        if not self._entries:
            return True
        for i, entry in enumerate(self._entries):
            if i == 0:
                expected_prev = "0" * 64
            else:
                expected_prev = self._entries[i - 1].hash
            if entry.previous_hash != expected_prev:
                return False
            entry_data = {
                "entry_id": entry.entry_id,
                "previous_hash": entry.previous_hash,
                "agent_id": entry.agent_id,
                "action": entry.action,
                "confidence": entry.confidence,
                "evidence": entry.evidence,
                "timestamp": entry.timestamp,
                "affected_party_id": entry.affected_party_id,
            }
            entry_json = json.dumps(entry_data, sort_keys=True, default=str)
            computed_hash = hashlib.sha256(entry_json.encode()).hexdigest()
            if entry.hash != computed_hash:
                return False
        return True

    def query_by_affected_party(self, party_id: str) -> tuple[ChainEntry, ...]:
        """Query entries by affected party ID."""
        return tuple(e for e in self._entries if e.affected_party_id == party_id)

    def to_json(self) -> str:
        """Serialize chain to JSON string."""
        entries_data = [asdict(entry) for entry in self._entries]
        return json.dumps({"entries": entries_data}, indent=2, default=str)

    @classmethod
    def from_json(cls, data: str) -> ProvenanceChain:
        """Restore chain from JSON string."""
        chain = cls()
        parsed = json.loads(data)
        entries_data = parsed.get("entries", [])
        for entry_dict in entries_data:
            entry = ChainEntry(
                entry_id=entry_dict["entry_id"],
                previous_hash=entry_dict["previous_hash"],
                agent_id=entry_dict["agent_id"],
                action=entry_dict["action"],
                confidence=entry_dict["confidence"],
                evidence=tuple(entry_dict.get("evidence", [])),
                timestamp=entry_dict.get("timestamp", ""),
                affected_party_id=entry_dict.get("affected_party_id"),
                hash=entry_dict["hash"],
            )
            chain._entries.append(entry)
        return chain
