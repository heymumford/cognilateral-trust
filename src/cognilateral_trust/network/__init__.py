"""Trust network primitives."""

from __future__ import annotations

from .contagion import ContagionAlert, ContagionTracker, PropagationEntry
from .consensus import ConsensusResult, WeightedVerdict, weighted_consensus
from .decay import decay, exponential_decay, linear_decay
from .provenance import ChainEntry, ProvenanceChain
from .sovereign import SovereignRefusal, sovereign_worker

__all__ = [
    "exponential_decay",
    "linear_decay",
    "decay",
    "PropagationEntry",
    "ContagionAlert",
    "ContagionTracker",
    "WeightedVerdict",
    "ConsensusResult",
    "weighted_consensus",
    "SovereignRefusal",
    "sovereign_worker",
    "ChainEntry",
    "ProvenanceChain",
]
