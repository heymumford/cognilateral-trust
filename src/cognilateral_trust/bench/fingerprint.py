"""Calibration fingerprint generator — visualize per-domain calibration gaps."""

from __future__ import annotations

from dataclasses import dataclass

from cognilateral_trust.bench.scoring import BenchScore

__all__ = [
    "FingerprintSpoke",
    "CalibrationFingerprint",
    "generate_fingerprint",
    "fingerprint_to_dict",
]


@dataclass(frozen=True)
class FingerprintSpoke:
    """A radial spoke in a calibration fingerprint.

    Attributes:
        domain: Domain name (factual, reasoning, etc.)
        gap: Calibration gap = 1.0 - calibration_error (0.0 = poor, 1.0 = perfect)
    """

    domain: str
    gap: float


@dataclass(frozen=True)
class CalibrationFingerprint:
    """A radial/star visualization of per-domain calibration.

    Each spoke represents a domain. Spoke length (gap) indicates calibration quality.
    Tight circle = well-calibrated across all domains.
    Spiky star = overconfident or underconfident in specific domains.

    Attributes:
        model: Model name or identifier
        spokes: Tuple of FingerprintSpoke, one per domain
    """

    model: str
    spokes: tuple[FingerprintSpoke, ...]


def generate_fingerprint(score: BenchScore) -> CalibrationFingerprint:
    """Generate a calibration fingerprint from benchmark scores.

    Converts per-domain ECE scores into "spokes" where spoke length represents
    calibration quality (gap = 1.0 - calibration_error).

    Args:
        score: BenchScore with domain_scores

    Returns:
        CalibrationFingerprint with one spoke per domain
    """
    spokes: list[FingerprintSpoke] = []

    for domain_score in score.domain_scores:
        # gap = 1.0 - calibration_error: high gap = good calibration
        gap = 1.0 - domain_score.calibration_error
        spokes.append(
            FingerprintSpoke(
                domain=domain_score.domain,
                gap=gap,
            )
        )

    return CalibrationFingerprint(
        model=score.model,
        spokes=tuple(spokes),
    )


def fingerprint_to_dict(fp: CalibrationFingerprint) -> dict:
    """Convert CalibrationFingerprint to JSON-serializable dict.

    Args:
        fp: CalibrationFingerprint instance

    Returns:
        Dict with 'model', 'spokes' (list of {domain, gap})
    """
    return {
        "model": fp.model,
        "spokes": [
            {
                "domain": spoke.domain,
                "gap": spoke.gap,
            }
            for spoke in fp.spokes
        ],
    }
