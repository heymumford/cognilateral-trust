"""Trust decay model — confidence attenuation with agent-to-agent distance.

Pure functions for calculating how confidence diminishes as claims propagate
through multi-hop agent networks. Two curves supported: exponential (half-life)
and linear (fixed rate per hop).

Properties:
- Pure function, no side effects
- decay(c, 0) == c (identity at zero hops)
- Monotonically non-increasing: decay(c, n) >= decay(c, n+1) for all n
- Result always in [0.0, 1.0]
"""

from __future__ import annotations


def exponential_decay(
    confidence: float,
    hops: int,
    half_life: float = 3.0,
) -> float:
    """Confidence decays exponentially with agent-to-agent distance.

    Formula: confidence * (0.5 ^ (hops / half_life))

    Args:
        confidence: Starting confidence [0.0, 1.0]
        hops: Number of agent-to-agent transfer steps
        half_life: Hops at which confidence reaches 50% (default 3.0)

    Returns:
        Attenuated confidence [0.0, 1.0]
    """
    if hops == 0:
        return confidence
    if confidence == 0.0:
        return 0.0
    exponent = hops / half_life
    decay_factor = 0.5**exponent
    result = confidence * decay_factor
    return max(0.0, min(1.0, result))


def linear_decay(
    confidence: float,
    hops: int,
    rate: float = 0.1,
) -> float:
    """Confidence decays linearly per hop.

    Formula: max(0, confidence - (rate * hops))

    Args:
        confidence: Starting confidence [0.0, 1.0]
        hops: Number of agent-to-agent transfer steps
        rate: Confidence loss per hop (default 0.1)

    Returns:
        Attenuated confidence [0.0, 1.0]
    """
    if hops == 0:
        return confidence
    result = confidence - (rate * hops)
    return max(0.0, min(1.0, result))


def decay(
    confidence: float,
    hops: int,
    curve: str = "exponential",
    **kwargs,
) -> float:
    """Dispatch to appropriate decay function.

    Args:
        confidence: Starting confidence [0.0, 1.0]
        hops: Number of agent-to-agent transfer steps
        curve: Decay curve type: "exponential" or "linear" (default "exponential")
        **kwargs: Additional parameters for the selected curve
                  - exponential: half_life (default 3.0)
                  - linear: rate (default 0.1)

    Returns:
        Attenuated confidence [0.0, 1.0]

    Raises:
        ValueError: If curve type is invalid
    """
    if curve == "exponential":
        half_life = kwargs.get("half_life", 3.0)
        return exponential_decay(confidence, hops, half_life=half_life)
    elif curve == "linear":
        rate = kwargs.get("rate", 0.1)
        return linear_decay(confidence, hops, rate=rate)
    else:
        msg = f"Invalid curve type: {curve}. Expected 'exponential' or 'linear'."
        raise ValueError(msg)
