# TrustBench Calibration Results

**Model:** cognilateral-trust v1.2.0
**Date:** 2026-03-26
**Scenarios:** 200 (40 per domain)
**Methodology:** Deterministic confidence → tier → route evaluation. No LLM in the loop.

## What This Measures

TrustBench measures calibration error: the gap between what the system says (confidence tier → routing decision) and what a correctly calibrated system would say.

A perfectly calibrated system has 0.0 error. A system that routes randomly has ~0.5 error.

## Results

| Domain | Calibration Error | Interpretation |
|--------|------------------:|----------------|
| Factual | 0.575 | Over-routes factual queries to higher tiers |
| Reasoning | 0.550 | Over-routes reasoning queries to higher tiers |
| Ambiguous | 0.250 | Good calibration on genuinely uncertain inputs |
| Out-of-distribution | 0.250 | Good calibration — correctly escalates unknowns |
| Adversarial | 0.250 | Good calibration — correctly escalates adversarial inputs |
| **Overall** | **0.375** | |

## Interpretation

The system is well-calibrated for uncertainty (ambiguous, OOD, adversarial domains — 0.25 error). It over-routes factual/reasoning queries, sending them to higher verification tiers than needed. This is the conservative direction — better to over-escalate than under-escalate.

Improving factual/reasoning calibration (reducing from 0.55 to 0.3) would bring the overall score to ~0.27.

## Reproducibility

```bash
pip install cognilateral-trust
trust-bench run --model "your-label" --output results.json
```

Results are deterministic (no randomness in the evaluation pipeline). Expected variance across runs: 0%.

## Falsifiability

Per D4 (World Cafe decision): these numbers are published so anyone can falsify them. Run TrustBench on your workload. If the calibration error is materially different from what we report, that's a bug — file an issue.
