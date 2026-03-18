# cognilateral-trust

**AI that tells you when it's guessing.**

An epistemic trust layer for AI agents. Maps confidence to action — so autonomous systems know when to act and when to ask.

[![PyPI](https://img.shields.io/pypi/v/cognilateral-trust)](https://pypi.org/project/cognilateral-trust/)
[![Python](https://img.shields.io/pypi/pyversions/cognilateral-trust)](https://pypi.org/project/cognilateral-trust/)
[![License](https://img.shields.io/github/license/heymumford/cognilateral-trust)](LICENSE)

## Live Demo

Dashboard: [cognilateral.fly.dev/trust](https://cognilateral.fly.dev/trust)

CLI:
```bash
$ trust-check 0.7
ACT — C7 (sovereignty_gate)

$ trust-check 0.92 --irreversible
ESCALATE — C9 (sovereignty_gate): irreversible action at sovereignty-grade tier

$ trust-check 0.3 --json
{"confidence": 0.3, "tier": "C3", "route": "basic", "should_proceed": true, "verdict": "ACT", ...}
```

REST API (no install needed):
```bash
$ curl -s https://cognilateral.fly.dev/api/trust/summary | jq .calibration.accuracy
0.0
```

## Install

```bash
pip install cognilateral-trust
```

Zero dependencies. Python 3.11+.

## Quick Start

```python
from cognilateral_trust import evaluate_trust

result = evaluate_trust(0.7)

if result.should_proceed:
    perform_action()
else:
    print(f"Escalate: {result.accountability_record.reasons}")
```

Every call produces an immutable accountability record — who decided, why, and at what confidence level.

## What It Does

Your AI agent is about to take an action. How confident is it? And is that confidence warranted?

`cognilateral-trust` answers three questions:

1. **What tier is this confidence?** Maps 0.0-1.0 to C0-C9 epistemic tiers
2. **What route should it take?** Basic (act freely), warrant check (verify evidence), or sovereignty gate (require approval)
3. **Should it proceed?** Considers reversibility, external impact, and confidence level

```
Confidence 0.0-0.3  →  C0-C3  →  basic           →  act freely
Confidence 0.4-0.6  →  C4-C6  →  warrant_check   →  verify evidence first
Confidence 0.7-1.0  →  C7-C9  →  sovereignty_gate →  require approval for irreversible actions
```

## Confidence Extraction

Extract confidence signals from LLM responses — no external dependencies.

```python
from cognilateral_trust import extract_confidence

# Auto-detect format (OpenAI, Anthropic, or plain text)
conf = extract_confidence(openai_response)

# From plain text
from cognilateral_trust import extract_confidence_from_text
conf = extract_confidence_from_text("I'm about 73% confident in this answer")
# => 0.73
```

Parses explicit percentages, decimal values, verbal qualifiers ("highly confident", "somewhat uncertain"), and logprobs from OpenAI/Anthropic response dicts.

## Warrants

Evidence-backed confidence that decays over time. A Warrant wraps a confidence value with provenance and a TTL — the effective confidence degrades linearly as the warrant ages, reaching zero at expiry.

```python
from cognilateral_trust import Warrant, evaluate_warrant, evaluate_trust_with_warrant

w = Warrant(
    confidence=0.9,
    evidence_source="unit tests pass",
    ttl_seconds=3600,
)

# Current effective confidence (decays toward 0 as TTL expires)
effective = evaluate_warrant(w)

# Full trust evaluation using warrant's decayed confidence
result = evaluate_trust_with_warrant(w)
```

`WarrantStore` manages collections of warrants with automatic expiry and lookup by key or evidence source.

## Middleware

Trust-gated execution for pipelines and frameworks.

```python
from cognilateral_trust import trust_gate, TrustMiddleware

# Decorator — raises TrustEscalation if confidence < threshold
@trust_gate(min_confidence=0.6)
def deploy(confidence: float, **kwargs):
    ...

# Async evaluation
from cognilateral_trust import async_evaluate_trust
result = await async_evaluate_trust(0.8, is_reversible=False)

# Pipeline middleware
mw = TrustMiddleware(min_confidence=0.5, on_escalation="raise")
result = mw(0.7, context={"action": "deploy"})
```

## Persistence

JSONL-backed stores that survive restarts. Drop-in replacements for the in-memory stores.

```python
from cognilateral_trust import JSONLPredictionStore, JSONLAccountabilityStore

predictions = JSONLPredictionStore("./data/predictions.jsonl")
accountability = JSONLAccountabilityStore("./data/accountability.jsonl")
```

Load on init, append on write, skip corrupt lines gracefully. Thread-safe.

## Claim Extraction

Extract typed claims from free-form text using regex and rule-based heuristics. No NLP dependencies required.

```python
from cognilateral_trust import extract_claims

claims = extract_claims("The model achieves 95% accuracy, outperforming GPT-4.")

for claim in claims.claims:
    print(f"{claim.claim_type}: {claim.text}")
# factual: The model achieves 95% accuracy
# comparative: outperforming GPT-4
```

Detects `factual`, `causal`, `comparative`, and `quantitative` claim types with source-span provenance.

## Fidelity Verification

Verify that a claim is actually supported by its source text using word-overlap heuristics (Jaccard/overlap coefficient).

```python
from cognilateral_trust import verify_fidelity

result = verify_fidelity(
    claim="The system processes 10,000 requests per second",
    source="Our benchmarks show the system handles 10,000 req/s under load",
)

print(result.supported)   # True
print(result.score)       # 0.78
print(result.method)      # "overlap"
```

`verify_fidelity_batch()` checks multiple claims against a source in one call.

## Epistemic Firewall

Detects when an action demands a higher epistemic level than the evidence supplies. Prevents agents from acting on hunches when the situation calls for tested facts.

```python
from cognilateral_trust import EpistemicFirewall, EpistemicLevel, check_epistemic_mismatch

# Quick check
result = check_epistemic_mismatch(
    demanded=EpistemicLevel.VALIDATED,
    supplied=EpistemicLevel.OBSERVED,
)
print(result.is_mismatch)  # True
print(result.gap)          # 3

# Registry-based firewall
fw = EpistemicFirewall()
fw.register("deploy", EpistemicLevel.TESTED)
fw.register("draft", EpistemicLevel.RAW)

fw.check("deploy", supplied=EpistemicLevel.OBSERVED)  # MismatchResult(is_mismatch=True, ...)
fw.check("draft", supplied=EpistemicLevel.RAW)          # MismatchResult(is_mismatch=False, ...)
```

Seven levels: `RAW` < `OBSERVED` < `MEASURED` < `TESTED` < `VALIDATED` < `FALSIFIABLE` < `GOVERNANCE`.

## Sovereignty Gate

The D-07 governance contract — decides whether an agent may act autonomously or must escalate to human review. Includes the D-05 welfare hard constraint: welfare-critical actions **always** escalate, regardless of confidence.

```python
from cognilateral_trust import evaluate_sovereignty, sovereignty_gate

# Evaluate directly
decision = evaluate_sovereignty(
    confidence=0.8,
    is_reversible=True,
    tests_pass=True,
    welfare_affected=False,
    external_impact=0,
)
print(decision.verdict)  # "ACT"

# As a decorator — wraps functions with automatic sovereignty checks
@sovereignty_gate(confidence_param="conf")
def autonomous_action(conf: float, **kwargs):
    return "executed"
```

`SovereigntyPolicy` configures thresholds. `DEFAULT_POLICY` provides reasonable defaults (min confidence 0.5, reversibility required, tests must pass, welfare gate active).

## Framework Integrations

### LangGraph

```python
from cognilateral_trust import evaluate_trust

def trust_gate(state):
    result = evaluate_trust(state["confidence"])
    return {**state, "proceed": result.should_proceed}
```

See [`examples/langgraph_trust_node.py`](examples/langgraph_trust_node.py) for a full LangGraph integration.

### CrewAI

```python
from cognilateral_trust import evaluate_trust

def trust_check(confidence, action="unknown", **kwargs):
    result = evaluate_trust(confidence, **kwargs)
    return "PROCEED" if result.should_proceed else f"ESCALATE: {result.accountability_record.reasons}"
```

See [`examples/crewai_trust_tool.py`](examples/crewai_trust_tool.py) for a full CrewAI tool.

### Any Framework

`evaluate_trust()` is framework-agnostic. It takes a float, returns a decision. Wrap it in whatever your framework expects.

## API

### `evaluate_trust(confidence, *, is_reversible=True, touches_external=False, context=None)`

Returns a `TrustEvaluation`:

| Field | Type | Description |
|-------|------|-------------|
| `confidence` | `float` | Input confidence (0.0-1.0) |
| `tier` | `ConfidenceTier` | C0-C9 epistemic tier |
| `route` | `str` | `"basic"`, `"warrant_check"`, or `"sovereignty_gate"` |
| `should_proceed` | `bool` | Whether the action should go ahead |
| `accountability_record` | `AccountabilityRecord` | Immutable audit trail entry |

### `route_by_tier(tier)` / `evaluate_tier_routing(tier)`

Lower-level routing if you need direct tier control.

### `AccountabilityStore`

Thread-safe in-memory store for accountability records. Every decision gets logged.

## Examples

| Example | Description |
|---------|-------------|
| [`demo_trust_agent.py`](examples/demo_trust_agent.py) | Full trust agent demo with extraction, warrants, and sovereignty |
| [`demo_calibration_loop.py`](examples/demo_calibration_loop.py) | Calibration loop with prediction tracking |
| [`langgraph_trust_node.py`](examples/langgraph_trust_node.py) | Trust gate node for LangGraph pipelines |
| [`crewai_trust_tool.py`](examples/crewai_trust_tool.py) | CrewAI tool wrapping trust evaluation |
| [`openai_trust_wrapper.py`](examples/openai_trust_wrapper.py) | Confidence extraction wrapper for OpenAI calls |
| [`anthropic_trust_wrapper.py`](examples/anthropic_trust_wrapper.py) | Confidence extraction wrapper for Anthropic calls |
| [`dspy_trust_module.py`](examples/dspy_trust_module.py) | DSPy module with trust-gated assertions |
| [`mem0_trust_provider.py`](examples/mem0_trust_provider.py) | Memory confidence scoring for Mem0 agent memories |
| [`cognee_trust_layer.py`](examples/cognee_trust_layer.py) | Knowledge graph trust gates for Cognee |
| [`load_simulator.py`](examples/load_simulator.py) | Concurrent load testing for trust evaluation throughput |
| [`http_trust_proxy.py`](examples/http_trust_proxy.py) | HTTP trust proxy with REST API |
| [`github_actions_trust_gate.yml`](examples/github_actions_trust_gate.yml) | CI trust gate for auto-merge decisions in GitHub Actions |

## Why This Exists

Guardrails protect systems. Trust protects people.

When an AI agent acts with misplaced confidence, the person at the other end bears the cost — the student who gets a zero because the AI fabricated a source, the operator who ships a broken deploy at 3am because the AI said it was fine.

This library exists so AI systems can say "I'm not sure enough to act on this" before the damage happens.

## License

Apache-2.0
