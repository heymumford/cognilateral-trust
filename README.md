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
| [`langgraph_trust_node.py`](examples/langgraph_trust_node.py) | Trust gate node for LangGraph pipelines |
| [`crewai_trust_tool.py`](examples/crewai_trust_tool.py) | CrewAI tool wrapping trust evaluation |
| [`openai_trust_wrapper.py`](examples/openai_trust_wrapper.py) | Confidence scoring wrapper for OpenAI/Anthropic LLM calls |
| [`mem0_trust_provider.py`](examples/mem0_trust_provider.py) | Memory confidence scoring for Mem0 agent memories |
| [`github_actions_trust_gate.yml`](examples/github_actions_trust_gate.yml) | CI trust gate for auto-merge decisions in GitHub Actions |

## Why This Exists

Guardrails protect systems. Trust protects people.

When an AI agent acts with misplaced confidence, the person at the other end bears the cost — the student who gets a zero because the AI fabricated a source, the operator who ships a broken deploy at 3am because the AI said it was fine.

This library exists so AI systems can say "I'm not sure enough to act on this" before the damage happens.

## License

Apache-2.0
