# Trust Evaluation Protocol v1

## Abstract

The Trust Evaluation Protocol v1 defines a machine-readable contract for agents to request epistemic confidence assessment and receive routing decisions (ACT or ESCALATE). It maps confidence values to guarantee tiers (C0-C9), applies context-dependent gates, and transfers confidence artifacts across agent boundaries. This protocol enables portable, auditable trust decisions across heterogeneous agent architectures.

## Terminology

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in RFC 2119.

- **Confidence**: A probability estimate (0.0-1.0) representing belief in a proposed action.
- **Tier**: A confidence band (C0-C9) derived from confidence, determining validation gates.
- **Route**: The decision path applied to a tier (basic, warrant_check, sovereignty_gate).
- **Verdict**: The final routing decision: ACT (proceed autonomously) or ESCALATE (require human/escalation agent approval).
- **Warrant**: Evidence-backed confidence claim with optional decay over time.
- **Passport**: Accumulated trust context (stamps, warrant chain) that travels with agent handoffs.
- **Stamp**: A record of a trust-relevant action (spawn, handoff, evaluate, terminate) taken by an agent.

## Message Format: Evaluation Request

An Evaluation Request is a JSON object requesting trust assessment for a proposed action.

### JSON Schema

```json
{
  "type": "object",
  "required": ["confidence"],
  "properties": {
    "confidence": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0,
      "description": "Epistemic confidence in the proposed action (0.0-1.0)"
    },
    "evidence": {
      "type": "array",
      "items": {"type": "string"},
      "default": [],
      "description": "Supporting evidence for the confidence estimate"
    },
    "action_reversible": {
      "type": "boolean",
      "default": true,
      "description": "Whether the action can be undone or reverted"
    },
    "welfare_relevant": {
      "type": "boolean",
      "default": false,
      "description": "Whether the action affects human/agent welfare"
    },
    "touches_external": {
      "type": "boolean",
      "default": false,
      "description": "Whether the action touches external systems (APIs, databases, file systems outside scope)"
    }
  }
}
```

### Fields

- `confidence` (number, required): The proposed action's confidence level as a decimal fraction between 0.0 and 1.0 inclusive.
- `evidence` (array of strings, optional): List of evidence items supporting the confidence estimate. Each item SHOULD be a brief string describing source, method, or claim (e.g., "observed in 3 of 4 test runs", "attestation from 2 independent agents").
- `action_reversible` (boolean, optional, default: true): True if the action can be completely undone or reverted; false if irreversible (e.g., deletion, commitment, external API call with side effects).
- `welfare_relevant` (boolean, optional, default: false): True if the action affects human welfare, safety, or agent lifecycle; false otherwise.
- `touches_external` (boolean, optional, default: false): True if the action interacts with external systems (network APIs, file systems, databases outside the agent's scope); false otherwise. Used in sovereignty_gate checks.

## Message Format: Evaluation Response

An Evaluation Response is a JSON object containing the trust decision and supporting metadata.

### JSON Schema

```json
{
  "type": "object",
  "required": ["verdict", "tier", "route", "record_id", "timestamp"],
  "properties": {
    "verdict": {
      "type": "string",
      "enum": ["ACT", "ESCALATE"],
      "description": "Final routing decision"
    },
    "tier": {
      "type": "integer",
      "minimum": 0,
      "maximum": 9,
      "description": "Confidence tier (C0-C9)"
    },
    "route": {
      "type": "string",
      "enum": ["basic", "warrant_check", "sovereignty_gate"],
      "description": "Validation path applied"
    },
    "record_id": {
      "type": "string",
      "description": "Unique identifier for this evaluation record (UUID or equivalent)"
    },
    "reasons": {
      "type": "array",
      "items": {"type": "string"},
      "default": [],
      "description": "If verdict is ESCALATE, reasons for escalation"
    },
    "timestamp": {
      "type": "string",
      "format": "date-time",
      "description": "ISO 8601 timestamp of the evaluation"
    }
  }
}
```

### Fields

- `verdict` (string, required): Either "ACT" (proceed autonomously) or "ESCALATE" (require escalation/approval).
- `tier` (integer, required): The confidence tier (0-9) derived from the input confidence value.
- `route` (string, required): The validation path applied (basic, warrant_check, or sovereignty_gate).
- `record_id` (string, required): A unique identifier for auditing this decision (MUST be globally unique or at least unique within a session/agent boundary).
- `reasons` (array of strings, optional): Human-readable reasons. If verdict is ESCALATE, MUST contain at least one reason explaining why the action cannot proceed autonomously.
- `timestamp` (string, required): ISO 8601 datetime (e.g., "2026-03-24T14:30:45Z") when the evaluation was completed.

## Verdict Semantics

### ACT
An ACT verdict authorizes the agent to proceed with the proposed action autonomously, without escalation or human approval.

ACT is returned when:
- Confidence tier is 0-3 (basic route) with no exclusionary conditions, OR
- Confidence tier is 4-6 (warrant_check route) with valid warrant evidence and no exclusionary conditions, OR
- Confidence tier is 7-9 (sovereignty_gate route) AND the action is reversible AND it does not touch external systems AND confidence >= 0.5.

### ESCALATE
An ESCALATE verdict forbids autonomous execution and requires escalation to a human or escalation agent before proceeding.

ESCALATE is returned when:
- Action is irreversible and tier >= 7, OR
- Action touches external systems and tier >= 7, OR
- Confidence < 0.5 and tier >= 7, OR
- Warrant check fails for tier 4-6 (no valid evidence or expired warrant), OR
- Any sovereignty gate condition is unmet.

ESCALATE verdicts MUST include at least one reason explaining the gate failure.

## Confidence Tiers

Each tier maps a confidence range to a governance route. Confidence is mapped to tier as: `tier = min(9, max(0, floor(confidence * 10)))`.

| Tier | Name | Confidence Range | Route | Semantics |
|:----:|------|------------------|-------|-----------|
| C0 | Unverified | 0.00–0.09 | basic | Pure speculation or absent foundation |
| C1 | Anecdotal | 0.10–0.19 | basic | Single observation or report |
| C2 | Observed | 0.20–0.29 | basic | Repeated observation, not measured |
| C3 | Measured | 0.30–0.39 | basic | Quantified but limited coverage |
| C4 | Tested | 0.40–0.49 | warrant_check | Experimental validation, baseline coverage |
| C5 | Validated | 0.50–0.59 | warrant_check | Multiple validation methods consistent |
| C6 | Falsifiable | 0.60–0.69 | warrant_check | Passes adverse conditions, boundary testing |
| C7 | Governance-grade | 0.70–0.79 | sovereignty_gate | Fitness for binding commitments or control transfer |
| C8 | Audited | 0.80–0.89 | sovereignty_gate | Independent verification, regulatory compliance |
| C9 | Resilient | 0.90–1.00 | sovereignty_gate | Proven across diverse conditions, production-proven |

**Route Mapping:**
- **basic** (tiers 0-3): No warrant evidence required. Return ACT for C0-C2 (confidence < 0.3). For C3 (0.3-0.39), escalate if action is irreversible AND welfare_relevant, otherwise ACT.
- **warrant_check** (tiers 4-6): Warrant evidence MUST be supplied and valid (not expired, not below decay threshold). ACT if warrant passes, otherwise ESCALATE.
- **sovereignty_gate** (tiers 7-9): Warrant evidence required, PLUS irreversibility check, external impact check, and minimum confidence threshold (>= 0.5). ACT only if all gates pass; otherwise ESCALATE.

## Warrant Format

A warrant is an evidence-backed confidence claim with optional time-based decay.

### Warrant Structure

```json
{
  "evidence": ["source1", "source2"],
  "issued_at": "2026-03-24T10:00:00Z",
  "ttl_seconds": 3600,
  "original_confidence": 0.75,
  "decay_rate": 0.0
}
```

### Warrant Fields

- `evidence` (array of strings, required): List of evidence items justifying the confidence claim.
- `issued_at` (string, ISO 8601): When the warrant was issued.
- `ttl_seconds` (number, optional, default: 86400): Time-to-live in seconds. After this period, the warrant is expired.
- `original_confidence` (number, optional): The initial confidence value at warrant issuance. Used in decay calculations. If omitted, defaults to the confidence value from the Evaluation Request.
- `decay_rate` (number, optional, default: 0.0): Exponential decay rate per second. Effective confidence = original_confidence * max(0.0, 1.0 - decay_rate * elapsed_seconds / ttl_seconds).

### Warrant Validity

A warrant is valid if:
- `issued_at` + `ttl_seconds` has not elapsed, AND
- Effective confidence (after decay) remains >= tier threshold.

If a warrant is expired or decayed below threshold, the evaluation MUST return ESCALATE with reason "warrant expired or decayed".

## Trust Passport

A Trust Passport is an immutable record of trust-relevant actions and warrant history that follows an agent through handoffs.

### Passport Structure

```json
{
  "agent_id": "agent-uuid-001",
  "created_at": "2026-03-24T09:00:00Z",
  "context": {
    "confidence": 0.75,
    "evidence": ["calibrated on 50 decisions", "0.82 precision"],
    "action_reversible": true,
    "welfare_relevant": false
  },
  "stamps": [
    {
      "agent_id": "agent-uuid-001",
      "action": "spawn",
      "confidence_at_action": 0.8,
      "timestamp": "2026-03-24T09:00:00Z",
      "evidence_added": ["initialized with priors"]
    },
    {
      "agent_id": "agent-uuid-001",
      "action": "evaluate",
      "confidence_at_action": 0.75,
      "timestamp": "2026-03-24T10:00:00Z",
      "evidence_added": ["observed 2 successes", "1 failure"]
    }
  ],
  "warrant_chain": ["warrant-001", "warrant-002"]
}
```

### Passport Fields

- `agent_id` (string): The ID of the agent that created this passport.
- `created_at` (string, ISO 8601): When the passport was issued.
- `context` (TrustContext object): The epistemic context (confidence, evidence, reversibility, welfare relevance).
- `stamps` (array of PassportStamp): A chronological sequence of trust-relevant actions.
- `warrant_chain` (array of strings): IDs of warrants supporting the passport's confidence.

### Passport Semantics

- A passport MUST be immutable once created (no mutations, only append-only stamp additions).
- When an agent hands off control to another agent, it SHOULD transfer the existing passport plus append a "handoff" stamp.
- When a passport is transferred, the receiving agent SHOULD validate all stamps and warrants before acting on the confidence value.
- Passport size SHOULD be bounded to prevent unbounded growth (recommended: truncate to last N stamps if > 1000 stamps).

## Conformance Requirements

Any implementation of the Trust Evaluation Protocol v1 MUST satisfy:

1. **Input validation**: Reject confidence values outside [0.0, 1.0]. Return error response with HTTP 400 or equivalent.

2. **Tier mapping**: Confidence MUST map to tier via `tier = min(9, max(0, floor(confidence * 10)))`.

3. **Route routing**: Route MUST be determined by: basic if tier <= 3, warrant_check if 4 <= tier <= 6, sovereignty_gate if tier >= 7.

4. **Verdict logic**:
   - basic route: Return ACT unless action is irreversible AND welfare_relevant.
   - warrant_check route: Return ACT if warrant is valid and not expired; otherwise ESCALATE.
   - sovereignty_gate route: Return ACT only if action_reversible is true AND no external impact AND confidence >= 0.5; otherwise ESCALATE.

5. **Response format**: Response MUST include verdict, tier, route, record_id, reasons (if ESCALATE), and ISO 8601 timestamp.

6. **Record ID uniqueness**: record_id MUST be unique per evaluation (UUID v4 or equivalent). MUST NOT reuse IDs within a session.

7. **Audit trail**: Each evaluation MUST be logged or recorded for later audit. Minimum fields: timestamp, confidence, verdict, tier, reasons.

8. **Warrant validation**: If warrant evidence is supplied, the implementation MUST validate expiration (issued_at + ttl_seconds) and decay before authorizing ACT.

9. **Passport immutability**: TrustPassport objects MUST be immutable. add_stamp() MUST return a new passport, not mutate in-place.

10. **Error handling**: If an error occurs during evaluation (e.g., invalid JSON, missing required field), return HTTP 400 or equivalent with an error message. Do not return a valid verdict for invalid input.

## Examples

### Example 1: Low-Confidence ACT (Basic Route)

**Request:**
```json
{
  "confidence": 0.25,
  "evidence": ["observed once"],
  "action_reversible": true,
  "welfare_relevant": false
}
```

**Evaluation:**
- Tier: C2 (0.25 * 10 = 2.5 → 2)
- Route: basic (tier 2 <= 3)
- Verdict: ACT (no exclusionary conditions for basic route)

**Response:**
```json
{
  "verdict": "ACT",
  "tier": 2,
  "route": "basic",
  "record_id": "eval-20260324-001-uuid",
  "reasons": [],
  "timestamp": "2026-03-24T14:30:45Z"
}
```

### Example 2: High-Confidence ESCALATE (Irreversible Action)

**Request:**
```json
{
  "confidence": 0.85,
  "evidence": ["tested in 5 scenarios", "manual review passed", "production migration plan approved"],
  "action_reversible": false,
  "welfare_relevant": true
}
```

**Evaluation:**
- Tier: C8 (0.85 * 10 = 8.5 → 8)
- Route: sovereignty_gate (tier 8 >= 7)
- Verdict: ESCALATE (action_reversible is false at sovereignty_gate tier)

**Response:**
```json
{
  "verdict": "ESCALATE",
  "tier": 8,
  "route": "sovereignty_gate",
  "record_id": "eval-20260324-002-uuid",
  "reasons": ["irreversible action at sovereignty-grade tier"],
  "timestamp": "2026-03-24T14:31:20Z"
}
```

### Example 3: Warrant-Checked Decision (Tier 5)

**Request:**
```json
{
  "confidence": 0.55,
  "evidence": ["validated on 100 test cases", "precision 0.89", "recall 0.91"],
  "action_reversible": true,
  "welfare_relevant": false
}
```

**Evaluation:**
- Tier: C5 (0.55 * 10 = 5.5 → 5)
- Route: warrant_check (tier 5 in [4, 6])
- Warrant supplied: evidence list covers validation. Warrant not expired.
- Verdict: ACT (warrant valid, no irreversibility or external impact)

**Response:**
```json
{
  "verdict": "ACT",
  "tier": 5,
  "route": "warrant_check",
  "record_id": "eval-20260324-003-uuid",
  "reasons": [],
  "timestamp": "2026-03-24T14:32:10Z"
}
```

---

**Version**: 1.0
**Date**: 2026-03-24
**Status**: Specification
**References**: TrustContext, PassportStamp, TrustPassport (cognilateral_trust.core)
