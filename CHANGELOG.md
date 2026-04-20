# Changelog

All notable changes to `cognilateral-trust` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.4.0] - 2026-04-13

### Added
- **LangGraph TrustNode** — drop-in trust-gate node for LangGraph pipelines (`cognilateral_trust.integrations.langgraph.trust_gate_node`, `should_proceed`).
- **Consent presets** — named `ConsentProfile` presets for common deployment profiles (healthcare, finance, consumer).
- **Top-level re-exports** — network primitives (`decay`, `ContagionTracker`, `weighted_consensus`, `ProvenanceChain`, `sovereign_worker`), lifecycle extensions (`handoff_trust`, `kill_warrant`, `suspend_trust`, `wake_trust`), routing (`RouteRule`, `RouterPolicy`, `route_decision`, `router_trust_policy`), and `AccountabilityRecord` / `AccountabilityStore` are now importable directly from `cognilateral_trust`.

### Changed
- **`OpenClawTrustProvider` renamed to `TrustServiceProvider`** — neutral naming; any provider-protocol-compliant orchestrator can consume it. Old import path removed. (PR #13)
- **Input validation hardened** — invalid `confidence` values raise `ValueError` with the exact bound that was violated.

### Fixed
- `require_calibration` removed — the previous behaviour silently escalated calls when calibration data was absent, which was too conservative. Callers now control this via `ConsentProfile`.
- Phantom `verdict` field in bug-bash defect fixture removed; `trust-bench` CLI now correctly registers as an entry point.

### Documentation
- CHANGELOG 1.2.0 advertised `trust_decay` and `ContagionGraph` as public exports. The actual names shipped are `decay` (with helpers `exponential_decay` / `linear_decay`) in `cognilateral_trust.network.decay`, and `ContagionTracker` (with supporting `PropagationEntry` and `ContagionAlert`) in `cognilateral_trust.network.contagion`. Both are now re-exported at the package root. The CHANGELOG 1.2.0 entry is preserved as historical record.

## [1.3.0] - 2026-03-27

### Added
- **Open consent** (D5) — `ConsentProfile` lets the person downstream configure trust sensitivity. Consent can only make evaluation stricter, never weaker. D-05 welfare constraint always enforced.
- **Nutrition label** (D6) — `nutrition_label()` generates standard disclosure: "Trust evaluated. Confidence: 0.70 (C7). Verdict: ACT." `not_evaluated_label()` for skipped responses.
- **TrustBench calibration results** (D4) — published at `docs/CALIBRATION.md`. Overall ECE: 0.375 across 200 scenarios. Falsifiable.
- **TrustBench model comparison** — qwen3:8b (0.51), qwen3:32b (0.50), coder:32b (0.33). Model size does not improve calibration.

## [1.2.0] - 2026-03-25

### Added
- **Lifecycle decorators** (Phase 5: A2-A4, A6)
  - `@spawn_gate(min_tier=)` — trust-gated agent creation with D-05 welfare constraint
  - `@handoff_trust` — TrustPassport transfer across agent boundaries with stamp accumulation
  - `@kill_warrant` — obligation checking before termination, D-05 welfare hard block
  - `suspend_trust()` / `wake_trust()` — serialization for agent suspend/resume
  - `TerminationBlocked` exception for welfare-critical agents
- **Trust routing** (Phase 5: A5)
  - `RouteRule` frozen dataclass for tier-to-destination mapping
  - `RouterPolicy` immutable collection with YAML loading
  - `route_decision()` function for declarative trust-based dispatch
  - `@router_trust_policy` decorator
- **Trust network primitives** (Phase 6: B1-B5)
  - `trust_decay()` — exponential decay across agent handoffs
  - `ContagionGraph` — epistemic contagion detection in agent networks
  - `weighted_consensus()` — Bayesian trust-weighted multi-agent agreement
  - `@sovereign_worker` — leaf-level self-governance via D-07 gate
  - `ProvenanceChain` — SHA-256 linked accountability chain
- **TrustBench** (Phase 6: B3)
  - Epistemic honesty benchmark: "Did the agent know it didn't know?"
  - `generate_fingerprint_svg()` — radial calibration chart per model
  - Scoring methodology with reproducibility (±2% across runs)
  - Leaderboard with JSONL persistence
  - CLI: `trust-bench run`, `trust-bench fingerprint`
- **MCP Trust Server** (Phase 5: A1)
  - `cognilateral-trust-mcp` CLI entry point
  - Zero-install trust evaluation via MCP protocol
  - Tools: health, evaluate, claims, fidelity
- **OpenClaw TrustProvider** (Phase 5: A7)
  - Full provider spec compliance for OpenClaw protocol
- **Cognee integration** — trust scoring on knowledge graphs
- **Trust Protocol v1 spec** — single-page specification at `docs/specs/trust-protocol-v1.md`

### Changed
- `TrustContext` now includes `welfare_relevant` field
- `TrustPassport` gains `serialize()` / `restore()` for suspend/wake round-trips
- `PassportStamp` gains `serialize()` / `restore()` methods

## [1.1.0] - 2026-03-18

### Added
- **LangGraph TrustNode** — drop-in StateGraph node + conditional edge
- **CrewAI TrustTool** — Tool protocol for agent chains

## [1.0.0] - 2026-03-17

### Added
- **Claims extraction** — typed claim extraction (factual, causal, comparative, quantitative)
- **Fidelity verification** — word-overlap + NLI dual-signal fidelity checking
- **Epistemic firewall** — 7-tier mismatch detection with configurable gap threshold
- **Sovereignty gate** — D-07 governance with D-05 welfare hard constraint
- **`CalibratedTrustEngine`** — learning loop with JSONL persistence

### Changed
- Package promoted to v1.0.0 stable

## [0.5.0] - 2026-03-18

### Added
- Apache-2.0 LICENSE file (was declared in pyproject.toml but missing from repo)
- CHANGELOG.md

### Fixed
- PyPI compliance: LICENSE file now ships with source distribution

## [0.4.0] - 2026-03-17

### Added
- `CalibratedTrustEngine` — trust evaluations informed by historical prediction accuracy
- HTTP trust proxy with TypeScript type definitions
- Calibration loop demo (`examples/calibration_demo.py`)
- Load simulator for stress-testing trust evaluation throughput
- Cognee knowledge graph trust layer example

## [0.3.0] - 2026-03-17

### Added
- `PredictionStore` for calibration accuracy tracking over time
- Anthropic trust wrapper with confidence extraction
- GitHub Pages landing page

## [0.2.0] - 2026-03-17

### Added
- `trust-check` CLI entry point
- Mem0 trust provider example (age decay scoring)
- OpenAI trust wrapper example
- GitHub Actions trust gate example
- DSPy trust module example
- Property tests proving calibration model differentiates tiers
- CONTRIBUTING.md

## [0.1.0] - 2026-03-17

### Added
- Initial release: epistemic trust layer for AI agents
- `evaluate_trust()` — maps confidence (0.0-1.0) to C0-C9 tiers with routing decisions
- `AccountabilityRecord` and `AccountabilityStore` for immutable audit trails
- Zero dependencies, Python 3.11+
