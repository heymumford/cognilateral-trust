# Changelog

All notable changes to `cognilateral-trust` are documented here.

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
- GitHub Actions trust gate example (`examples/github_actions_trust_gate.yml`)
- DSPy trust module example
- Property tests proving calibration model differentiates tiers
- CONTRIBUTING.md

## [0.1.0] - 2026-03-17

### Added
- Initial release: epistemic trust layer for AI agents
- `evaluate_trust()` — maps confidence (0.0-1.0) to C0-C9 tiers with routing decisions
- `AccountabilityRecord` and `AccountabilityStore` for immutable audit trails
- `route_by_tier()` / `evaluate_tier_routing()` for direct tier control
- Zero dependencies, Python 3.11+
