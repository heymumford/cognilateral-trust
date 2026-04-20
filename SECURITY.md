# Security Policy

## Reporting a Vulnerability

Email `eric@cognilateral.com` with the subject prefix `[SECURITY]`.

You will receive an acknowledgement within 48 hours. I am a single maintainer — expect honest, plain-language communication, not a ticketing system.

Please include:

- The version affected (see `pip show cognilateral-trust` or the `pyproject.toml` in your checkout)
- Steps to reproduce, or a minimal proof-of-concept
- Impact — what a malicious caller could actually do
- Suggested remediation if you have one

Do not open a public GitHub Issue or Discussion for anything that could be used to harm a downstream user of the library or the hosted API. Use email.

## Supported Versions

The latest minor version on PyPI is supported. Older versions receive security fixes only for critical vulnerabilities and only when a backport is straightforward.

| Version | Supported |
|---------|-----------|
| `1.4.x` | Yes |
| `< 1.4` | No — upgrade |

## Scope

In scope:

- The `cognilateral-trust` Python package on PyPI
- The hosted API at `https://cognilateral.com/api/v1/*`
- The MCP server entry point `cognilateral-trust-mcp`
- Framework integration modules under `cognilateral_trust.integrations.*`

Out of scope:

- Example code in `examples/` — these are illustrative and not production-grade
- Third-party dependencies — report upstream
- Social engineering of the maintainer

## Response Timeline

- Acknowledgement: within 48 hours
- Triage + impact assessment: within 7 days
- Fix or mitigation published: within 30 days for critical issues; longer for complex ones, with interim guidance

## Disclosure

I prefer coordinated disclosure. Once a fix is available on PyPI, I will publish an advisory via GitHub Security Advisories, credit you (if you want credit), and note the CVE if one is assigned.
