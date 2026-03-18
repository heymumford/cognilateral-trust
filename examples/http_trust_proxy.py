"""HTTP trust proxy — wrap any API call with confidence evaluation.

A lightweight reverse proxy that intercepts requests, evaluates trust
based on headers, and either forwards or blocks the request.

Usage:
    pip install cognilateral-trust uvicorn httpx

    # Start the proxy
    python http_trust_proxy.py --target http://localhost:8000 --port 9000

    # Calls with X-Confidence header get trust-evaluated
    curl -H "X-Confidence: 0.8" http://localhost:9000/api/deploy
    # → Forwards to target if confidence sufficient

    curl -H "X-Confidence: 0.3" -H "X-Irreversible: true" http://localhost:9000/api/deploy
    # → Returns 403 with ESCALATE reason

Works as a sidecar for any service. No code changes to the target.
"""

from __future__ import annotations

import json
from typing import Any

from cognilateral_trust import evaluate_trust


def evaluate_request_trust(headers: dict[str, str]) -> dict[str, Any]:
    """Evaluate trust from request headers.

    Headers:
        X-Confidence: float (0.0-1.0) — required
        X-Irreversible: bool — optional, default false
        X-External: bool — optional, default false
    """
    confidence_str = headers.get("x-confidence", headers.get("X-Confidence", ""))
    if not confidence_str:
        return {"error": "Missing X-Confidence header", "should_forward": True}

    try:
        confidence = float(confidence_str)
    except ValueError:
        return {"error": f"Invalid X-Confidence: {confidence_str}", "should_forward": True}

    is_reversible = headers.get("x-irreversible", headers.get("X-Irreversible", "false")).lower() != "true"
    touches_external = headers.get("x-external", headers.get("X-External", "false")).lower() == "true"

    result = evaluate_trust(
        confidence,
        is_reversible=is_reversible,
        touches_external=touches_external,
    )

    return {
        "should_forward": result.should_proceed,
        "verdict": "ACT" if result.should_proceed else "ESCALATE",
        "tier": result.tier.name,
        "route": result.route,
        "confidence": confidence,
        "reasons": list(result.accountability_record.reasons) if result.accountability_record else [],
        "record_id": result.accountability_record.record_id if result.accountability_record else "",
    }


# FastAPI proxy app (requires uvicorn + httpx)
try:
    from fastapi import FastAPI, Request, Response
    import httpx

    def create_proxy_app(target_url: str = "http://localhost:8000") -> FastAPI:
        app = FastAPI(title="Trust Proxy")
        client = httpx.AsyncClient(base_url=target_url)

        @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
        async def proxy(request: Request, path: str) -> Response:
            headers = dict(request.headers)
            trust = evaluate_request_trust(headers)

            if not trust.get("should_forward", True):
                return Response(
                    content=json.dumps(trust),
                    status_code=403,
                    media_type="application/json",
                    headers={"X-Trust-Verdict": trust["verdict"], "X-Trust-Tier": trust["tier"]},
                )

            # Forward to target
            body = await request.body()
            resp = await client.request(
                method=request.method,
                url=f"/{path}",
                headers={k: v for k, v in headers.items() if k.lower() not in ("host", "content-length")},
                content=body,
            )

            response_headers = dict(resp.headers)
            response_headers["X-Trust-Verdict"] = trust.get("verdict", "PASS")
            response_headers["X-Trust-Tier"] = trust.get("tier", "N/A")

            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=response_headers,
                media_type=resp.headers.get("content-type"),
            )

        return app

except ImportError:
    pass


if __name__ == "__main__":
    # Demo without proxy server — just evaluate headers
    print("=== High confidence deploy ===")
    r1 = evaluate_request_trust({"X-Confidence": "0.9"})
    print(f"Forward: {r1['should_forward']} | {r1['verdict']} | {r1['tier']}")

    print("\n=== Low confidence irreversible ===")
    r2 = evaluate_request_trust({"X-Confidence": "0.8", "X-Irreversible": "true"})
    print(f"Forward: {r2['should_forward']} | {r2['verdict']} | {r2['tier']}")
    if r2.get("reasons"):
        print(f"Reasons: {r2['reasons']}")

    print("\n=== No confidence header ===")
    r3 = evaluate_request_trust({})
    print(f"Forward: {r3['should_forward']} | {r3.get('error', 'ok')}")
