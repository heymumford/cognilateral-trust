"""Entry point for the MCP Trust Server.

Run via:
    uvx cognilateral-trust-mcp
    uv run cognilateral-trust-mcp
    python -m cognilateral_trust.mcp
"""

from __future__ import annotations

import sys


def run() -> None:
    """Start the MCP Trust Server on stdio."""
    from cognilateral_trust.mcp.server import MCPTrustServer

    server = MCPTrustServer()

    try:
        server.run()
        sys.exit(0)
    except KeyboardInterrupt:
        sys.exit(0)
    except BrokenPipeError:
        sys.exit(0)


if __name__ == "__main__":
    run()
