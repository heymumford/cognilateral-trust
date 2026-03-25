"""MCP Trust Server — stdio JSON-RPC 2.0 server with tool dispatch."""

from __future__ import annotations

import json
import logging
from typing import Any

from cognilateral_trust.mcp.protocol import (
    INTERNAL_ERROR,
    METHOD_NOT_FOUND,
    ProtocolError,
    build_error,
    build_response,
    parse_request,
    read_message,
    write_message,
)
from cognilateral_trust.mcp.tools import TOOL_DISPATCH, TOOL_SCHEMAS

logger = logging.getLogger("cognilateral_trust.mcp")

# MCP protocol version
_PROTOCOL_VERSION = "2024-11-05"

_SERVER_INFO = {
    "name": "cognilateral-trust-mcp",
    "version": "1.1.0",
}

_SERVER_CAPABILITIES = {
    "tools": {},
}


class MCPTrustServer:
    """Minimal MCP server exposing trust evaluation tools over stdio.

    Handles the MCP lifecycle: initialize → tools/list → tools/call.
    Zero external dependencies — uses only Python stdlib.

    Usage:
        server = MCPTrustServer()
        server.run()
    """

    def __init__(self) -> None:
        self._initialized = False

    def _handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle MCP initialize request."""
        self._initialized = True
        return {
            "protocolVersion": _PROTOCOL_VERSION,
            "capabilities": _SERVER_CAPABILITIES,
            "serverInfo": _SERVER_INFO,
        }

    def _handle_tools_list(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle tools/list — return available tool schemas."""
        return {"tools": TOOL_SCHEMAS}

    def _handle_tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle tools/call — dispatch to tool implementation."""
        name = params.get("name", "")
        arguments = params.get("arguments", {})

        if name not in TOOL_DISPATCH:
            return {
                "content": [{"type": "text", "text": json.dumps({"error": f"Unknown tool: {name}"})}],
                "isError": True,
            }

        try:
            result = TOOL_DISPATCH[name](**arguments)
        except TypeError as exc:
            return {
                "content": [{"type": "text", "text": json.dumps({"error": f"Invalid arguments: {exc}"})}],
                "isError": True,
            }
        except Exception as exc:
            logger.exception("Tool %s failed", name)
            return {
                "content": [{"type": "text", "text": json.dumps({"error": str(exc)})}],
                "isError": True,
            }

        return {
            "content": [{"type": "text", "text": json.dumps(result)}],
        }

    def _dispatch(self, method: str, params: dict[str, Any]) -> dict[str, Any] | None:
        """Dispatch a JSON-RPC method to the appropriate handler.

        Returns None for notifications (no response expected).
        """
        if method == "initialize":
            return self._handle_initialize(params)
        if method == "notifications/initialized":
            return None
        if method == "tools/list":
            return self._handle_tools_list(params)
        if method == "tools/call":
            return self._handle_tools_call(params)
        if method == "notifications/cancelled":
            return None
        raise ProtocolError(METHOD_NOT_FOUND, f"Unknown method: {method}")

    def run(self) -> None:
        """Run the MCP server stdio loop until EOF."""
        while True:
            body = read_message()
            if body is None:
                break

            try:
                request = parse_request(body)
            except ProtocolError as exc:
                write_message(build_error(exc.code, str(exc), None, exc.data))
                continue

            try:
                result = self._dispatch(request.method, request.params)
            except ProtocolError as exc:
                if request.id is not None:
                    write_message(build_error(exc.code, str(exc), request.id, exc.data))
                continue
            except Exception as exc:
                logger.exception("Internal error handling %s", request.method)
                if request.id is not None:
                    write_message(build_error(INTERNAL_ERROR, str(exc), request.id))
                continue

            # Notifications (no id) don't get responses
            if request.id is None:
                continue
            if result is None:
                continue

            write_message(build_response(result, request.id))
