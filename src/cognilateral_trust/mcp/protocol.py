"""JSON-RPC 2.0 over stdio with Content-Length framing — stdlib only.

Implements the base protocol used by MCP (Model Context Protocol) for
stdio transport. Messages are framed with Content-Length headers,
identical to the LSP base protocol.

Wire format:
    Content-Length: <byte_count>\r\n
    \r\n
    <json_body>
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class JSONRPCRequest:
    """Parsed JSON-RPC 2.0 request."""

    method: str
    params: dict[str, Any]
    id: str | int | None


class ProtocolError(Exception):
    """JSON-RPC protocol-level error."""

    def __init__(self, code: int, message: str, data: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data or {}


# JSON-RPC 2.0 error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def parse_request(body: str) -> JSONRPCRequest:
    """Parse a JSON-RPC 2.0 request body.

    Raises ProtocolError for malformed or invalid requests.
    """
    try:
        obj = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ProtocolError(PARSE_ERROR, f"Invalid JSON: {exc}") from exc

    if not isinstance(obj, dict):
        raise ProtocolError(INVALID_REQUEST, "Request must be a JSON object")

    method = obj.get("method")
    if not isinstance(method, str):
        raise ProtocolError(INVALID_REQUEST, "Missing or invalid 'method'")

    params = obj.get("params", {})
    if not isinstance(params, dict):
        params = {}

    return JSONRPCRequest(method=method, params=params, id=obj.get("id"))


def build_response(result: Any, request_id: str | int | None) -> str:
    """Build a JSON-RPC 2.0 success response."""
    return json.dumps({"jsonrpc": "2.0", "result": result, "id": request_id})


def build_error(code: int, message: str, request_id: str | int | None, data: dict[str, Any] | None = None) -> str:
    """Build a JSON-RPC 2.0 error response."""
    error: dict[str, Any] = {"code": code, "message": message}
    if data:
        error["data"] = data
    return json.dumps({"jsonrpc": "2.0", "error": error, "id": request_id})


def read_message() -> str | None:
    """Read one Content-Length framed message from stdin.

    Returns the message body as a string, or None on EOF.
    """
    content_length = -1

    while True:
        line = sys.stdin.readline()
        if not line:
            return None

        line = line.strip()
        if not line:
            break

        if line.lower().startswith("content-length:"):
            try:
                content_length = int(line.split(":", 1)[1].strip())
            except (ValueError, IndexError):
                continue

    if content_length < 0:
        return None

    body = sys.stdin.read(content_length)
    if len(body) < content_length:
        return None

    return body


def write_message(body: str) -> None:
    """Write one Content-Length framed message to stdout."""
    encoded = body.encode("utf-8")
    header = f"Content-Length: {len(encoded)}\r\n\r\n"
    sys.stdout.buffer.write(header.encode("ascii"))
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()
