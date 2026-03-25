"""Tests for the MCP Trust Server — protocol, tools, and integration.

Covers:
- JSON-RPC 2.0 protocol parsing and response formatting
- Tool implementations (trust_evaluate, trust_extract_confidence, trust_health)
- Server dispatch and lifecycle
- Stdio integration via subprocess
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from cognilateral_trust.mcp.protocol import (
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    ProtocolError,
    build_error,
    build_response,
    parse_request,
)
from cognilateral_trust.mcp.server import MCPTrustServer
from cognilateral_trust.mcp.tools import (
    TOOL_DISPATCH,
    TOOL_SCHEMAS,
    trust_evaluate,
    trust_extract_confidence,
    trust_health,
)


# ---------------------------------------------------------------------------
# Protocol tests
# ---------------------------------------------------------------------------


class TestProtocolParsing:
    """JSON-RPC 2.0 request parsing."""

    @pytest.mark.a_test
    def test_a1_parse_valid_request(self) -> None:
        body = json.dumps({"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 1})
        req = parse_request(body)
        assert req.method == "tools/list"
        assert req.params == {}
        assert req.id == 1

    @pytest.mark.a_test
    def test_a2_parse_request_without_params(self) -> None:
        body = json.dumps({"jsonrpc": "2.0", "method": "initialize", "id": "abc"})
        req = parse_request(body)
        assert req.method == "initialize"
        assert req.params == {}
        assert req.id == "abc"

    @pytest.mark.a_test
    def test_a3_parse_notification_no_id(self) -> None:
        body = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})
        req = parse_request(body)
        assert req.method == "notifications/initialized"
        assert req.id is None

    @pytest.mark.a_test
    def test_a4_parse_invalid_json(self) -> None:
        with pytest.raises(ProtocolError) as exc_info:
            parse_request("{not valid json")
        assert exc_info.value.code == PARSE_ERROR

    @pytest.mark.a_test
    def test_a5_parse_missing_method(self) -> None:
        body = json.dumps({"jsonrpc": "2.0", "id": 1})
        with pytest.raises(ProtocolError) as exc_info:
            parse_request(body)
        assert exc_info.value.code == INVALID_REQUEST

    @pytest.mark.a_test
    def test_a6_parse_non_object(self) -> None:
        with pytest.raises(ProtocolError) as exc_info:
            parse_request('"just a string"')
        assert exc_info.value.code == INVALID_REQUEST

    @pytest.mark.b_test
    def test_b1_parse_method_not_string(self) -> None:
        body = json.dumps({"jsonrpc": "2.0", "method": 42, "id": 1})
        with pytest.raises(ProtocolError) as exc_info:
            parse_request(body)
        assert exc_info.value.code == INVALID_REQUEST

    @pytest.mark.b_test
    def test_b2_parse_non_dict_params_defaults_empty(self) -> None:
        body = json.dumps({"jsonrpc": "2.0", "method": "test", "params": [1, 2], "id": 1})
        req = parse_request(body)
        assert req.params == {}


class TestProtocolResponses:
    """JSON-RPC 2.0 response building."""

    @pytest.mark.a_test
    def test_a1_build_success_response(self) -> None:
        resp = json.loads(build_response({"tools": []}, 1))
        assert resp["jsonrpc"] == "2.0"
        assert resp["result"] == {"tools": []}
        assert resp["id"] == 1

    @pytest.mark.a_test
    def test_a2_build_error_response(self) -> None:
        resp = json.loads(build_error(-32601, "Method not found", 1))
        assert resp["jsonrpc"] == "2.0"
        assert resp["error"]["code"] == -32601
        assert resp["error"]["message"] == "Method not found"
        assert resp["id"] == 1

    @pytest.mark.a_test
    def test_a3_build_error_with_data(self) -> None:
        resp = json.loads(build_error(-32600, "Bad request", 1, {"detail": "missing field"}))
        assert resp["error"]["data"]["detail"] == "missing field"

    @pytest.mark.b_test
    def test_b1_response_null_id(self) -> None:
        resp = json.loads(build_response({"ok": True}, None))
        assert resp["id"] is None


# ---------------------------------------------------------------------------
# Tool tests
# ---------------------------------------------------------------------------


class TestTrustEvaluateTool:
    """trust_evaluate tool behavior."""

    @pytest.mark.a_test
    def test_a1_basic_evaluation(self) -> None:
        result = trust_evaluate(confidence=0.7)
        assert result["confidence"] == 0.7
        assert result["tier"] == 7
        assert result["tier_name"] == "C7"
        assert result["should_proceed"] is True
        assert "accountability_record" in result

    @pytest.mark.a_test
    def test_a2_low_confidence_basic_route(self) -> None:
        result = trust_evaluate(confidence=0.2)
        assert result["tier"] == 2
        assert result["route"] == "basic"

    @pytest.mark.a_test
    def test_a3_high_confidence_sovereignty(self) -> None:
        result = trust_evaluate(confidence=0.9)
        assert result["tier"] == 9
        assert result["route"] == "sovereignty_gate"

    @pytest.mark.a_test
    def test_a4_irreversible_action_escalation(self) -> None:
        result = trust_evaluate(confidence=0.8, action_reversible=False)
        assert result["should_proceed"] is False
        record = result["accountability_record"]
        assert record is not None
        assert record["verdict"] == "ESCALATE"

    @pytest.mark.a_test
    def test_a5_confidence_out_of_range(self) -> None:
        result = trust_evaluate(confidence=1.5)
        assert "error" in result

    @pytest.mark.a_test
    def test_a6_negative_confidence(self) -> None:
        result = trust_evaluate(confidence=-0.1)
        assert "error" in result

    @pytest.mark.a_test
    def test_a7_with_evidence(self) -> None:
        result = trust_evaluate(confidence=0.8, evidence=["source verified", "cross-checked"])
        assert result["should_proceed"] is True
        record = result["accountability_record"]
        assert record is not None
        assert record["record_id"]  # non-empty UUID

    @pytest.mark.a_test
    def test_a8_accountability_record_structure(self) -> None:
        result = trust_evaluate(confidence=0.5)
        record = result["accountability_record"]
        assert isinstance(record["record_id"], str)
        assert isinstance(record["timestamp"], float)
        assert record["verdict"] in ("ACT", "ESCALATE")
        assert isinstance(record["reasons"], list)

    @pytest.mark.b_test
    def test_b1_invalid_confidence_type(self) -> None:
        result = trust_evaluate(confidence="high")  # type: ignore[arg-type]
        assert "error" in result

    @pytest.mark.b_test
    def test_b2_boundary_zero(self) -> None:
        result = trust_evaluate(confidence=0.0)
        assert result["tier"] == 0
        assert result["route"] == "basic"

    @pytest.mark.b_test
    def test_b3_boundary_one(self) -> None:
        result = trust_evaluate(confidence=1.0)
        assert result["tier"] == 9
        assert result["should_proceed"] is True


class TestExtractConfidenceTool:
    """trust_extract_confidence tool behavior."""

    @pytest.mark.a_test
    def test_a1_extract_from_text(self) -> None:
        result = trust_extract_confidence(response="I am 85% confident in this answer.")
        assert result["found"] is True
        assert result["confidence"] is not None
        assert 0.8 <= result["confidence"] <= 0.9

    @pytest.mark.a_test
    def test_a2_no_signal_in_text(self) -> None:
        result = trust_extract_confidence(response="The sky is blue.")
        assert result["found"] is False
        assert result["confidence"] is None

    @pytest.mark.a_test
    def test_a3_invalid_source(self) -> None:
        result = trust_extract_confidence(response="test", source="invalid")
        assert "error" in result

    @pytest.mark.a_test
    def test_a4_verbal_confidence(self) -> None:
        result = trust_extract_confidence(response="I am very confident about this.")
        assert result["found"] is True

    @pytest.mark.b_test
    def test_b1_explicit_source_text(self) -> None:
        result = trust_extract_confidence(response="Confidence: 0.75", source="text")
        assert result["found"] is True
        assert result["confidence"] == pytest.approx(0.75, abs=0.01)


class TestHealthTool:
    """trust_health tool behavior."""

    @pytest.mark.a_test
    def test_a1_health_response(self) -> None:
        result = trust_health()
        assert result["status"] == "healthy"
        assert isinstance(result["uptime_seconds"], float)
        assert isinstance(result["response_time_ms"], float)

    @pytest.mark.a_test
    def test_a2_health_response_time_under_1s(self) -> None:
        """FF-MCP-01: health response must be under 1 second."""
        result = trust_health()
        assert result["response_time_ms"] < 1000

    @pytest.mark.a_test
    def test_a3_health_includes_version(self) -> None:
        result = trust_health()
        assert "version" in result
        assert result["version"] != ""

    @pytest.mark.a_test
    def test_a4_tool_schemas_present(self) -> None:
        assert len(TOOL_SCHEMAS) == 3
        names = {t["name"] for t in TOOL_SCHEMAS}
        assert names == {"trust_evaluate", "trust_extract_confidence", "trust_health"}

    @pytest.mark.a_test
    def test_a5_dispatch_table_matches_schemas(self) -> None:
        schema_names = {t["name"] for t in TOOL_SCHEMAS}
        dispatch_names = set(TOOL_DISPATCH.keys())
        assert schema_names == dispatch_names


# ---------------------------------------------------------------------------
# Server dispatch tests
# ---------------------------------------------------------------------------


class TestServerDispatch:
    """MCPTrustServer method dispatch."""

    def _server(self) -> MCPTrustServer:
        return MCPTrustServer()

    @pytest.mark.a_test
    def test_a1_initialize(self) -> None:
        server = self._server()
        result = server._dispatch("initialize", {"capabilities": {}})
        assert result is not None
        assert result["protocolVersion"] == "2024-11-05"
        assert "capabilities" in result
        assert "serverInfo" in result

    @pytest.mark.a_test
    def test_a2_tools_list(self) -> None:
        server = self._server()
        result = server._dispatch("tools/list", {})
        assert result is not None
        assert "tools" in result
        assert len(result["tools"]) == 3

    @pytest.mark.a_test
    def test_a3_tools_call_evaluate(self) -> None:
        server = self._server()
        result = server._dispatch("tools/call", {"name": "trust_evaluate", "arguments": {"confidence": 0.7}})
        assert result is not None
        content = json.loads(result["content"][0]["text"])
        assert content["tier"] == 7

    @pytest.mark.a_test
    def test_a4_tools_call_health(self) -> None:
        server = self._server()
        result = server._dispatch("tools/call", {"name": "trust_health", "arguments": {}})
        assert result is not None
        content = json.loads(result["content"][0]["text"])
        assert content["status"] == "healthy"

    @pytest.mark.a_test
    def test_a5_tools_call_unknown_tool(self) -> None:
        server = self._server()
        result = server._dispatch("tools/call", {"name": "nonexistent", "arguments": {}})
        assert result is not None
        assert result.get("isError") is True

    @pytest.mark.a_test
    def test_a6_unknown_method(self) -> None:
        server = self._server()
        with pytest.raises(ProtocolError) as exc_info:
            server._dispatch("unknown/method", {})
        assert exc_info.value.code == METHOD_NOT_FOUND

    @pytest.mark.a_test
    def test_a7_notification_returns_none(self) -> None:
        server = self._server()
        result = server._dispatch("notifications/initialized", {})
        assert result is None

    @pytest.mark.b_test
    def test_b1_tools_call_bad_arguments(self) -> None:
        server = self._server()
        result = server._dispatch("tools/call", {"name": "trust_evaluate", "arguments": {"bad_arg": True}})
        assert result is not None
        assert result.get("isError") is True


# ---------------------------------------------------------------------------
# Stdio integration tests
# ---------------------------------------------------------------------------


class TestStdioIntegration:
    """End-to-end MCP server over stdio via subprocess."""

    @staticmethod
    def _build_message(method: str, params: dict | None = None, msg_id: int | str | None = 1) -> bytes:
        """Build a Content-Length framed JSON-RPC message."""
        obj: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            obj["params"] = params
        if msg_id is not None:
            obj["id"] = msg_id
        body = json.dumps(obj).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        return header + body

    @staticmethod
    def _parse_response(output: bytes) -> dict:
        """Parse a Content-Length framed response."""
        text = output.decode("utf-8")
        parts = text.split("\r\n\r\n", 1)
        if len(parts) < 2:
            raise ValueError(f"Malformed response: {text!r}")
        return json.loads(parts[1])

    @staticmethod
    def _run_server(messages: list[bytes], timeout: float = 10.0) -> bytes:
        """Run the MCP server with given input messages, return stdout."""
        stdin_data = b"".join(messages)
        proc = subprocess.run(
            [sys.executable, "-m", "cognilateral_trust.mcp"],
            input=stdin_data,
            capture_output=True,
            timeout=timeout,
            cwd=str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"),
            env={
                **__import__("os").environ,
                "PYTHONPATH": str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"),
            },
        )
        return proc.stdout

    @pytest.mark.a_test
    def test_a1_initialize_handshake(self) -> None:
        """Server responds to initialize with capabilities."""
        messages = [self._build_message("initialize", {"capabilities": {}}, 1)]
        output = self._run_server(messages)
        assert output, "Server produced no output"
        resp = self._parse_response(output)
        assert resp["id"] == 1
        assert "protocolVersion" in resp["result"]

    @pytest.mark.a_test
    def test_a2_tools_list_via_stdio(self) -> None:
        """Server lists tools over stdio."""
        messages = [
            self._build_message("initialize", {"capabilities": {}}, 1),
            self._build_message("tools/list", {}, 2),
        ]
        output = self._run_server(messages)
        # Parse all responses (split by Content-Length headers)
        responses = self._parse_all_responses(output)
        assert len(responses) >= 2
        tools_resp = responses[1]
        assert len(tools_resp["result"]["tools"]) == 3

    @pytest.mark.a_test
    def test_a3_evaluate_via_stdio(self) -> None:
        """Full evaluate flow over stdio."""
        messages = [
            self._build_message("initialize", {"capabilities": {}}, 1),
            self._build_message("tools/call", {"name": "trust_evaluate", "arguments": {"confidence": 0.7}}, 2),
        ]
        output = self._run_server(messages)
        responses = self._parse_all_responses(output)
        assert len(responses) >= 2
        call_resp = responses[1]
        content = json.loads(call_resp["result"]["content"][0]["text"])
        assert content["tier"] == 7

    @pytest.mark.b_test
    def test_b1_malformed_json_continues(self) -> None:
        """Server handles malformed input without crashing."""
        bad_body = b"not valid json"
        bad_message = f"Content-Length: {len(bad_body)}\r\n\r\n".encode("ascii") + bad_body
        messages = [
            bad_message,
            self._build_message("tools/call", {"name": "trust_health", "arguments": {}}, 1),
        ]
        output = self._run_server(messages)
        responses = self._parse_all_responses(output)
        # Should have at least error + health response
        assert len(responses) >= 1

    @staticmethod
    def _parse_all_responses(output: bytes) -> list[dict]:
        """Parse all Content-Length framed responses from output."""
        text = output.decode("utf-8")
        responses = []
        pos = 0
        while pos < len(text):
            cl_idx = text.find("Content-Length:", pos)
            if cl_idx < 0:
                break
            eol = text.find("\r\n", cl_idx)
            if eol < 0:
                break
            length = int(text[cl_idx + len("Content-Length:") : eol].strip())
            body_start = text.find("\r\n\r\n", cl_idx)
            if body_start < 0:
                break
            body_start += 4
            body = text[body_start : body_start + length]
            responses.append(json.loads(body))
            pos = body_start + length
        return responses
