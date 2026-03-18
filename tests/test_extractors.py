"""Sprint S1: Tests for cognilateral_trust.extractors module.

Covers confidence extraction from plain text, OpenAI-format logprob responses,
Anthropic-format responses, the unified extract_confidence() entry point, and
all edge cases (0%, 100%, negative, >1.0, missing signals).
"""

from __future__ import annotations

from typing import Any

import pytest

from cognilateral_trust.extractors import (
    extract_confidence,
    extract_confidence_from_anthropic_response,
    extract_confidence_from_openai_response,
    extract_confidence_from_text,
)


# ---------------------------------------------------------------------------
# extract_confidence_from_text — regex, no deps
# ---------------------------------------------------------------------------


class TestExtractFromTextPercentage:
    """'N% confident' and similar percentage patterns."""

    def test_eighty_five_percent_confident(self) -> None:
        assert extract_confidence_from_text("I'm 85% confident in this.") == pytest.approx(0.85)

    def test_low_percentage(self) -> None:
        assert extract_confidence_from_text("Only 20% confident here.") == pytest.approx(0.20)

    def test_hundred_percent(self) -> None:
        assert extract_confidence_from_text("I am 100% confident.") == pytest.approx(1.0)

    def test_zero_percent(self) -> None:
        assert extract_confidence_from_text("I'm 0% confident about this.") == pytest.approx(0.0)

    def test_percent_without_confident_word(self) -> None:
        # "85%" alone is not a confidence signal — must include context word
        result = extract_confidence_from_text("There are 85% of cases like this.")
        assert result is None

    def test_confidence_colon_decimal(self) -> None:
        assert extract_confidence_from_text("Confidence: 0.72") == pytest.approx(0.72)

    def test_confidence_colon_no_leading_zero(self) -> None:
        assert extract_confidence_from_text("Confidence: .91") == pytest.approx(0.91)

    def test_confidence_score_label(self) -> None:
        assert extract_confidence_from_text("confidence_score: 0.55") == pytest.approx(0.55)

    def test_certainty_label(self) -> None:
        assert extract_confidence_from_text("Certainty: 0.88") == pytest.approx(0.88)

    def test_confidence_embedded_in_text(self) -> None:
        text = "After analysis, Confidence: 0.67 seems appropriate."
        assert extract_confidence_from_text(text) == pytest.approx(0.67)


class TestExtractFromTextVerbalTiers:
    """Verbal confidence expressions: high/medium/low/very high/etc."""

    def test_high_confidence(self) -> None:
        result = extract_confidence_from_text("I have high confidence in this answer.")
        assert result is not None
        assert result >= 0.7

    def test_medium_confidence(self) -> None:
        result = extract_confidence_from_text("I have medium confidence in this.")
        assert result is not None
        assert 0.4 <= result <= 0.7

    def test_low_confidence(self) -> None:
        result = extract_confidence_from_text("This is low confidence — treat with caution.")
        assert result is not None
        assert result <= 0.4

    def test_very_high_confidence(self) -> None:
        result = extract_confidence_from_text("I'm very confident this is correct.")
        assert result is not None
        assert result >= 0.85

    def test_uncertain(self) -> None:
        result = extract_confidence_from_text("I'm quite uncertain about this.")
        assert result is not None
        assert result <= 0.35


class TestExtractFromTextNoSignal:
    """No recognizable confidence pattern — return None."""

    def test_plain_factual_text(self) -> None:
        assert extract_confidence_from_text("The capital of France is Paris.") is None

    def test_empty_string(self) -> None:
        assert extract_confidence_from_text("") is None

    def test_numbers_without_context(self) -> None:
        assert extract_confidence_from_text("There are 50 states in the US.") is None

    def test_percentage_unrelated(self) -> None:
        assert extract_confidence_from_text("CPU usage is at 73%.") is None


class TestExtractFromTextEdgeCases:
    """Edge cases: clipping, large percentages, negative."""

    def test_value_greater_than_one_clipped(self) -> None:
        # "Confidence: 1.5" — clips to 1.0
        result = extract_confidence_from_text("Confidence: 1.5")
        assert result == pytest.approx(1.0)

    def test_percentage_over_100_clipped(self) -> None:
        # "120% confident" — clips to 1.0
        result = extract_confidence_from_text("I'm 120% confident!")
        assert result == pytest.approx(1.0)

    def test_negative_decimal_clipped(self) -> None:
        # "Confidence: -0.2" — clips to 0.0
        result = extract_confidence_from_text("Confidence: -0.2")
        assert result == pytest.approx(0.0)

    def test_integer_one(self) -> None:
        assert extract_confidence_from_text("Confidence: 1") == pytest.approx(1.0)

    def test_integer_zero(self) -> None:
        assert extract_confidence_from_text("Confidence: 0") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# extract_confidence_from_openai_response — dict-based, no SDK deps
# ---------------------------------------------------------------------------


class TestExtractFromOpenAIResponse:
    """OpenAI-format response dict with logprobs."""

    def _make_openai_response(self, content: str, logprobs: list[dict] | None = None) -> dict[str, Any]:
        """Build a minimal OpenAI chat completion response dict."""
        choice: dict[str, Any] = {
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
            "index": 0,
        }
        if logprobs is not None:
            choice["logprobs"] = {"content": logprobs}
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [choice],
        }

    def test_confidence_in_content_text(self) -> None:
        resp = self._make_openai_response("The answer is 4.\nConfidence: 0.97")
        assert extract_confidence_from_openai_response(resp) == pytest.approx(0.97)

    def test_logprobs_average_top_token(self) -> None:
        """High logprob values (near 0.0 in log space) → high confidence."""
        import math

        logprobs = [
            {"token": "The", "logprob": math.log(0.95)},
            {"token": "answer", "logprob": math.log(0.92)},
            {"token": "is", "logprob": math.log(0.98)},
        ]
        resp = self._make_openai_response("The answer is 4.", logprobs=logprobs)
        result = extract_confidence_from_openai_response(resp)
        assert result is not None
        assert result > 0.7  # high logprobs → high confidence

    def test_logprobs_low_values(self) -> None:
        """Low logprob values → lower confidence."""
        import math

        logprobs = [
            {"token": "Maybe", "logprob": math.log(0.15)},
            {"token": "this", "logprob": math.log(0.20)},
        ]
        resp = self._make_openai_response("Maybe this is right.", logprobs=logprobs)
        result = extract_confidence_from_openai_response(resp)
        assert result is not None
        assert result < 0.5

    def test_no_logprobs_falls_back_to_text(self) -> None:
        resp = self._make_openai_response("The answer.\nConfidence: 0.60")
        assert extract_confidence_from_openai_response(resp) == pytest.approx(0.60)

    def test_no_logprobs_no_text_signal_returns_none(self) -> None:
        resp = self._make_openai_response("The capital of France is Paris.")
        assert extract_confidence_from_openai_response(resp) is None

    def test_empty_choices_returns_none(self) -> None:
        assert extract_confidence_from_openai_response({"choices": []}) is None

    def test_missing_choices_key_returns_none(self) -> None:
        assert extract_confidence_from_openai_response({}) is None

    def test_percentage_pattern_in_content(self) -> None:
        resp = self._make_openai_response("I'm 75% confident this is correct.")
        assert extract_confidence_from_openai_response(resp) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# extract_confidence_from_anthropic_response — dict-based, no SDK deps
# ---------------------------------------------------------------------------


class TestExtractFromAnthropicResponse:
    """Anthropic Messages API format dict."""

    def _make_anthropic_response(self, text: str, stop_reason: str = "end_turn") -> dict[str, Any]:
        """Build a minimal Anthropic messages response dict."""
        return {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": stop_reason,
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }

    def test_confidence_in_text_block(self) -> None:
        resp = self._make_anthropic_response("The answer is 42.\nConfidence: 0.91")
        assert extract_confidence_from_anthropic_response(resp) == pytest.approx(0.91)

    def test_percentage_pattern_in_text(self) -> None:
        resp = self._make_anthropic_response("I'm 68% confident about this.")
        assert extract_confidence_from_anthropic_response(resp) == pytest.approx(0.68)

    def test_verbal_high_confidence(self) -> None:
        resp = self._make_anthropic_response("I have high confidence this is the correct answer.")
        result = extract_confidence_from_anthropic_response(resp)
        assert result is not None
        assert result >= 0.7

    def test_no_confidence_signal_returns_none(self) -> None:
        resp = self._make_anthropic_response("The sky is blue.")
        assert extract_confidence_from_anthropic_response(resp) is None

    def test_empty_content_returns_none(self) -> None:
        resp = {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        assert extract_confidence_from_anthropic_response(resp) is None

    def test_missing_content_key_returns_none(self) -> None:
        assert extract_confidence_from_anthropic_response({}) is None

    def test_multiple_content_blocks_uses_first_text(self) -> None:
        resp = {
            "content": [
                {"type": "text", "text": "First block. Confidence: 0.80"},
                {"type": "text", "text": "Second block. Confidence: 0.30"},
            ]
        }
        # Should use the first text block's confidence
        result = extract_confidence_from_anthropic_response(resp)
        assert result == pytest.approx(0.80)

    def test_stop_reason_max_tokens_penalises_confidence(self) -> None:
        """Truncated responses get a modest confidence penalty."""
        resp = self._make_anthropic_response("The answer is correct.\nConfidence: 0.90", stop_reason="max_tokens")
        result = extract_confidence_from_anthropic_response(resp)
        assert result is not None
        assert result < 0.90  # truncation penalty applied


# ---------------------------------------------------------------------------
# extract_confidence — unified entry point with source detection
# ---------------------------------------------------------------------------


class TestUnifiedExtractConfidence:
    """extract_confidence(response, source='auto') — auto-detection."""

    def test_string_input_delegates_to_text(self) -> None:
        assert extract_confidence("Confidence: 0.77") == pytest.approx(0.77)

    def test_openai_dict_auto_detected(self) -> None:
        resp = {
            "id": "chatcmpl-x",
            "object": "chat.completion",
            "choices": [{"message": {"content": "Answer.\nConfidence: 0.82"}, "index": 0}],
        }
        assert extract_confidence(resp) == pytest.approx(0.82)

    def test_anthropic_dict_auto_detected(self) -> None:
        resp = {
            "id": "msg_x",
            "type": "message",
            "content": [{"type": "text", "text": "Answer.\nConfidence: 0.78"}],
        }
        assert extract_confidence(resp) == pytest.approx(0.78)

    def test_source_explicit_text(self) -> None:
        assert extract_confidence("Confidence: 0.55", source="text") == pytest.approx(0.55)

    def test_source_explicit_openai(self) -> None:
        resp = {
            "choices": [{"message": {"content": "Answer.\nConfidence: 0.65"}}],
        }
        assert extract_confidence(resp, source="openai") == pytest.approx(0.65)

    def test_source_explicit_anthropic(self) -> None:
        resp = {
            "content": [{"type": "text", "text": "Answer.\nConfidence: 0.70"}],
        }
        assert extract_confidence(resp, source="anthropic") == pytest.approx(0.70)

    def test_unknown_dict_returns_none(self) -> None:
        assert extract_confidence({"some_key": "some_value"}) is None

    def test_none_input_returns_none(self) -> None:
        assert extract_confidence(None) is None

    def test_integer_input_returns_none(self) -> None:
        assert extract_confidence(42) is None  # type: ignore[arg-type]

    def test_no_signal_text_returns_none(self) -> None:
        assert extract_confidence("The sky is blue.") is None
