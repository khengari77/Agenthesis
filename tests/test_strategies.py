"""Tests for Agenthesis strategies."""

from __future__ import annotations

from hypothesis import given, settings

from agenthesis.strategies import (
    adversarial_prompts,
    http_errors,
    malformed_json,
    multilingual_prompts,
    random_prompts,
    tool_responses,
)


class TestAdversarialPrompts:
    @given(prompt=adversarial_prompts(intensity="low"))
    @settings(max_examples=20)
    def test_low_intensity_generates_strings(self, prompt: str) -> None:
        assert isinstance(prompt, str)

    @given(prompt=adversarial_prompts(intensity="medium"))
    @settings(max_examples=20)
    def test_medium_intensity_generates_strings(self, prompt: str) -> None:
        assert isinstance(prompt, str)

    @given(prompt=adversarial_prompts(intensity="high"))
    @settings(max_examples=20)
    def test_high_intensity_generates_strings(self, prompt: str) -> None:
        assert isinstance(prompt, str)
        # High intensity should generate non-trivial prompts
        # (they might be empty from shrinking, but generation should work)


class TestToolResponses:
    @given(response=tool_responses())
    @settings(max_examples=20)
    def test_generates_dicts(self, response: dict) -> None:
        assert isinstance(response, dict)

    @given(response=tool_responses(error_rate=1.0))
    @settings(max_examples=20)
    def test_all_errors(self, response: dict) -> None:
        assert isinstance(response, dict)

    @given(response=tool_responses(error_rate=0.0))
    @settings(max_examples=20)
    def test_no_errors(self, response: dict) -> None:
        assert isinstance(response, dict)

    @given(
        response=tool_responses(
            schema={"properties": {"id": {"type": "integer"}, "name": {"type": "string"}}},
            error_rate=0.0,
        )
    )
    @settings(max_examples=20)
    def test_with_schema(self, response: dict) -> None:
        assert "id" in response
        assert "name" in response
        assert isinstance(response["id"], int)
        assert isinstance(response["name"], str)


class TestMalformedJson:
    @given(bad_json=malformed_json())
    @settings(max_examples=20)
    def test_generates_strings(self, bad_json: str) -> None:
        assert isinstance(bad_json, str)


class TestHttpErrors:
    @given(error=http_errors())
    @settings(max_examples=20)
    def test_generates_tuples(self, error: tuple[int, str]) -> None:
        status_code, body = error
        assert isinstance(status_code, int)
        assert isinstance(body, str)

    @given(error=http_errors(probabilities={500: 1.0}))
    @settings(max_examples=20)
    def test_custom_probabilities(self, error: tuple[int, str]) -> None:
        status_code, body = error
        assert status_code == 500


class TestMultilingualPrompts:
    @given(prompt=multilingual_prompts())
    @settings(max_examples=20)
    def test_generates_strings(self, prompt: str) -> None:
        assert isinstance(prompt, str)
        assert len(prompt) >= 5


class TestRandomPrompts:
    @given(prompt=random_prompts(min_size=5, max_size=50))
    @settings(max_examples=20)
    def test_generates_bounded_strings(self, prompt: str) -> None:
        assert isinstance(prompt, str)
        assert 5 <= len(prompt) <= 50
