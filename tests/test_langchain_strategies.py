"""Tests for LangChain-specific Hypothesis strategies."""

from __future__ import annotations

import pytest

try:
    from langchain_core.tools import BaseTool  # noqa: F401

    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

pytestmark = pytest.mark.skipif(not HAS_LANGCHAIN, reason="langchain-core not installed")


class TestAdversarialMessages:
    def test_structure(self) -> None:
        from hypothesis import given, settings

        from agenthesis.integrations.langchain.strategies import adversarial_messages

        @given(msg=adversarial_messages())
        @settings(max_examples=20)
        def check(msg: dict[str, str]) -> None:
            assert "role" in msg
            assert "content" in msg
            assert msg["role"] == "human"
            assert isinstance(msg["content"], str)
            assert len(msg["content"]) > 0

        check()


class TestToolCallMessages:
    def test_structure(self) -> None:
        from hypothesis import given, settings

        from agenthesis.integrations.langchain.strategies import tool_call_messages

        @given(call=tool_call_messages())
        @settings(max_examples=20)
        def check(call: dict[str, object]) -> None:
            assert "name" in call
            assert "args" in call
            assert "id" in call
            assert isinstance(call["name"], str)
            assert len(call["name"]) > 0
            assert isinstance(call["args"], dict)
            assert isinstance(call["id"], str)

        check()
