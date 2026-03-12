"""Tests for the LangChain integration."""

from __future__ import annotations

import pytest

try:
    from langchain_core.outputs import LLMResult

    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

pytestmark = pytest.mark.skipif(not HAS_LANGCHAIN, reason="langchain-core not installed")


@pytest.fixture
def handler():
    from agentcheck.integrations.langchain import AgentCheckCallbackHandler

    return AgentCheckCallbackHandler()


class TestAgentCheckCallbackHandler:
    def test_callback_records_llm_call(self, handler, agent) -> None:
        from agentcheck.intercept import Intercept

        response = LLMResult(
            generations=[],
            llm_output={"token_usage": {"total_tokens": 42}},
        )

        with Intercept(agent) as ctx:
            handler.on_llm_end(response)

        assert ctx.trace.llm_calls == 1
        assert ctx.trace.total_tokens == 42

    def test_callback_no_intercept_no_crash(self, handler) -> None:
        response = LLMResult(generations=[], llm_output={})
        # Should not raise even without an active Intercept
        handler.on_llm_end(response)

    def test_on_tool_start_records_step(self, handler, agent) -> None:
        from agentcheck.intercept import Intercept

        with Intercept(agent) as ctx:
            handler.on_tool_start(serialized={}, input_str="test")

        assert ctx.trace.steps == 1
