"""Tests for the LangChain integration."""

from __future__ import annotations

from uuid import uuid4

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

    def test_invariant_violation_propagates_on_llm_end(self, handler, agent) -> None:
        from agentcheck.intercept import Intercept
        from agentcheck.types import InvariantViolation

        response = LLMResult(
            generations=[],
            llm_output={"token_usage": {"total_tokens": 10}},
        )

        with pytest.raises(InvariantViolation, match="max_llm_calls"), Intercept(agent) as ctx:
            ctx.set_llm_call_limit(1)
            handler.on_llm_end(response)  # call 1: ok
            handler.on_llm_end(response)  # call 2: exceeds limit

    def test_invariant_violation_propagates_on_tool_start(self, handler, agent) -> None:
        from agentcheck.intercept import Intercept
        from agentcheck.types import InvariantViolation

        with pytest.raises(InvariantViolation, match="max_steps"), Intercept(agent) as ctx:
            ctx.set_step_limit(1)
            handler.on_tool_start(serialized={}, input_str="a")  # step 1: ok
            handler.on_tool_start(serialized={}, input_str="b")  # step 2: exceeds

    def test_handler_reset_clears_state(self, handler, agent) -> None:
        from agentcheck.intercept import Intercept

        response = LLMResult(
            generations=[],
            llm_output={"token_usage": {"total_tokens": 10}},
        )

        with Intercept(agent):
            handler.on_llm_end(response)

        assert handler.get_trace()["llm_calls"] == 1
        handler.reset()
        assert handler.get_trace()["llm_calls"] == 0
        assert handler.get_trace()["total_tokens"] == 0
        assert handler.get_trace()["tool_calls"] == 0

    def test_on_tool_end_records_tool_call(self, handler, agent) -> None:
        from agentcheck.intercept import Intercept

        run_id = uuid4()

        with Intercept(agent) as ctx:
            handler.on_tool_start(
                serialized={"name": "search"}, input_str="query", run_id=run_id
            )
            handler.on_tool_end("result data", run_id=run_id)

        assert len(ctx.calls) == 1
        assert ctx.calls[0].name == "search"
        assert ctx.calls[0].result == "result data"

    def test_on_tool_error_cleans_pending(self, handler, agent) -> None:
        from agentcheck.intercept import Intercept

        run_id = uuid4()

        with Intercept(agent):
            handler.on_tool_start(
                serialized={"name": "failing_tool"}, input_str="bad input", run_id=run_id,
            )
            assert run_id in handler._pending_tools
            handler.on_tool_error(RuntimeError("boom"), run_id=run_id)
            assert run_id not in handler._pending_tools

    def test_on_tool_end_without_matching_start(self, handler, agent) -> None:
        from agentcheck.intercept import Intercept

        with Intercept(agent) as ctx:
            handler.on_tool_end("result", run_id=uuid4())

        assert len(ctx.calls) == 1
        assert ctx.calls[0].name == "unknown"
