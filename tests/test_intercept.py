"""Tests for the Intercept context manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from agenthesis.intercept import Intercept
from agenthesis.types import InterceptError, InvariantViolation

if TYPE_CHECKING:
    from agenthesis._testing import DummyAgent


class TestIntercept:
    def test_records_tool_calls(self, agent: DummyAgent) -> None:
        with Intercept(agent) as ctx:
            agent.run("search for something")

        assert len(ctx.trace.tool_calls) >= 1
        assert ctx.trace.tool_calls[0].name == "search"

    def test_respond_fixed_value(self, agent: DummyAgent) -> None:
        with Intercept(agent) as ctx:
            ctx.on("search").respond({"results": [], "query": "mocked"})
            agent.run("search for something")

        tc = ctx.trace.tool_calls[0]
        assert tc.was_intercepted is True
        assert tc.result == {"results": [], "query": "mocked"}

    def test_raise_error(self, agent: DummyAgent) -> None:
        with Intercept(agent) as ctx:
            ctx.on("search").raise_error(TimeoutError("slow"))

            with pytest.raises(TimeoutError, match="slow"):
                agent.run("search for something")

    def test_respond_with_callable(self, agent: DummyAgent) -> None:
        def custom_search(*args, **kwargs):
            return {"results": ["custom"], "query": "custom"}

        with Intercept(agent) as ctx:
            ctx.on("search").respond_with(custom_search)
            agent.run("search for info")

        assert ctx.trace.tool_calls[0].result["results"] == ["custom"]

    def test_respond_sequence(self, agent: DummyAgent) -> None:
        with Intercept(agent) as ctx:
            ctx.on("search").respond_sequence([
                {"results": ["first"]},
                {"results": ["second"]},
            ])
            # Call search twice
            agent.run("search one")
            agent.run("search two")

        calls = [tc for tc in ctx.trace.tool_calls if tc.name == "search"]
        assert calls[0].result == {"results": ["first"]}
        assert calls[1].result == {"results": ["second"]}

    def test_passthrough(self, agent: DummyAgent) -> None:
        with Intercept(agent) as ctx:
            ctx.on("calculator").passthrough()
            agent.run("calculate something")

        tc = ctx.trace.tool_calls[0]
        assert tc.was_intercepted is False
        assert tc.result["result"] == 4  # 2+2

    def test_restores_tools_on_exit(self, agent: DummyAgent) -> None:
        original_search = agent.get_tools()["search"]
        with Intercept(agent) as ctx:
            ctx.on("search").respond({"mocked": True})
        # After exit, original should be restored
        assert agent.get_tools()["search"] is original_search

    def test_no_tool_match_passes_through(self, agent: DummyAgent) -> None:
        with Intercept(agent) as ctx:
            agent.run("calculate math")

        assert len(ctx.trace.tool_calls) >= 1
        assert ctx.trace.tool_calls[0].name == "calculator"

    def test_explicit_tools_dict(self) -> None:
        call_log = []

        def my_tool(x: int) -> int:
            call_log.append(x)
            return x * 2

        tools = {"multiply": my_tool}
        with Intercept(tools=tools) as ctx:
            ctx.on("multiply").respond(42)
            result = tools["multiply"](5)

        assert result == 42
        assert len(call_log) == 0  # Original not called

    def test_trace_available_during_execution(self, agent: DummyAgent) -> None:
        with Intercept(agent) as ctx:
            agent.run("search something")
            assert len(ctx.calls) >= 1

    def test_record_step(self, agent: DummyAgent) -> None:
        with Intercept(agent) as ctx:
            ctx.record_step()
            ctx.record_step()
            ctx.record_step()

        assert ctx.trace.steps == 3

    def test_record_llm_call(self, agent: DummyAgent) -> None:
        with Intercept(agent) as ctx:
            ctx.record_llm_call(tokens=100)
            ctx.record_llm_call(tokens=200)

        assert ctx.trace.llm_calls == 2
        assert ctx.trace.total_tokens == 300


    def test_runtime_step_limit_aborts_immediately(self, agent: DummyAgent) -> None:
        """Verify that set_step_limit raises during execution, not post-mortem."""
        call_count = 0

        with pytest.raises(InvariantViolation, match="max_steps"), Intercept(agent) as ctx:
            ctx.set_step_limit(1)
            agent.run("search one")  # 1 tool call → ok
            call_count += 1
            agent.run("search two")  # 2nd tool call → should abort here
            call_count += 1  # Should never reach this

        assert call_count == 1

    def test_runtime_token_limit_aborts_immediately(self, agent: DummyAgent) -> None:
        with pytest.raises(InvariantViolation, match="max_token_cost"), Intercept(agent) as ctx:
            ctx.set_token_limit(50)
            ctx.record_llm_call(tokens=100)  # Exceeds 50 → abort

    def test_pending_limits_ingested_on_enter(self, agent: DummyAgent) -> None:
        """Verify that pending limits from decorators are ingested by __enter__."""
        from agenthesis._context import set_pending_limits

        set_pending_limits(max_steps=2)
        with Intercept(agent) as ctx:
            assert ctx._max_steps == 2

    def test_async_wrapper_with_stub(self) -> None:
        """Verify that async tools get async wrappers and stubs work."""
        import asyncio

        async def async_search(query: str) -> dict:
            return {"results": [query]}

        tools = {"search": async_search}
        with Intercept(tools=tools) as ctx:
            ctx.on("search").respond({"results": ["mocked"]})
            result = asyncio.run(tools["search"]("test"))

        assert result == {"results": ["mocked"]}
        assert ctx.trace.tool_calls[0].was_intercepted is True

    def test_no_dict_pollution_after_exit(self) -> None:
        """Verify that tool_* attribute interception doesn't pollute instance __dict__."""

        class AttrAgent:
            def tool_greet(self, name: str = "world") -> str:
                return f"hello {name}"

        agent = AttrAgent()
        assert "tool_greet" not in agent.__dict__

        with Intercept(agent) as ctx:
            ctx.on("greet").respond("mocked")
            # During interception, wrapper is in __dict__
            assert "tool_greet" in agent.__dict__

        # After exit, __dict__ should be clean
        assert "tool_greet" not in agent.__dict__
        # Class-level method should still work
        assert agent.tool_greet() == "hello world"

    def test_async_wrapper_passthrough(self) -> None:
        """Verify that async passthrough properly awaits the original."""
        import asyncio

        async def async_fetch(url: str) -> dict:
            return {"status": 200, "url": url}

        tools = {"fetch": async_fetch}
        with Intercept(tools=tools) as ctx:
            result = asyncio.run(tools["fetch"]("https://example.com"))

        assert result == {"status": 200, "url": "https://example.com"}
        assert ctx.trace.tool_calls[0].was_intercepted is False


class TestSlotsAgent:
    def test_slots_agent_intercept_and_restore(self) -> None:
        """Agent with __slots__ can be intercepted and restored."""

        class SlotsAgent:
            __slots__ = ("tool_greet",)

            def __init__(self):
                self.tool_greet = lambda name="world": f"hello {name}"

        agent = SlotsAgent()
        original = agent.tool_greet

        with Intercept(agent) as ctx:
            ctx.on("greet").respond("mocked")
            assert agent.tool_greet() == "mocked"

        # After exit, original should be restored
        assert agent.tool_greet is original
        assert agent.tool_greet() == "hello world"

    def test_slots_agent_missing_slot_raises(self) -> None:
        """Missing slot raises clear InterceptError."""

        class SlotsAgent:
            __slots__ = ()

            def __init__(self):
                pass

        # Add a class-level tool method that can't be overridden on instances
        SlotsAgent.tool_greet = lambda self, name="world": f"hello {name}"

        agent = SlotsAgent()
        with pytest.raises(InterceptError, match="__slots__"), Intercept(agent):
            pass


class TestToolStub:
    def test_respond_sequence_empty(self, agent: DummyAgent) -> None:
        with Intercept(agent) as ctx:
            ctx.on("search").respond_sequence([])

            with pytest.raises(InterceptError, match="empty sequence"):
                agent.run("search something")
