"""Tests for the Intercept context manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from agentcheck.intercept import Intercept
from agentcheck.types import InterceptError

if TYPE_CHECKING:
    from agentcheck._testing import DummyAgent


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


class TestToolStub:
    def test_respond_sequence_empty(self, agent: DummyAgent) -> None:
        with Intercept(agent) as ctx:
            ctx.on("search").respond_sequence([])

            with pytest.raises(InterceptError, match="empty sequence"):
                agent.run("search something")
