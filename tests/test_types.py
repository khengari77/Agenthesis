"""Tests for core types."""

from __future__ import annotations

import pytest

from agentcheck.types import (
    AgentCheckError,
    AgentResult,
    AgentTrace,
    InterceptError,
    InvariantViolation,
    ToolCall,
)


class TestToolCall:
    def test_frozen(self) -> None:
        tc = ToolCall(
            name="search", arguments={"q": "test"}, result="ok",
            timestamp=1.0, was_intercepted=False,
        )
        with pytest.raises(AttributeError):
            tc.name = "other"  # type: ignore[misc]

    def test_equality(self) -> None:
        kwargs = dict(
            name="search", arguments={"q": "test"}, result="ok",
            timestamp=1.0, was_intercepted=False,
        )
        assert ToolCall(**kwargs) == ToolCall(**kwargs)

    def test_unhashable_with_dict_args(self) -> None:
        tc = ToolCall(
            name="search", arguments={"q": "test"}, result=None,
            timestamp=0.0, was_intercepted=True,
        )
        with pytest.raises(TypeError):
            hash(tc)


class TestAgentTrace:
    def test_defaults(self) -> None:
        trace = AgentTrace()
        assert trace.tool_calls == ()
        assert trace.llm_calls == 0
        assert trace.steps == 0

    def test_with_calls(self) -> None:
        tc = ToolCall(name="x", arguments={}, result=None, timestamp=0.0, was_intercepted=False)
        trace = AgentTrace(tool_calls=(tc,), steps=1)
        assert len(trace.tool_calls) == 1


class TestAgentResult:
    def test_defaults(self) -> None:
        result = AgentResult(output="hello")
        assert result.output == "hello"
        assert result.trace == AgentTrace()
        assert result.metadata == {}

    def test_frozen(self) -> None:
        result = AgentResult(output="hello")
        with pytest.raises(AttributeError):
            result.output = "other"  # type: ignore[misc]


class TestExceptions:
    def test_invariant_violation(self) -> None:
        v = InvariantViolation("max_steps", "took 10, max 5")
        assert "max_steps" in str(v)
        assert v.invariant == "max_steps"

    def test_hierarchy(self) -> None:
        assert issubclass(InvariantViolation, AgentCheckError)
        assert issubclass(InterceptError, AgentCheckError)
