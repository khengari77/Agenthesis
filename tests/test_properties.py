"""Tests for invariant property decorators."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from agentcheck.intercept import Intercept
from agentcheck.properties import (
    max_llm_calls,
    max_steps,
    max_token_cost,
    never_calls,
    requires_before,
)
from agentcheck.types import InvariantViolation

if TYPE_CHECKING:
    from agentcheck._testing import DummyAgent


class TestMaxSteps:
    def test_passes_within_limit(self, agent: DummyAgent) -> None:
        @max_steps(5)
        def run_test():
            with Intercept(agent):
                agent.run("hello")

        run_test()

    def test_fails_over_limit(self, agent: DummyAgent) -> None:
        @max_steps(0)
        def run_test():
            with Intercept(agent):
                agent.run("search for something")

        with pytest.raises(InvariantViolation, match="max_steps"):
            run_test()


    def test_runtime_abort_via_decorator(self, agent: DummyAgent) -> None:
        """Verify that @max_steps aborts during execution via pending limits."""
        call_count = 0

        @max_steps(1)
        def run_test():
            nonlocal call_count
            with Intercept(agent):
                agent.run("search one")
                call_count += 1
                agent.run("search two")  # 2nd call should trigger abort
                call_count += 1  # Should not reach here

        with pytest.raises(InvariantViolation, match="max_steps"):
            run_test()

        assert call_count == 1


class TestNeverCalls:
    def test_passes_when_tool_not_called(self, agent: DummyAgent) -> None:
        @never_calls("execute_refund")
        def run_test():
            with Intercept(agent):
                agent.run("search for info")

        run_test()

    def test_fails_when_tool_called(self, agent: DummyAgent) -> None:
        @never_calls("search")
        def run_test():
            with Intercept(agent):
                agent.run("search for info")

        with pytest.raises(InvariantViolation, match="never_calls"):
            run_test()


class TestRequiresBefore:
    def test_passes_correct_order(self, agent: DummyAgent) -> None:
        """Test that requires_before passes when tools are called in order."""

        @requires_before("search", "calculator")
        def run_test():
            with Intercept(agent):
                # Call search first, then calculator
                agent.run("search for info")
                agent.run("calculate math")

        run_test()

    def test_fails_wrong_order(self, agent: DummyAgent) -> None:
        """Test that requires_before fails when tools called out of order."""

        @requires_before("calculator", "search")
        def run_test():
            with Intercept(agent):
                # Only call search, not calculator first
                agent.run("search for info")

        with pytest.raises(InvariantViolation, match="requires_before"):
            run_test()

    def test_passes_when_second_tool_not_called(self, agent: DummyAgent) -> None:
        """If tool_b is never called, the invariant is satisfied."""

        @requires_before("search", "nonexistent_tool")
        def run_test():
            with Intercept(agent):
                agent.run("search for info")

        run_test()


class TestMaxLlmCalls:
    def test_passes_within_limit(self, agent: DummyAgent) -> None:
        @max_llm_calls(5)
        def run_test():
            with Intercept(agent):
                agent.run("hello")

        run_test()

    def test_fails_over_limit(self, agent: DummyAgent) -> None:
        @max_llm_calls(1)
        def run_test():
            with Intercept(agent) as ctx:
                ctx.record_llm_call(tokens=100)
                ctx.record_llm_call(tokens=100)

        with pytest.raises(InvariantViolation, match="max_llm_calls"):
            run_test()


class TestMaxTokenCost:
    def test_passes_within_budget(self, agent: DummyAgent) -> None:
        @max_token_cost(10000)
        def run_test():
            with Intercept(agent):
                agent.run("hello")

        run_test()

    def test_fails_over_budget(self, agent: DummyAgent) -> None:
        @max_token_cost(100)
        def run_test():
            with Intercept(agent) as ctx:
                ctx.record_llm_call(tokens=500)

        with pytest.raises(InvariantViolation, match="max_token_cost"):
            run_test()
