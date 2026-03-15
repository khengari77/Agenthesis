"""Tests for invariant property decorators."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from agenthesis.intercept import Intercept
from agenthesis.properties import (
    max_llm_calls,
    max_steps,
    max_token_cost,
    never_calls,
    requires_before,
)
from agenthesis.types import InvariantViolation

if TYPE_CHECKING:
    from agenthesis._testing import DummyAgent


def _has_lark() -> bool:
    try:
        import lark  # noqa: F401

        return True
    except ImportError:
        return False


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


    def test_multi_agent_limits_apply_to_all(self, agent: DummyAgent) -> None:
        """Verify that pending limits apply to all Intercept contexts in the same test."""
        from agenthesis._testing import DummyAgent

        agent_b = DummyAgent()

        @max_steps(1)
        def run_test():
            # First agent — should have max_steps=1
            with Intercept(agent) as ctx_a:
                assert ctx_a._max_steps == 1
            # Second agent — should also have max_steps=1
            with Intercept(agent_b) as ctx_b:
                assert ctx_b._max_steps == 1

        run_test()


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


class TestMultiAgentNeverCalls:
    def test_catches_violation_in_first_agent(self, agent: DummyAgent) -> None:
        """Violation in first agent caught even when second agent is clean."""
        from agenthesis._testing import DummyAgent as DummyAgentFactory

        agent_b = DummyAgentFactory()

        @never_calls("search")
        def run_test():
            with Intercept(agent):
                agent.run("search for something")  # Violates!
            with Intercept(agent_b):
                agent_b.run("hello")  # Clean

        with pytest.raises(InvariantViolation, match="never_calls"):
            run_test()


class TestStackedDecorators:
    def test_stacked_max_steps_and_never_calls(self, agent: DummyAgent) -> None:
        """Both decorators validate when stacked."""

        @max_steps(10)
        @never_calls("search")
        def run_test():
            with Intercept(agent):
                agent.run("search for something")  # violates never_calls

        with pytest.raises(InvariantViolation, match="never_calls"):
            run_test()

    def test_stacked_outer_catches_violation(self, agent: DummyAgent) -> None:
        """Outer decorator catches violation even after inner passes."""

        @max_steps(0)
        @never_calls("nonexistent")
        def run_test():
            with Intercept(agent):
                agent.run("calculate math")  # 1 tool call step, violates max_steps(0)

        with pytest.raises(InvariantViolation, match="max_steps"):
            run_test()

    def test_stacked_multi_agent(self, agent: DummyAgent) -> None:
        """Stacked decorators + multi-agent: outer sees all intercepts."""
        from agenthesis._testing import DummyAgent as DummyAgentFactory

        agent_b = DummyAgentFactory()

        @max_steps(10)
        @never_calls("search")
        def run_test():
            with Intercept(agent):
                agent.run("search for info")  # violates never_calls
            with Intercept(agent_b):
                agent_b.run("hello")

        with pytest.raises(InvariantViolation, match="never_calls"):
            run_test()


class TestMultiAgentRequiresBefore:
    def test_catches_order_violation_in_earlier_agent(self, agent: DummyAgent) -> None:
        """Ordering violation in first agent caught even when second is clean."""
        from agenthesis._testing import DummyAgent as DummyAgentFactory

        agent_b = DummyAgentFactory()

        @requires_before("calculator", "search")
        def run_test():
            with Intercept(agent):
                agent.run("search for info")  # search without calculator first → violation
            with Intercept(agent_b):
                agent_b.run("hello")

        with pytest.raises(InvariantViolation, match="requires_before"):
            run_test()


class TestMarkdownJsonStripping:
    def test_json_with_fence(self) -> None:
        from agenthesis.properties import output_matches_schema

        schema = {"type": "object", "properties": {"key": {"type": "string"}}}

        @output_matches_schema(schema)
        def run_test():
            return '```json\n{"key":"val"}\n```'

        run_test()

    def test_json_with_bare_fence(self) -> None:
        from agenthesis.properties import output_matches_schema

        schema = {"type": "object", "properties": {"key": {"type": "string"}}}

        @output_matches_schema(schema)
        def run_test():
            return '```\n{"key":"val"}\n```'

        run_test()

    def test_plain_json_still_works(self) -> None:
        from agenthesis.properties import output_matches_schema

        schema = {"type": "object", "properties": {"key": {"type": "string"}}}

        @output_matches_schema(schema)
        def run_test():
            return '{"key":"val"}'

        run_test()

    def test_invalid_json_in_fence_still_fails(self) -> None:
        from agenthesis.properties import output_matches_schema

        schema = {"type": "object"}

        @output_matches_schema(schema)
        def run_test():
            return "```json\nnot valid json\n```"

        with pytest.raises(InvariantViolation, match="output_matches_schema"):
            run_test()


class TestOutputMatchesGrammar:
    def test_callable_parser_passes(self) -> None:
        from agenthesis.properties import output_matches_grammar

        def my_parser(text: str) -> None:
            if not text.startswith("OK"):
                raise ValueError("must start with OK")

        @output_matches_grammar(my_parser)
        def run_test():
            return "OK everything is fine"

        run_test()

    def test_callable_parser_fails(self) -> None:
        from agenthesis.properties import output_matches_grammar

        def my_parser(text: str) -> None:
            if not text.startswith("OK"):
                raise ValueError("must start with OK")

        @output_matches_grammar(my_parser)
        def run_test():
            return "ERROR something went wrong"

        with pytest.raises(InvariantViolation, match="output_matches_grammar"):
            run_test()

    @pytest.mark.skipif(not _has_lark(), reason="lark not installed")
    def test_lark_grammar_passes(self) -> None:
        from agenthesis.properties import lark_grammar, output_matches_grammar

        parser = lark_grammar(
            '''
            start: greeting NAME
            greeting: "hello" | "hi"
            NAME: /[a-z]+/
            %import common.WS
            %ignore WS
            ''',
        )

        @output_matches_grammar(parser)
        def run_test():
            return "hello world"

        run_test()

    @pytest.mark.skipif(not _has_lark(), reason="lark not installed")
    def test_lark_grammar_fails(self) -> None:
        from agenthesis.properties import lark_grammar, output_matches_grammar

        parser = lark_grammar(
            '''
            start: "hello" NAME
            NAME: /[a-z]+/
            %import common.WS
            %ignore WS
            ''',
        )

        @output_matches_grammar(parser)
        def run_test():
            return "goodbye world"

        with pytest.raises(InvariantViolation, match="output_matches_grammar"):
            run_test()
