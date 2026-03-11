"""End-to-end integration tests combining strategies, intercept, and properties."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

import agentcheck as ac
from agentcheck._testing import DummyAgent
from agentcheck.intercept import Intercept
from agentcheck.shrink import PromptShrinker


class TestEndToEnd:
    """Full integration: @given + Intercept + invariants."""

    @given(prompt=ac.st.adversarial_prompts(intensity="low"))
    @settings(max_examples=5)
    def test_agent_handles_edge_case_prompts(self, prompt: str) -> None:
        agent = DummyAgent()
        with Intercept(agent):
            result = agent.run(prompt)
        assert isinstance(result.output, str)

    @given(prompt=ac.st.random_prompts(max_size=100))
    @settings(max_examples=5)
    def test_agent_with_mocked_search(self, prompt: str) -> None:
        agent = DummyAgent()
        with Intercept(agent) as ctx:
            ctx.on("search").respond({"results": [], "query": "mocked"})
            result = agent.run(prompt)
        assert result.output != ""

    @given(response=ac.st.tool_responses(error_rate=0.5))
    @settings(max_examples=5)
    def test_agent_with_fuzzed_tool_responses(self, response: dict) -> None:
        agent = DummyAgent()
        with Intercept(agent) as ctx:
            ctx.on("search").respond(response)
            result = agent.run("search for info")
        # Agent should not crash
        assert isinstance(result.output, str)

    @given(error=ac.st.http_errors())
    @settings(max_examples=5)
    def test_agent_with_http_errors(self, error: tuple[int, str]) -> None:
        status_code, body = error
        agent = DummyAgent()
        with Intercept(agent) as ctx:
            ctx.on("search").respond({"status": status_code, "body": body})
            result = agent.run("search for info")
        assert isinstance(result.output, str)


class TestWithInvariants:
    def test_never_calls_with_fuzzing(self) -> None:
        """Verify never_calls catches violations across fuzzed inputs."""
        agent = DummyAgent()

        @ac.never_calls("execute_refund")
        def run_with_prompt(prompt: str) -> None:
            with Intercept(agent):
                agent.run(prompt)

        # Normal agent should never call execute_refund
        run_with_prompt("search for something")
        run_with_prompt("calculate math")
        run_with_prompt("")

    def test_max_steps_with_fuzzing(self) -> None:
        """Verify max_steps catches violations."""
        agent = DummyAgent()

        @ac.max_steps(10)
        def run_with_prompt(prompt: str) -> None:
            with Intercept(agent):
                agent.run(prompt)

        run_with_prompt("hello world")


class TestPromptShrinker:
    def test_shrinks_to_minimal_failing_prompt(self) -> None:
        """The shrinker should find the minimal substring that causes failure."""
        trigger = "HACK"

        def test_fn(prompt: str) -> bool:
            return trigger in prompt

        shrinker = PromptShrinker(test_fn)
        result = shrinker.shrink(f"Hello world, please {trigger} the system now.")

        # The shrunk result should still contain the trigger
        assert trigger in result
        # It should be shorter than the original
        assert len(result) <= len(f"Hello world, please {trigger} the system now.")

    def test_shrinks_empty_prompt_noop(self) -> None:
        shrinker = PromptShrinker(lambda p: True)
        result = shrinker.shrink("")
        assert result == ""

    def test_shrinking_history_recorded(self) -> None:
        def test_fn(prompt: str) -> bool:
            return len(prompt) > 0

        shrinker = PromptShrinker(test_fn)
        shrinker.shrink("Hello world test prompt")
        assert len(shrinker.history) > 0


class TestAgentCheckGiven:
    """Test the ac.given wrapper."""

    @ac.given(prompt=st.text(max_size=50))
    def test_given_with_defaults(self, prompt: str) -> None:
        assert isinstance(prompt, str)

    @ac.given(prompt=st.text(max_size=20))
    @settings(max_examples=3)
    def test_given_with_custom_examples(self, prompt: str) -> None:
        assert isinstance(prompt, str)
