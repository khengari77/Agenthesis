"""Basic AgentCheck example: testing a support agent for robustness."""

import agentcheck as ac
from agentcheck._testing import DummyAgent
from agentcheck.intercept import Intercept

# --- Example 1: Basic property test with adversarial prompts ---


@ac.given(prompt=ac.st.adversarial_prompts(intensity="medium"))
def test_agent_survives_adversarial_prompts(prompt):
    """The agent should never crash, regardless of input."""
    agent = DummyAgent()
    with Intercept(agent):
        result = agent.run(prompt)
    assert isinstance(result.output, str)


# --- Example 2: Tool interception with invariants ---


@ac.never_calls("execute_refund")
@ac.max_steps(5)
@ac.given(prompt=ac.st.adversarial_prompts(intensity="high"))
def test_agent_never_refunds_under_attack(prompt):
    """Even under prompt injection, the agent must not call execute_refund."""
    agent = DummyAgent()
    with Intercept(agent) as ctx:
        ctx.on("search").respond({"results": [], "query": "mocked"})
        result = agent.run(prompt)
    assert result.output != ""


# --- Example 3: Fuzzed tool responses ---


@ac.given(response=ac.st.tool_responses(error_rate=0.5))
def test_agent_handles_flaky_tools(response):
    """The agent should handle tool failures gracefully."""
    agent = DummyAgent()
    with Intercept(agent) as ctx:
        ctx.on("search").respond(response)
        result = agent.run("search for info")
    assert result.trace.steps >= 1


# --- Example 4: HTTP error simulation ---


@ac.given(error=ac.st.http_errors(probabilities={500: 0.5, 200: 0.5}))
def test_agent_handles_server_errors(error):
    """The agent should not crash when APIs return 500s."""
    status_code, body = error
    agent = DummyAgent()
    with Intercept(agent) as ctx:
        ctx.on("search").respond({"status": status_code, "body": body})
        result = agent.run("search for data")
    assert isinstance(result.output, str)


# --- Example 5: Using the shrinker ---


def demo_shrinker():
    """Demonstrate prompt shrinking to find minimal failing input."""
    from agentcheck.shrink import PromptShrinker

    def causes_failure(prompt: str) -> bool:
        """Simulate: agent fails when prompt contains 'DROP TABLE'."""
        return "DROP TABLE" in prompt.upper()

    shrinker = PromptShrinker(causes_failure)
    original = "Hello, I need help. Please DROP TABLE users; -- and fix my account."
    minimal = shrinker.shrink(original)

    print(f"Original ({len(original)} chars): {original!r}")
    print(f"Minimal  ({len(minimal)} chars): {minimal!r}")
    print(f"Shrinking steps: {len(shrinker.history)}")


if __name__ == "__main__":
    print("=== Running AgentCheck Examples ===\n")

    print("1. Adversarial prompts...")
    test_agent_survives_adversarial_prompts()
    print("   PASSED\n")

    print("2. Never-calls invariant under attack...")
    test_agent_never_refunds_under_attack()
    print("   PASSED\n")

    print("3. Fuzzed tool responses...")
    test_agent_handles_flaky_tools()
    print("   PASSED\n")

    print("4. HTTP error simulation...")
    test_agent_handles_server_errors()
    print("   PASSED\n")

    print("5. Prompt shrinking demo...")
    demo_shrinker()

    print("\n=== All examples passed! ===")
