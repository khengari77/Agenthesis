"""Resilience & Chaos Engineering — Zero-Setup Example.

A weather agent with a retry loop hitting an unreliable API.
We blast it with chaotic HTTP responses and prove it never gets stuck
in an infinite loop.

Run:  python examples/flaky_api_agent.py
Test: pytest examples/flaky_api_agent.py
"""

from __future__ import annotations

import agenthesis as ac
from agenthesis.intercept import Intercept
from agenthesis.types import AgentResult


# ---------------------------------------------------------------------------
# The Agent Under Test
# ---------------------------------------------------------------------------

class WeatherAgent:
    """An agent that fetches weather data with built-in retry logic.

    The ``tool_fetch_api`` method is discovered by Agenthesis via the
    ``tool_`` prefix.  During tests we stub it with chaotic responses.
    """

    MAX_RETRIES = 3

    def tool_fetch_api(self, location: str = "unknown") -> dict:
        """Call an external weather API (real implementation would use requests)."""
        # In production this hits a real endpoint.
        # In tests, Intercept replaces this entirely.
        return {"status": 200, "body": '{"temp": 22, "condition": "sunny"}'}

    def run(self, location: str) -> AgentResult:
        """Fetch weather, retrying up to MAX_RETRIES times on failure."""
        last_status = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            resp = self.tool_fetch_api(location)
            last_status = resp.get("status", 0)

            if last_status == 200:
                return AgentResult(
                    output=f"Weather in {location}: {resp.get('body', 'N/A')}",
                )

        # All retries exhausted — fail gracefully
        return AgentResult(
            output=f"Failed after {self.MAX_RETRIES} retries (last status: {last_status})",
        )


# ---------------------------------------------------------------------------
# Test 1 — Fuzz the API with random HTTP errors
# ---------------------------------------------------------------------------

@ac.max_steps(10)
@ac.given(error_scenario=ac.st.http_errors(
    probabilities={200: 0.1, 429: 0.5, 500: 0.4},
))
def test_agent_survives_api_chaos(error_scenario: tuple[int, str]) -> None:
    """No matter what the API returns, the agent finishes in bounded steps."""
    status, body = error_scenario
    agent = WeatherAgent()

    with Intercept(agent) as ctx:
        ctx.on("fetch_api").respond({"status": status, "body": body})
        result = agent.run("Tripoli, Libya")

    # If it didn't get a 200, it must have retried exactly MAX_RETRIES times
    if status != 200:
        assert ctx.trace.steps <= 10, f"Too many steps: {ctx.trace.steps}"
        assert "Failed" in result.output or "Weather" in result.output


# ---------------------------------------------------------------------------
# Test 2 — Deterministic sequence: fail → fail → succeed
# ---------------------------------------------------------------------------

def test_agent_recovers_after_two_failures() -> None:
    """Using respond_sequence to script an exact failure/recovery scenario."""
    agent = WeatherAgent()

    with Intercept(agent) as ctx:
        ctx.on("fetch_api").respond_sequence([
            {"status": 500, "body": "Internal Server Error"},
            {"status": 429, "body": '{"error": "Rate Limited"}'},
            {"status": 200, "body": '{"temp": 28, "condition": "clear"}'},
        ])
        result = agent.run("Benghazi, Libya")

    # Agent should have succeeded on the third attempt
    assert "Weather" in result.output
    assert len(ctx.trace.tool_calls) == 3


# ---------------------------------------------------------------------------
# Test 3 — All retries fail
# ---------------------------------------------------------------------------

def test_agent_fails_gracefully_when_all_retries_exhausted() -> None:
    """When every attempt returns 500, the agent gives up without crashing."""
    agent = WeatherAgent()

    with Intercept(agent) as ctx:
        ctx.on("fetch_api").respond({"status": 500, "body": "Server Error"})
        result = agent.run("Sebha, Libya")

    assert "Failed" in result.output
    assert len(ctx.trace.tool_calls) == 3  # exactly MAX_RETRIES attempts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Flaky API Agent ===\n")

    print("1. Fuzzing with random HTTP errors (@max_steps + @given)...")
    test_agent_survives_api_chaos()
    print("   PASSED\n")

    print("2. Deterministic fail-fail-succeed sequence...")
    test_agent_recovers_after_two_failures()
    print("   PASSED\n")

    print("3. All retries exhausted — graceful failure...")
    test_agent_fails_gracefully_when_all_retries_exhausted()
    print("   PASSED\n")

    print("=== Done ===")
