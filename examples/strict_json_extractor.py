"""Structured Output & Schema Verification — Zero-Setup Example.

An agent that extracts user profiles from messy text and outputs JSON.
We prove its output parser doesn't crash on RTL languages, CJK characters,
or massive context overloads.

Run:  python examples/strict_json_extractor.py
Test: pytest examples/strict_json_extractor.py

Requires: pip install agenthesis[json]   (for jsonschema)
"""

from __future__ import annotations

import json

from hypothesis import strategies as hst

import agenthesis as ac
from agenthesis.intercept import Intercept
from agenthesis.types import AgentResult

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

USER_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"],
}


# ---------------------------------------------------------------------------
# The Agent Under Test
# ---------------------------------------------------------------------------

class ExtractorAgent:
    """Extracts a user profile from messy text and returns strict JSON.

    Simulates a common LLM pattern: works fine on normal input but
    produces invalid types when the input gets weird or too long.
    """

    def run(self, messy_text: str) -> AgentResult:
        """Parse messy_text into a JSON user profile."""
        # Long or unusual inputs cause the "LLM" to hallucinate bad types
        if len(messy_text) > 100:
            # Bug: age becomes a string instead of an integer
            profile = json.dumps({"name": "Unknown", "age": "NaN"})
        else:
            profile = json.dumps({"name": "Valid User", "age": 30})

        return AgentResult(output=profile)


# ---------------------------------------------------------------------------
# Property-Based Test — output must always match the schema
# ---------------------------------------------------------------------------

def _make_schema_test():
    """Build the test function, handling missing jsonschema gracefully."""
    try:
        import jsonschema  # noqa: F401
    except ImportError:
        return None

    # Inner helper decorated with @output_matches_schema — validates
    # that the AgentResult output conforms to USER_SCHEMA.
    @ac.output_matches_schema(USER_SCHEMA)
    def _extract(text: str) -> AgentResult:
        agent = ExtractorAgent()
        with Intercept(agent):
            return agent.run(text)

    # Outer test decorated with @given — generates diverse inputs.
    @ac.given(text=hst.one_of(
        ac.st.multilingual_prompts(),
        ac.st.token_overflow(max_tokens=50),
    ))
    def test_extractor_always_produces_valid_schema(text: str) -> None:
        """Output must conform to USER_SCHEMA regardless of input script or size."""
        _extract(text)  # @output_matches_schema raises on violation

    return test_extractor_always_produces_valid_schema


# Create the test at module level so pytest discovers it
test_extractor_always_produces_valid_schema = _make_schema_test()


# ---------------------------------------------------------------------------
# Deterministic Tests (no jsonschema required)
# ---------------------------------------------------------------------------

def test_short_input_produces_valid_json() -> None:
    """Normal-length input should produce parseable JSON with correct types."""
    agent = ExtractorAgent()
    result = agent.run("Alice, 30 years old")
    parsed = json.loads(result.output)
    assert isinstance(parsed["name"], str)
    assert isinstance(parsed["age"], int)


def test_long_input_still_produces_parseable_json() -> None:
    """Even when the agent hallucinates bad types, the output is valid JSON."""
    agent = ExtractorAgent()
    result = agent.run("x" * 200)
    parsed = json.loads(result.output)
    assert "name" in parsed
    assert "age" in parsed
    # Note: age will be "NaN" (string) — a schema violation, but valid JSON


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Strict JSON Extractor ===\n")

    print("1. Short input — valid JSON with correct types...")
    test_short_input_produces_valid_json()
    print("   PASSED\n")

    print("2. Long input — valid JSON but wrong types...")
    test_long_input_still_produces_parseable_json()
    print("   PASSED\n")

    print("3. Schema validation under multilingual + overflow inputs...")
    if test_extractor_always_produces_valid_schema is None:
        print("   SKIPPED — install jsonschema: pip install agenthesis[json]\n")
    else:
        try:
            test_extractor_always_produces_valid_schema()
            print("   PASSED (no schema violations found)\n")
        except ac.InvariantViolation as exc:
            print(f"   FAILED — {exc}")
            print("   ^ This is expected! Long inputs cause age='NaN' (string, not int).\n")

    print("=== Done ===")
